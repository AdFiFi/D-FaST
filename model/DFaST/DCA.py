from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as f


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3, reduction=2, init_weight=True):
        super(InceptionBlock, self).__init__()
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels,
                                     kernel_size=(1, 2 * i + 1),
                                     stride=(1, reduction),
                                     padding=(0, i),
                                     groups=in_channels))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class DynamicConnectogramAttention(nn.Module):
    def __init__(self, config, reduction=1, dropout=0.1, pooling=True):
        super(DynamicConnectogramAttention, self).__init__()
        self.sparsity = config.sparsity
        self.num_heads = config.num_heads
        self.output_node_size = config.D
        kern_length = config.frequency // 2
        # kern_length = 3 # ZuCo

        self.conv = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                  nn.Conv2d(1, config.num_kernels, (1, kern_length), padding=(0, (kern_length-1)//2),
                                            bias=False),
                                  nn.BatchNorm2d(config.num_kernels, False)
                                  )
        self.sub_graph_query_conv = nn.Conv2d(config.num_kernels, config.num_kernels * config.D,
                                              kernel_size=(config.node_size, 1),
                                              stride=(1, reduction),
                                              groups=config.num_kernels)
        self.key_conv = InceptionBlock(config.num_kernels, config.num_kernels, reduction=reduction)
        self.value_conv = InceptionBlock(config.num_kernels, config.num_kernels, reduction=reduction)
        self.pooling = nn.AvgPool2d((1, config.p1)) if pooling else None
        self.batch_norm = nn.BatchNorm2d(config.num_kernels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_state, return_adjacency=False):
        if hidden_state.shape[1] == 1:
            hidden_state = self.conv(hidden_state)
        B, F, N, E = hidden_state.shape
        H = self.num_heads if E % self.num_heads == 0 else 1
        M = self.output_node_size
        scale = 1. / sqrt(E)

        key = self.key_conv(hidden_state).view(B, F, N, H, -1)
        value = self.value_conv(hidden_state).view(B, F, N, H, -1)

        sub_graph_query = self.sub_graph_query_conv(hidden_state).view(B, F, M, H, -1)
        score = torch.einsum('b f m h e, b f n h e -> b f h m n', sub_graph_query, key)
        adjacency = self.sparse_adjacency(scale * score)
        adjacency = self.dropout(torch.softmax(adjacency, dim=-1))

        graph = torch.einsum('b f h m n, b f n h e -> b f m h e', adjacency, value)
        graph = graph + sub_graph_query
        graph = rearrange(graph, 'b f m h e -> b f m (h e) ')
        graph = self.batch_norm(f.gelu(graph))
        if self.pooling is not None:
            graph = self.pooling(graph)
            graph = rearrange(graph, 'b f m t -> b (f m) t ').unsqueeze(2)
        if return_adjacency:
            return graph, adjacency
        else:
            return graph

    def sparse_adjacency(self, space_feature):
        E = space_feature.size(-1)
        mask = space_feature < torch.topk(space_feature, int(self.sparsity*E), dim=-1)[0][..., -1, None]
        # space_feature[mask] = 0
        space_feature.masked_fill_(mask, torch.finfo(space_feature.dtype).min)
        return space_feature

    def get_adjacency(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        B, F, N, E = hidden_state.shape
        H = self.num_heads if E % self.num_heads == 0 else 1
        M = self.output_node_size
        scale = 1. / sqrt(E)

        key = self.key_conv(hidden_state).view(B, F, N, H, -1)

        sub_graph_query = self.sub_graph_query_conv(hidden_state).view(B, F, M, H, -1)
        score = torch.einsum('b f m h e, b f n h e -> b f h m n', sub_graph_query, key)
        adjacency = self.sparse_adjacency(scale * score)
        adjacency = torch.softmax(adjacency, dim=-1)
        return adjacency


class BaseSpatialModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013

    """
    def __init__(self, config, input_node_size=30, output_node_size=5, pooling=True):
        super(BaseSpatialModule, self).__init__()
        self.config = config
        self.dw_conv = nn.Conv2d(config.num_kernels, config.num_kernels * output_node_size, (input_node_size, 1),
                                 bias=False, groups=config.num_kernels)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * output_node_size, False)
        self.pooling = nn.AvgPool2d((1, config.p1)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        spatial_feature = f.elu(self.batch_norm(self.dw_conv(hidden_state)))
        if self.pooling is not None:
            spatial_feature = self.pooling(spatial_feature)
        spatial_feature = self.dropout(spatial_feature)
        return spatial_feature

