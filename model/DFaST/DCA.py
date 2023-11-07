from math import sqrt

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.nn import TransformerEncoderLayer
from einops import rearrange
from ..BrainNetworkTransformer.dec import DEC


class FeatureExtractor(nn.Module):
    def __init__(self, config, layer_idx):
        super(FeatureExtractor, self).__init__()
        self.layer_idx = layer_idx
        self.layer_norm1 = nn.LayerNorm(config.time_series_size)

    def forward(self, x, node_feature):
        if self.layer_idx == 0:
            return node_feature
        else:
            x = self.layer_norm1(x)
            # feature = self.empirical(x)
            # feature = self.oas(x)
            feature = self.ledoit_wolf(x)
        return feature

    def empirical(self, x):
        c = torch.einsum('bne, bme -> bnm', x, x)
        c = c / (x.size(-1) - 1)
        c = torch.clamp(c, -1.0, 1.0)
        return c

    def ledoit_wolf(self, x):
        _, n_features, n_samples = x.shape

        x2 = x**2
        mu = 1
        emp_cov_trace = torch.sum(x2, dim=-1) / n_samples
        c0 = torch.einsum('bne, bme -> bnm', x, x)
        delta_ = torch.sum(c0 ** 2) / (n_samples**2)
        beta_ = torch.sum(torch.einsum('bne, bme -> bnm', x2, x2))
        beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
        delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu ** 2
        delta /= n_features
        beta = min(beta, delta)
        shrinkage = 0 if beta == 0 else beta / delta

        emp_cov = c0 / (n_samples-1)
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flatten()[:: n_features + 1] += shrinkage * mu
        shrunk_cov = torch.clamp(shrunk_cov, -1.0, 1.0)
        return shrunk_cov

    def oas(self, x):
        _, n_features, n_samples = x.shape
        c = torch.einsum('bne, bme -> bnm', x, x) / (n_samples-1)
        mu = 1

        alpha = torch.mean(c ** 2)
        num = alpha + mu ** 2
        den = (n_samples + 1.0) * (alpha - (mu ** 2) / n_features)
        shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
        shrunk_cov = (1.0 - shrinkage) * c
        shrunk_cov.flatten()[:: n_features + 1] += shrinkage * mu
        shrunk_cov = torch.clamp(shrunk_cov, -1.0, 1.0)
        return shrunk_cov


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self,
                 config,
                 node_feature_size,
                 input_node_size,
                 output_node_size,
                 pooling=True):
        super().__init__()
        # self.transformer = Layers.BrainCubeEncoderLayer(config,
        #                                                 layer_idx,
        #                                                 node_size=input_node_num,
        #                                                 node_feature_size=config.node_feature_size)
        self.transformer = TransformerEncoderLayer(d_model=node_feature_size,
                                                   nhead=config.num_heads,
                                                   dim_feedforward=config.dim_feedforward,
                                                   batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(node_feature_size *
                          input_node_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          node_feature_size * input_node_size),
            )
            self.dec = DEC(cluster_number=output_node_size,
                           hidden_dimension=node_feature_size,
                           encoder=self.encoder,
                           orthogonal=config.orthogonal,
                           freeze_center=config.freeze_center,
                           project_assignment=config.project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BNT(nn.Module):
    """BrainNetworkTransformer"""

    def __init__(self, config, layer_idx, node_feature_size, input_node_size, output_node_size):
        super().__init__()
        self.attention_layer = TransPoolingEncoder(config,
                                                   node_feature_size=node_feature_size,
                                                   input_node_size=input_node_size,
                                                   output_node_size=output_node_size,
                                                   pooling=config.distill[layer_idx])
        self.layer_norm = nn.LayerNorm(node_feature_size)

    def forward(self, node_feature: torch.tensor):
        loss = 0
        node_feature, assignment = self.attention_layer(node_feature)
        node_feature = self.layer_norm(node_feature)
        loss = loss + self.reg_ortho_loss(node_feature)
        return node_feature, loss

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def kl_loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all

    @staticmethod
    def reg_ortho_loss(encoding):
        B, N, E = encoding.shape
        covariance = torch.einsum('bne, bme -> bnm', encoding, encoding) / N
        loss = (covariance - torch.eye(N, device=covariance.device)).triu().norm(dim=(1, 2)).mean()
        return loss


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
        # kern_length = config.frequency // 2
        kern_length = 3

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

    # def forward(self, hidden_state):
    #     hidden_state = self.conv(hidden_state)
    #     B, F, N, E = hidden_state.shape
    #     H = self.num_heads if E % self.num_heads == 0 else 1
    #     M = self.output_node_size
    #     scale = 1. / sqrt(E)
    #
    #     query = self.query_conv(hidden_state).view(B, F, N, H, -1)
    #     value = self.value_conv(hidden_state).view(B, F, N, H, -1)
    #
    #     sub_graph_key = self.sub_graph_key_conv(hidden_state).view(B, F, M, H, -1)
    #     score = torch.einsum('b f n h e, b f m h e -> b f h n m', query, sub_graph_key)
    #     adjacency = self.sparse_adjacency(scale * score)
    #     adjacency = self.dropout(torch.softmax(adjacency, dim=-1))
    #
    #     graph = torch.einsum('b f h n m, b f n h e -> b f m h e', adjacency, value)
    #     graph = graph + sub_graph_key
    #     graph = rearrange(graph, 'b f m h e -> b f m (h e) ')
    #     graph = self.batch_norm(f.gelu(graph))
    #     if self.pooling is not None:
    #         graph = self.pooling(graph)
    #         graph = rearrange(graph, 'b f m t -> b (f m) t ').unsqueeze(2)
    #     # return graph, score
    #     return graph

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

