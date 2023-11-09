from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F


class LocalTemporalSlidingAttention(nn.Module):
    def __init__(self, config, d_model, pooling=True):
        super(LocalTemporalSlidingAttention, self).__init__()
        self.sliding_window_size = config.window_size
        kern_length = config.frequency // 4
        groups = self.d_hidden = d_model
        if config.d_hidden is not None:
            assert config.d_hidden % config.num_kernels == 0
            self.d_hidden = config.d_hidden
            groups = config.num_kernels
        self.scale = 1. / sqrt(d_model)
        self.query_conv = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                        nn.Conv2d(d_model, self.d_hidden, (1, kern_length),
                                                  padding=(0, (kern_length-1)//2), groups=groups))
        self.key_conv = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                      nn.Conv2d(d_model, self.d_hidden, (1, kern_length),
                                                padding=(0, (kern_length-1)//2), groups=groups))
        self.value_conv = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                        nn.Conv2d(d_model, self.d_hidden, (1, kern_length),
                                                  padding=(0, (kern_length-1)//2), groups=groups))
        self.out_conv1 = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                       nn.Conv2d(self.d_hidden, d_model, (1, kern_length),
                                                 padding=(0, (kern_length-1)//2), groups=groups))
        self.out_conv2 = nn.Sequential(nn.ZeroPad2d((0, 1 - kern_length % 2, 0, 0)),
                                       nn.Conv2d(d_model, d_model, (1, kern_length),
                                                 padding=(0, (kern_length-1)//2), groups=groups))
        self.batch_norm1 = nn.BatchNorm2d(d_model)
        self.batch_norm2 = nn.BatchNorm2d(d_model)

        self.pooling = nn.AvgPool2d((1, config.p2)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        query = self.query_conv(hidden_state)
        query = rearrange(query, 'b f m t -> b t (f m)')
        key = self.key_conv(hidden_state)
        key = rearrange(key, 'b f m t -> b t (f m)')
        value = self.value_conv(hidden_state)
        value = rearrange(value, 'b f m t -> b t (f m)')

        scores = torch.einsum('bse, bte -> bst', query, key)
        mask = self.slide_window_mask(value)
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
        A = self.dropout(torch.softmax(scores * self.scale, dim=-1))

        V = torch.einsum('bst, bte -> bse', A, value)
        V = rearrange(V, 'b t (f m) -> b f m t', f=self.d_hidden)

        hidden_state = self.batch_norm1(hidden_state + self.out_conv1(V))
        hidden_state = self.batch_norm2(self.out_conv2(hidden_state))
        if self.pooling is not None:
            hidden_state = self.pooling(hidden_state)
        hidden_state = rearrange(hidden_state, 'b f m t -> b (f m) t')
        return hidden_state

    def slide_window_mask(self, time_feature):
        B, T, E = time_feature.shape
        size = (T, T)
        attn_mask = torch.triu(torch.ones(size), diagonal=self.sliding_window_size) - (
                    torch.triu(torch.ones(size), diagonal=-self.sliding_window_size) - 1)
        attn_mask = attn_mask == 1
        return attn_mask.to(time_feature.device)


class BaseTemporalModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013

    """
    def __init__(self, config, D=5, pooling=True):
        super(BaseTemporalModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.separable_conv = nn.Conv2d(config.num_kernels * D, config.num_kernels * D, (1, int(kern_length*0.5)),
                                        padding=(0, int(kern_length*0.5)//2-1), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels * D, False)
        self.pooling = nn.AvgPool2d((1, config.p2)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state)
        hidden_state = F.elu(self.batch_norm(self.separable_conv(hidden_state)))
        if self.pooling is not None:
            hidden_state = self.pooling(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state.squeeze(2)
        return hidden_state
