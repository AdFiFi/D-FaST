import torch
import torch.nn as nn
import numpy as np
import einops
import ptwt


class TimeFrequencyRepresentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.BatchNorm2d(config.channel)

    def forward(self, time_series):
        widths = np.arange(4, 31)
        hidden_state, _ = ptwt.cwt(time_series, widths, "cmor1-1", sampling_period=1/200)
        # hidden_state, _ = ptwt.cwt(time_series, widths, "gaus1", sampling_period=1/200)
        hidden_state = torch.abs(hidden_state).float()
        hidden_state = hidden_state.transpose(0, 1)
        # hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state.transpose(1, 2)
        return hidden_state


class RegionAttentionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.delta = config.delta
        self.loss_fn = nn.ReLU()

    def forward(self, attention_weight):
        return torch.mean(self.loss_fn(self.delta - (attention_weight.max(-1)[0] - attention_weight.min(-1)[0])))


class FeatureExtractionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convs = nn.Conv2d(config.node_size, config.node_size, (3, 3), padding=(1, 1), groups=config.node_size)

    def forward(self, hidden_state):
        return self.convs(hidden_state)


class SelfAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.k
        self.attention = nn.Sequential(nn.AvgPool2d((config.channel, config.time_series_size)),
                                       nn.Tanh())
        self.region_attention_loss = RegionAttentionLoss(config)

    def forward(self, hidden_state):
        B, N, F, T = hidden_state.shape
        attention_weight = self.attention(hidden_state)
        attention_weight = attention_weight.reshape(B, -1, self.k)
        scale = 1 / attention_weight.sum(-1)
        hidden_state = hidden_state.reshape(B, -1, self.k, F * T)
        hidden_state = torch.einsum("bmk, bmke -> bme", attention_weight, hidden_state)
        hidden_state = torch.einsum("bme, bm -> bme", hidden_state, scale)
        loss = self.region_attention_loss(attention_weight)
        return hidden_state, attention_weight.min(-1)[0], loss


class RegionalAttentionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_layer = nn.Linear(config.channel * config.time_series_size, 1)

    def forward(self, hidden_state, min_global_attention):
        attention_weight = torch.softmax(self.fc_layer(hidden_state), dim=1)
        scale = torch.einsum("bm, bm -> bm", attention_weight.squeeze(-1), min_global_attention)
        hidden_state = torch.einsum("bm, bme -> be", scale, hidden_state)
        hidden_state = torch.einsum("be, b -> be", hidden_state, 1 / scale.sum(1))
        return hidden_state
