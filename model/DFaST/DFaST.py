import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .DCA import DynamicConnectogramAttention, BaseSpatialModule
from .LTSA import LocalTemporalSlidingAttention, BaseTemporalModule
from .MVA import MultiViewAttention, BaseFrequencyModule
from ..base import BaseConfig, ModelOutputs


class DFaSTConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 frequency=200,
                 window_size=50,
                 window_stride=5,
                 dynamic_stride=1,
                 dynamic_length=600,
                 k=5,
                 sparsity=0.7,
                 num_kernels=3,
                 D=5,
                 p1=4,
                 p2=8,
                 d_model=15,
                 d_hidden=None,
                 num_heads=4,
                 dim_feedforward=1024,
                 num_spatial_layers=2,
                 num_node_temporal_layers=2,
                 num_graph_temporal_layers=2,
                 attention_depth=3,
                 activation='gelu',
                 dropout=0.1,
                 distill=(False, True),
                 output_attention=False,
                 initializer=None,
                 label_smoothing=0,
                 lam1=0.15,
                 lam2=5e-7,
                 ):
        super(DFaSTConfig, self).__init__(d_model=d_model,
                                          d_hidden=d_hidden,
                                          node_size=node_size,
                                          node_feature_size=node_feature_size,
                                          time_series_size=time_series_size,
                                          dim_feedforward=dim_feedforward,
                                          activation=activation,
                                          dropout=dropout,
                                          num_classes=num_classes,
                                          initializer=initializer,
                                          label_smoothing=label_smoothing)
        self.k = k
        self.sparsity = sparsity
        self.multiple_periods = time_series_size >= 300
        # self.multiple_periods = False
        self.frequency = frequency
        self.num_kernels = num_kernels
        self.num_heads = num_heads
        self.num_spatial_layers = num_spatial_layers
        self.num_graph_temporal_layers = num_graph_temporal_layers
        self.num_node_temporal_layers = num_node_temporal_layers
        self.attention_depth = attention_depth
        self.output_attention = output_attention
        self.distill = distill
        self.lam1 = lam1
        self.lam2 = lam2
        self.window_size = window_size
        self.window_stride = window_stride
        self.dynamic_stride = dynamic_stride
        self.dynamic_length = dynamic_length
        self.aggregate = 'flatten'
        # self.aggregate = 'mean'
        # self.aggregate = 'attention'
        self.reg_lambda = 1e-4
        self.orthogonal = True
        self.freeze_center = True
        self.project_assignment = True
        self.p1 = p1
        self.p2 = p2
        self.D = D
        self.fs_fusion_type = 'add'
        # self.fs_fusion_type = 'concat'
        self.qkv_projector = 'linear'
        # self.qkv_projector = 'conv'


class DFaST(nn.Module):
    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.output_time_feature_size = config.num_kernels * self.output_node_size
        # self.output_time_series_size = int(config.time_series_size//config.p1//config.p2) + 1
        self.output_time_series_size = int(config.time_series_size // config.p1 // config.p2)
        self.fs_fusion_type = config.fs_fusion_type
        self.dropout = nn.Dropout(config.dropout)
        if self.fs_fusion_type == 'concat':
            self.config.num_kernels = self.config.num_kernels // 2
        self.mva = MultiViewAttention(config)
        self.dca = DynamicConnectogramAttention(config)
        if self.fs_fusion_type == 'concat':
            self.config.num_kernels = self.config.num_kernels * 2
        self.ltsa = LocalTemporalSlidingAttention(self.config, d_model=self.output_time_feature_size)

        self.batch_norm1 = nn.BatchNorm2d(self.config.num_kernels * self.output_node_size)

    def forward(self, hidden_state):
        hidden_state = hidden_state.unsqueeze(1)
        hidden_state1 = self.mva(hidden_state)
        hidden_state2 = self.dca(hidden_state)
        hidden_state = self.dropout(self.frequency_spatial_fusion(hidden_state1, hidden_state2))
        hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.ltsa(hidden_state)
        return hidden_state

    def frequency_spatial_fusion(self, frequency_feature, spatial_feature):
        if self.fs_fusion_type == 'add':
            return frequency_feature + spatial_feature
        elif self.fs_fusion_type == 'concat':
            frequency_feature = rearrange(frequency_feature, 'b (f n) m t -> b f (n m) t',
                                          f=self.config.num_kernels // 2)
            spatial_feature = rearrange(spatial_feature, 'b (f n) m t -> b f (n m) t', f=self.config.num_kernels // 2)
            fusion = torch.concat([frequency_feature, spatial_feature], dim=1)
            fusion = rearrange(fusion, 'b f (n m) t -> b (f n) m t', m=1)
            return fusion


class DFaSTOnlySpatial(nn.Module):
    output_node_feature_size: int

    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D // 2 // 2 // 2
        self.output_node_feature_size = config.D // 2 // 2 * config.num_kernels
        # self.output_node_feature_size = config.D // 2
        self.dca1 = DynamicConnectogramAttention(config)

        config2 = copy.deepcopy(config)
        config2.node_size = config2.D
        config2.D = config2.D // 2
        self.dca2 = DynamicConnectogramAttention(config2)

        config3 = copy.deepcopy(config2)
        config3.node_size = config2.D
        config3.D = config3.D // 2
        self.dca3 = DynamicConnectogramAttention(config3)

        config4 = copy.deepcopy(config3)
        config4.node_size = config4.D
        config4.D = config4.D // 2
        self.dca4 = DynamicConnectogramAttention(config4)

    def forward(self, hidden_state):
        hidden_state = hidden_state.unsqueeze(1)
        hidden_state, adjacency = self.dca1(hidden_state, return_adjacency=True)

        hidden_state = rearrange(hidden_state, "b (f n) m t -> b f n (m t)", f=self.config.num_kernels)
        hidden_state, adjacency = self.dca2(hidden_state, return_adjacency=True)

        hidden_state = rearrange(hidden_state, "b (f n) m t -> b f n (m t)", f=self.config.num_kernels)
        hidden_state, adjacency = self.dca3(hidden_state, return_adjacency=True)

        hidden_state = rearrange(hidden_state, "b (f n) m t -> b f n (m t)", f=self.config.num_kernels)
        hidden_state, adjacency = self.dca4(hidden_state, return_adjacency=True)

        adjacency = adjacency.mean(2)
        adjacency = rearrange(adjacency, "b f n m -> b n (f m)")
        # adjacency = rearrange(adjacency, "b f n m -> b n f m")
        return adjacency


class DFaSTForClassification(nn.Module):
    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.d_fast = DFaST(config)
        self.dropout = nn.Dropout(config.dropout)
        if config.aggregate == 'flatten':
            hidden_size = self.d_fast.output_time_series_size * self.d_fast.output_time_feature_size
        elif config.aggregate == 'mean':
            hidden_size = self.d_fast.output_time_feature_size
        elif config.aggregate == 'attention':
            self.attention = nn.Linear(self.d_fast.output_time_series_size, 1, bias=False)
            hidden_size = self.d_fast.output_time_feature_size
        else:
            assert ""
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, node_feature=None, labels=None):
        features = self.d_fast(time_series)
        features = self.aggregate(features)
        logits = F.softmax(self.fc(features), dim=1)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss,
                                hidden_state=features)
        else:
            return logits, loss

    def aggregate(self, features):
        if self.config.aggregate == 'flatten':
            N, _, _ = features.shape
            features = features.reshape(N, -1)
        elif self.config.aggregate == 'mean':
            features = features.mean(dim=-1)
        elif self.config.aggregate == 'attention':
            features = self.attention(features).squeeze(-1)
        return features


class DFaSTOnlySpatialForClassification(nn.Module):
    def __init__(self, config: DFaSTConfig):
        super().__init__()
        self.config = config
        self.d_fast = DFaSTOnlySpatial(config)
        self.dropout = nn.Dropout(config.dropout)
        if config.aggregate == 'flatten':
            hidden_size = self.d_fast.output_node_size * self.d_fast.output_node_feature_size
        elif config.aggregate == 'mean':
            hidden_size = self.d_fast.output_node_feature_size
        elif config.aggregate == 'attention':
            self.attention = nn.Linear(self.d_fast.output_node_feature_size, 1, bias=False)
            hidden_size = self.d_fast.output_node_feature_size
        else:
            assert ""
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, node_feature=None, labels=None):
        features = self.d_fast(time_series)
        features = self.aggregate(features)
        logits = F.softmax(self.fc(features), dim=1)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss,
                                hidden_state=features)
        else:
            return logits, loss

    def aggregate(self, features):
        if self.config.aggregate == 'flatten':
            N, _, _ = features.shape
            features = features.reshape(N, -1)
        elif self.config.aggregate == 'mean':
            features = features.mean(dim=-1)
        elif self.config.aggregate == 'attention':
            features = self.attention(features).squeeze(-1)
        return features
