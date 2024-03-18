import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseConfig, ModelOutputs
from ..DFaST import BaseSpatialModule, BaseTemporalModule, BaseFrequencyModule


class EEGNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 dropout=0.5,
                 frequency=128,
                 D=2,
                 num_kernels=8,
                 p1=4,
                 p2=8):
        super(EEGNetConfig, self).__init__(node_size=node_size,
                                        node_feature_size=node_feature_size,
                                        time_series_size=time_series_size,
                                        num_classes=num_classes,
                                        dropout=dropout)
        self.frequency = frequency
        self.num_kernels = num_kernels
        self.D = D
        self.p1 = p1
        self.p2 = p2


class EEGNet(nn.Module):
    """
    A Reproduction of EEGNet:
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013
    """
    def __init__(self, config: EEGNetConfig):
        super(EEGNet, self).__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.time_feature_size = config.num_kernels * self.output_node_size
        self.output_time_series_size = int(config.time_series_size//config.p1//config.p2)

        self.frequency_layer = BaseFrequencyModule(config)

        self.spatial_layer = BaseSpatialModule(config,
                                               input_node_size=self.input_node_size,
                                               output_node_size=self.output_node_size)

        self.temporal_layer = BaseTemporalModule(config, D=self.output_node_size)
        hidden_size = config.num_kernels * config.D * int(config.time_series_size // config.p1 // config.p2)
        # hidden_size = config.num_kernels * config.D

        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.frequency_layer(time_series)
        hidden_state = self.spatial_layer(hidden_state)
        hidden_state = self.temporal_layer(hidden_state)
        # hidden_state = torch.max(hidden_state, dim=-1)[0]  # ZuCo
        hidden_state = hidden_state.reshape(hidden_state.size(0), -1)
        logits = F.softmax(self.fc(hidden_state), dim=1)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss


class EEGNetP(nn.Module):
    def __init__(self, config: EEGNetConfig):
        super(EEGNetP, self).__init__()
        self.config = config
        self.input_node_size = config.node_size
        self.output_node_size = config.D
        self.time_feature_size = config.num_kernels * self.output_node_size
        self.output_time_series_size = int(config.time_series_size//config.p1//config.p2)

        self.frequency_layer = BaseFrequencyModule(config)

        self.spatial_layer = BaseSpatialModule(config,
                                               input_node_size=self.input_node_size,
                                               output_node_size=self.output_node_size)

        self.temporal_layer = BaseTemporalModule(config, D=self.output_node_size)
        hidden_size = config.num_kernels * config.D * int(config.time_series_size // config.p1 // config.p2)

        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.frequency_layer(time_series)
        hidden_state = self.spatial_layer(hidden_state)
        hidden_state = self.temporal_layer(hidden_state)
        hidden_state = hidden_state.reshape(hidden_state.size(0), -1)
        logits = F.softmax(self.fc(hidden_state), dim=1)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss
