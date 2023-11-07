from ..base import BaseConfig, ModelOutputs
from .RACNNLayers import *


class RACNNConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 delta=0.15,
                 k=6,
                 channel=27,
                 lam=0.05):
        super(RACNNConfig, self).__init__(node_size=node_size,
                                          node_feature_size=node_feature_size,
                                          time_series_size=time_series_size,
                                          num_classes=num_classes)
        self.delta = delta
        self.k = k
        self.channel = channel
        self.lam = lam


class RACNN(nn.Module):
    """
    A Reproduction of RACNN:
    Learning regional attention convolutional neural network for motion intention recognition based on EEG data

    Z. Fang, W. Wang, S. Ren, J. Wang, W. Shi, X. Liang, et al.

    Proceedings of the Twenty-Ninth International Conference on International Joint Conferences
        on Artificial Intelligence 2021

    Pages: 1570-1576
    """
    def __init__(self, config: RACNNConfig):
        super(RACNN, self).__init__()
        self.config = config
        self.time_frequency_representation = TimeFrequencyRepresentation(config)
        self.feature_extraction = FeatureExtractionModule(config)
        self.self_attention = SelfAttentionModule(config)
        self.regional_attention = RegionalAttentionModule(config)
        hidden_size = config.channel * config.time_series_size
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 1024),
                                        nn.Dropout(p=0.6),
                                        nn.Linear(1024, config.num_classes))

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        hidden_state = self.time_frequency_representation(time_series)
        hidden_state = self.feature_extraction(hidden_state)
        hidden_state, min_global_attention, region_attention_loss = self.self_attention(hidden_state)
        hidden_state = self.regional_attention(hidden_state, min_global_attention)
        logits = self.classifier(hidden_state)
        loss = self.loss_fn(logits, labels) + self.config.lam * region_attention_loss
        # loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss
