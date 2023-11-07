from ..base import BaseConfig, ModelOutputs
from .EEGChannlNetLayers import *


class EEGChannelNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes):
        super(EEGChannelNetConfig, self).__init__(node_size=node_size,
                                                  node_feature_size=node_feature_size,
                                                  time_series_size=time_series_size,
                                                  num_classes=num_classes)


class EEGChannelNet(nn.Module):
    """
    A Reproduction of EEGChannelNet:
    Decoding brain representations by multimodal learning of neural activity and visual features

    S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt and M. Shah

    IEEE Transactions on Pattern Analysis and Machine Intelligence 2020 Vol. 43 Issue 11 Pages 3833-3849

    """
    def __init__(self, config: EEGChannelNetConfig):
        super(EEGChannelNet, self).__init__()
        self.config = config
        self.temporal_block = TemporalBlock()
        self.spatial_block = SpatialBlock()
        self.residual_block = ResidualBlock()
        inp_channels = 200
        num_kernels = 50
        # hidden_size = (((((config.time_series_size // 2) // 2) // 2 + 1) // 2 + 1) // 2 + 1 - 2) * \
        #               (((((config.node_size // 2) // 2) // 2) // 2 + 1) // 2 + 1 - 2) * num_kernels
        hidden_size = (((config.time_series_size // 2 + 1) // 4 + 1) // 4 - 2) * num_kernels
        self.output_layer = nn.Sequential(nn.Conv2d(inp_channels, num_kernels, (3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)),
                                          nn.Flatten(),
                                          nn.Linear(hidden_size, hidden_size * 2))
        # self.output_layer = nn.Sequential(nn.Conv2d(inp_channels, num_kernels, (1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0)),
        #                                   nn.Flatten(),
        #                                   nn.Dropout(config.dropout),
        #                                   nn.Linear(hidden_size, hidden_size * 2))
        self.classifier = nn.Linear(hidden_size * 2, config.num_classes)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.temporal_block(time_series)
        hidden_state = self.spatial_block(hidden_state)
        hidden_state = self.residual_block(hidden_state)
        features = self.output_layer(hidden_state)
        logits = self.classifier(features)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss
