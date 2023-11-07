import torch
import torch.nn as nn
from ..base import BaseConfig, ModelOutputs


class DeepConvNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 num_kernels=25):
        super(DeepConvNetConfig, self).__init__(node_size=node_size,
                                                node_feature_size=node_feature_size,
                                                time_series_size=time_series_size,
                                                num_classes=num_classes)
        self.num_kernels = num_kernels


class DeepConvNet(nn.Module):
    """
    A Reproduction of DeepConvNet:
    Deep learning with convolutional neural networks for EEG decoding and visualization

    R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, et al.

    Human brain mapping 2017 Vol. 38 Issue 11 Pages 5391-5420
    """
    def __init__(self, config: DeepConvNetConfig):
        super(DeepConvNet, self).__init__()
        self.config = config
        self.block1 = nn.Sequential(
            nn.Conv2d(1, config.num_kernels, (1, 5)),
            nn.Conv2d(config.num_kernels, config.num_kernels, (config.node_size, 1), bias=False),
            nn.BatchNorm2d(config.num_kernels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((config.time_series_size - 5 + 1) - 2) // 2 + 1)
        self.block2 = nn.Sequential(
            nn.Conv2d(config.num_kernels, config.num_kernels*2, (1, 5)),
            nn.BatchNorm2d(config.num_kernels*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1)
        self.block3 = nn.Sequential(
            nn.Conv2d(config.num_kernels*2, config.num_kernels*4, (1, 5)),
            nn.BatchNorm2d(config.num_kernels*4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(0.5)
        )
        hidden_size = (((hidden_size - 5 + 1) - 2) // 2 + 1) * config.num_kernels * 4
        # hidden_size = config.num_kernels * 4  # ZuCo
        self.classifier = nn.Linear(hidden_size, config.num_classes)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.block1(time_series)
        hidden_state = self.block2(hidden_state)
        hidden_state = self.block3(hidden_state)
        # hidden_state = hidden_state.mean(-1)
        features = torch.flatten(hidden_state, 1)
        logits = self.classifier(features)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss
