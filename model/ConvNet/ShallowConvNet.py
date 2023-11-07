import torch
import torch.nn as nn
from ..base import BaseConfig, ModelOutputs


class ShallowConvNetConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 time_series_size,
                 num_classes,
                 num_kernels=40):
        super(ShallowConvNetConfig, self).__init__(node_size=node_size,
                                                   node_feature_size=node_feature_size,
                                                   time_series_size=time_series_size,
                                                   num_classes=num_classes)
        self.num_kernels = num_kernels


class ShallowConvNet(nn.Module):
    """
    A Reproduction of DeepConvNet:
    Deep learning with convolutional neural networks for EEG decoding and visualization

    R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, et al.

    Human brain mapping 2017 Vol. 38 Issue 11 Pages 5391-5420
    """
    def __init__(self, config: ShallowConvNetConfig):
        super(ShallowConvNet, self).__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(1, config.num_kernels, (1, 25)),
            nn.Conv2d(config.num_kernels, config.num_kernels, (config.node_size, 1), bias=False),
            nn.BatchNorm2d(config.num_kernels)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()
        hidden_size = (((config.time_series_size - 25 + 1) - 75) // 15 + 1) * config.num_kernels
        # hidden_size = config.num_kernels  # ZuCo
        self.classifier = nn.Linear(hidden_size, config.num_classes)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing,
                                           weight=torch.tensor(config.class_weight))

    def forward(self, time_series, labels):
        time_series = time_series.unsqueeze(1)
        hidden_state = self.features(time_series)
        hidden_state = torch.square(hidden_state)
        hidden_state = self.avgpool(hidden_state)
        hidden_state = torch.clip(torch.log(hidden_state), min=1e-7, max=1e7)
        hidden_state = self.dropout(hidden_state)
        # hidden_state = hidden_state.mean(-1)
        features = torch.flatten(hidden_state, 1)  # 使用卷积网络代替全连接层进行分类, 因此需要返回x和卷积层个数
        logits = self.classifier(features)
        loss = self.loss_fn(logits, labels)
        if self.config.dict_output:
            return ModelOutputs(logits=logits,
                                loss=loss)
        else:
            return logits, loss
