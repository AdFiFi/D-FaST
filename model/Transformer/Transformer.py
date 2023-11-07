import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from ..base import BaseConfig, ModelOutputs


class TransformerConfig(BaseConfig):
    def __init__(self,
                 node_size,
                 node_feature_size,
                 readout='concat',
                 num_layers=2,
                 num_classes=2,
                 dim_feedforward=1024):
        super(TransformerConfig, self).__init__(node_size=node_size,
                                                node_feature_size=node_feature_size,
                                                num_classes=num_classes,
                                                dim_feedforward=dim_feedforward)
        self.readout = readout
        self.num_layers = num_layers


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = config.readout
        self.node_size = config.node_size

        for _ in range(config.num_layers):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=config.node_feature_size, nhead=4,
                                        dim_feedforward=config.dim_feedforward,
                                        batch_first=True)
            )

        final_dim = config.node_feature_size

        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(
                nn.Linear(config.node_feature_size, 8),
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_size

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(config.node_feature_size)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, config.num_classes)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, node_feature, labels):
        bz, _, _, = node_feature.shape

        for atten in self.attention_list:
            node_feature = atten(node_feature)

        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)
            node_feature = node_feature.reshape((bz, -1))

        elif self.readout == "mean":
            node_feature = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            node_feature, _ = torch.max(node_feature, dim=1)
        elif self.readout == "sum":
            node_feature = torch.sum(node_feature, dim=1)
            node_feature = self.norm(node_feature)

        logits = self.fc(node_feature)
        loss = self.loss_fn(logits, labels)
        return ModelOutputs(logits=logits,
                            loss=loss)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
