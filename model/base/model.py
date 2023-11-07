import torch

from abc import abstractmethod
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, time_series: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:
        pass

    def loss(self):
        pass


class ModelOutputs:
    def __init__(self,
                 logits=None,
                 loss=None,
                 hidden_state=None):
        self.logits = logits
        self.loss = loss
        self.hidden_state = hidden_state
