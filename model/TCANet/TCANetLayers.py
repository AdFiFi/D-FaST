import torch
from torch import nn
from torch.nn import functional as F

from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class GlobalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.num_kernels, kernel_size=(1, 20), stride=(1, 1), padding=0)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(config.num_kernels, config.num_kernels, kernel_size=(config.node_size, 1), stride=(1, 1), padding=0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        nn.init.constant_(self.conv2.bias, 0)

        self.conv_nonlin = Expression(square)

        self.bn2 = nn.BatchNorm2d(config.num_kernels, momentum=0.1, affine=True)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        self.pool_nonlin = Expression(safe_log)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        x = self.conv_nonlin(x)
        x = F.avg_pool2d(x, (1, 60), stride=(1, 40))
        x = self.pool_nonlin(x)
        return x


class LocalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(1, config.num_kernels, kernel_size=(1, 15), padding=0, stride=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(config.num_kernels, config.num_kernels, kernel_size=(config.node_size, 1), padding=0, stride=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        nn.init.constant_(self.conv2.bias, 0)

        self.conv12_nonlin = Expression(square)

        self.bn2 = nn.BatchNorm2d(config.num_kernels, momentum=0.1, affine=True)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        self.conv3 = nn.Conv2d(config.num_kernels, config.num_kernels, kernel_size=(1, 7), padding=0, stride=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=1)
        nn.init.constant_(self.conv3.bias, 0)

        self.conv3_nonlin = Expression(square)

        self.bn3 = nn.BatchNorm2d(config.num_kernels, momentum=0.1, affine=True)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

        self.conv4 = nn.Conv2d(config.num_kernels, config.num_kernels, kernel_size=(1, 3), padding=0, stride=1)
        nn.init.xavier_uniform_(self.conv4.weight, gain=1)
        nn.init.constant_(self.conv4.bias, 0)

        self.conv4_nonlin = Expression(square)

        self.bn4 = nn.BatchNorm2d(config.num_kernels, momentum=0.1, affine=True)
        nn.init.constant_(self.bn4.weight, 1)
        nn.init.constant_(self.bn4.bias, 0)

        self.pool_nonlin = Expression(safe_log)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(self.conv2(x))
        x = self.conv12_nonlin(x)
        x = F.avg_pool2d(x, (1, 3), stride=(1, 3))
        x = self.pool_nonlin(x)
        x = self.bn3(self.conv3(x))
        x = self.conv3_nonlin(x)
        x = F.avg_pool2d(x, (1, 3), stride=(1, 3))
        x = self.pool_nonlin(x)
        x = self.bn4(self.conv4(x))
        x = self.conv4_nonlin(x)
        x = F.avg_pool2d(x, (1, 3), stride=(1, 3))
        x = self.pool_nonlin(x)
        return x


class TopNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(config.num_kernels, config.num_classes,
                               kernel_size=(1, ((config.time_series_size-19) - 60) // 40 + 1),
                               stride=(1, 1))
        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        nn.init.constant_(self.conv1.bias, 0)

        self.softmax = nn.Softmax(dim=1)
        self.squeeze = Expression(_squeeze_final_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.squeeze(x)
        return x
