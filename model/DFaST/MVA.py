import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as f


class ECAAttention(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

    Q. Wang, B. Wu, P. F. Zhu, P. Li, W. Zuo and Q. Hu

    2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2019 Pages 11531-11539

    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x * y.expand_as(x)


class SEAttention(nn.Module):
    """
    Squeeze-and-excitation networks

    J. Hu, L. Shen and G. Sun

    Proceedings of the IEEE conference on computer vision and pattern recognition 2018

    Pages: 7132-7141

    """
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def get_attention(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y


class MultiViewAttention(nn.Module):
    def __init__(self, config, pooling=True):
        super(MultiViewAttention, self).__init__()
        k = config.num_kernels // 4
        self.kernel_list = list(range(1, config.frequency//2, config.frequency//2//k))[:k]
        self.kernels = nn.ModuleList([nn.Sequential(nn.ZeroPad2d((0, (k1+k2) % 2, 0, 0)),
                                                    nn.Conv2d(1, 2,
                                                              kernel_size=(1, k1),
                                                              padding=(0, (k1-1)//2)),
                                                    nn.GELU(),
                                                    nn.Conv2d(2, 4,
                                                              kernel_size=(1, k2),
                                                              padding=(0, (k2-1)//2)))
                                      for k1, k2 in zip(self.kernel_list, reversed(self.kernel_list))])
        # self.conv = nn.Conv2d(1, config.num_kernels, (1, 32), padding=(0, 32 // 2 - 1), bias=False)
        # self.attention = ECAAttention()
        self.attention = SEAttention(channel=config.num_kernels, reduction=1)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels)

        self.dw_conv = nn.Sequential(nn.Conv2d(config.num_kernels, config.num_kernels * config.D, (config.node_size, 1),
                                               bias=False),
                                     nn.BatchNorm2d(config.num_kernels * config.D, False),
                                     nn.GELU()
                                     )
        self.pooling = nn.AvgPool2d((1, config.p1)) if pooling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_state):
        multi_freq_feature = []
        for kernel in self.kernels:
            multi_freq_feature.append(kernel(hidden_state))
        multi_freq_feature = torch.stack(multi_freq_feature, dim=1).squeeze(2)
        multi_freq_feature = rearrange(multi_freq_feature, 'b g c n t -> b (g c) n t')
        multi_freq_feature = self.attention(multi_freq_feature)
        multi_freq_feature = self.batch_norm(multi_freq_feature)

        multi_freq_feature = self.dw_conv(multi_freq_feature)
        if self.pooling is not None:
            multi_freq_feature = self.pooling(multi_freq_feature)
        multi_freq_feature = self.dropout(multi_freq_feature)

        return multi_freq_feature

    def get_attention(self, hidden_state):
        multi_freq_feature = []
        for kernel in self.kernels:
            multi_freq_feature.append(kernel(hidden_state))
        multi_freq_feature = torch.stack(multi_freq_feature, dim=1).squeeze(2)
        multi_freq_feature = rearrange(multi_freq_feature, 'b g c n t -> b (g c) n t')
        attention = self.attention.get_attention(multi_freq_feature)
        return attention


class BaseFrequencyModule(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces

    V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung and B. J. Lance

    Journal of neural engineering 2018 Vol. 15 Issue 5 Pages 056013

    """
    def __init__(self, config):
        super(BaseFrequencyModule, self).__init__()
        self.config = config
        kern_length = config.frequency // 2
        self.padding_flag = False if kern_length % 2 else True

        # Layer 1
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.conv = nn.Conv2d(1, config.num_kernels, (1, kern_length), padding=(0, (kern_length-1)//2), bias=False)
        self.batch_norm = nn.BatchNorm2d(config.num_kernels, False)

    def forward(self, hidden_state):
        hidden_state = self.padding(hidden_state) if self.padding_flag else hidden_state
        hidden_state = self.batch_norm(self.conv(hidden_state))
        return hidden_state
