"""Complete LSConv from LSNet (CVPR 2025), preserving LKP/SKA/BN/residual.

Architecture: https://github.com/THU-MIG/lsnet/blob/master/model/lsnet.py
Reference commit: cbe737c92b7c43ecf02d08545a07f03f1010177c.
"""
from torch import nn

from .aggregation import small_kernel_aggregation


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, padding=0, groups=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_channels, out_channels, kernel,
                                     padding=padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


class LargeKernelPerception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 8:
            raise ValueError('LSConv uses eight channels per shared kernel')
        hidden, groups = channels // 2, channels // 8
        self.cv1 = ConvBN(channels, hidden)
        self.act = nn.ReLU()
        self.cv2 = ConvBN(hidden, hidden, kernel=7, padding=3, groups=hidden)
        self.cv3 = ConvBN(hidden, hidden)
        self.cv4 = nn.Conv2d(hidden, 9 * groups, 1)
        self.norm = nn.GroupNorm(groups, 9 * groups)

    def forward(self, x):
        y = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        weights = self.norm(self.cv4(y))
        b, _, h, w = weights.shape
        return weights.reshape(b, x.shape[1] // 8, 9, h, w)


class LSConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lkp = LargeKernelPerception(channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn(small_kernel_aggregation(x, self.lkp(x)))
