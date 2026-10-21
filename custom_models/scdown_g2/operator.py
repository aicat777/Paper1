"""Two contiguous channel groups, retaining the original BN and SiLU policy."""
from copy import deepcopy

from mmcv.cnn import ConvModule
from torch import nn


def grouped_conv_from(original):
    if not isinstance(original, ConvModule):
        raise TypeError('Expected CSPNeXtBlock.conv1 ConvModule')
    conv = original.conv
    if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
            or conv.stride != (1, 1) or conv.padding != (1, 1)
            or conv.dilation != (1, 1) or conv.groups != 1
            or conv.in_channels != conv.out_channels or conv.in_channels % 2
            or conv.bias is not None or conv.padding_mode != 'zeros'
            or original.order != ('conv', 'norm', 'act')
            or not original.with_norm or not isinstance(original.norm, nn.BatchNorm2d)
            or not original.with_activation or not isinstance(original.activate, nn.SiLU)):
        raise ValueError('Expected an even-width dense stride-1 3x3 with BN and SiLU')
    return ConvModule(conv.in_channels, conv.out_channels, 3, padding=1, groups=2,
                      norm_cfg=deepcopy(original.norm_cfg), act_cfg=deepcopy(original.act_cfg))
