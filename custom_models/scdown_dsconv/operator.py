"""Standard DW3x3 -> PW1x1, retaining RTMDet BN/SiLU after each step.

This is the MobileNet-style depthwise-separable ordering, with the project's
SiLU rather than MobileNet's ReLU. It is not SCDown's PW -> DW ordering.
"""
from copy import deepcopy

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import nn


def depthwise_separable_from(original):
    if not isinstance(original, ConvModule):
        raise TypeError('Expected CSPNeXtBlock.conv1 ConvModule')
    conv = original.conv
    if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
            or conv.stride != (1, 1) or conv.padding != (1, 1)
            or conv.dilation != (1, 1) or conv.groups != 1
            or conv.in_channels != conv.out_channels
            or conv.bias is not None or conv.padding_mode != 'zeros'
            or original.order != ('conv', 'norm', 'act')
            or not original.with_norm or not isinstance(original.norm, nn.BatchNorm2d)
            or not original.with_activation or not isinstance(original.activate, nn.SiLU)):
        raise ValueError('Expected equal widths, dense stride-1 3x3, BN and SiLU')
    return DepthwiseSeparableConvModule(
        conv.in_channels, conv.out_channels, 3, stride=1, padding=1,
        norm_cfg=deepcopy(original.norm_cfg), act_cfg=deepcopy(original.act_cfg))
