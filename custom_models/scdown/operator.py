"""YOLOv10 SCDown: pointwise mixing before depthwise downsampling.

Reference: THU-MIG/yolov10, commit
453c6e38a51e9d1d5a2aa5fb7f1014a711913397, modules/block.py::SCDown.
The project keeps the RTMDet BN configuration and SiLU. No dynamic kernel,
extra residual, terminal activation, or channel compression is added.
"""
from copy import deepcopy

from mmcv.cnn import ConvModule
from torch import nn


class SCDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, act_cfg=None):
        super().__init__()
        norm_cfg = deepcopy(norm_cfg) if norm_cfg is not None else dict(type='BN', eps=.001, momentum=.03)
        act_cfg = deepcopy(act_cfg) if act_cfg is not None else dict(type='SiLU', inplace=True)
        if norm_cfg.get('type') != 'BN' or act_cfg.get('type') != 'SiLU':
            raise ValueError('This SCDown adaptation requires BN and SiLU')
        self.cv1 = ConvModule(in_channels, out_channels, 1,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(out_channels, out_channels, 3, stride=2, padding=1,
                              groups=out_channels, norm_cfg=deepcopy(norm_cfg), act_cfg=None)

    @classmethod
    def from_conv_module(cls, original):
        if not isinstance(original, ConvModule):
            raise TypeError('SCDown replaces one dense Conv/BN/SiLU transition')
        conv = original.conv
        if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
                or conv.stride != (2, 2) or conv.padding != (1, 1)
                or conv.dilation != (1, 1) or conv.groups != 1 or conv.bias is not None
                or conv.padding_mode != 'zeros' or original.order != ('conv', 'norm', 'act')
                or not original.with_norm or not isinstance(original.norm, nn.BatchNorm2d)
                or not original.with_activation or not isinstance(original.activate, nn.SiLU)):
            raise ValueError('Expected a dense stride-2 3x3 with BN and SiLU')
        return cls(conv.in_channels, conv.out_channels, original.norm_cfg, original.act_cfg)

    def forward(self, x):
        return self.cv2(self.cv1(x))
