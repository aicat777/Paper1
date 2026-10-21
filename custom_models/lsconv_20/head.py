"""Twelve LSConv -> PW -> SiLU executions with shared Conv and separate norms."""
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS
from torch import nn

from .initialization import preserve_lsconv_initialization
from .layers import activation, dense_3x3
from .operator import ConvBN, LSConv


def share_convolutions(operator, projection, reference):
    shared_operator, shared_projection = reference
    for name in ('cv1', 'cv2', 'cv3'):
        getattr(operator.lkp, name).c = getattr(shared_operator.lkp, name).c
    operator.lkp.cv4 = shared_operator.lkp.cv4
    projection.c = shared_projection.c


@MODELS.register_module()
class LSConv20SepBNHeadModule(RTMDetSepBNHeadModule):
    def _init_layers(self):
        super()._init_layers()
        self.lsconv_replacements = []
        for name in ('cls_convs', 'reg_convs'):
            towers = getattr(self, name)
            for depth in range(self.stacked_convs):
                reference = None
                for level in range(len(self.featmap_strides)):
                    previous = towers[level][depth]
                    conv = dense_3x3(previous)
                    operator = LSConv(conv.in_channels)
                    projection = ConvBN(conv.in_channels, conv.out_channels)
                    if self.share_conv and reference is not None:
                        share_convolutions(operator, projection, reference)
                    if reference is None:
                        reference = operator, projection
                    towers[level][depth] = nn.Sequential(
                        operator, projection, activation(previous))
                    self.lsconv_replacements.append(f'{name}.{level}.{depth}')

    def init_weights(self):
        if self.is_init:
            return
        # RTMDet recursively normal-initializes every Conv2d at std=.01. Apply
        # that policy to PW/prediction layers, keeping the author's LSConv
        # constructor state (including cv4 bias and independent norm buffers).
        with preserve_lsconv_initialization(self):
            super().init_weights()
