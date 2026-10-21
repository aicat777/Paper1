"""Expanded RTMDet towers: LSConv -> pointwise Conv/BN -> activation."""
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS
from torch import nn

from .layers import activation, dense_3x3
from .operator import ConvBN, LSConv


def share_convolutions(operator, projection, reference):
    """Share only convolution parameters; each scale keeps its own BN and GN."""
    shared_operator, shared_projection = reference
    for name in ('cv1', 'cv2', 'cv3'):
        getattr(operator.lkp, name).c = getattr(shared_operator.lkp, name).c
    operator.lkp.cv4 = shared_operator.lkp.cv4
    projection.c = shared_projection.c


@MODELS.register_module()
class FullLSConvSepBNHeadModule(RTMDetSepBNHeadModule):
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
        # The inherited forward, prediction layers, initialization, and losses
        # remain those of RTMDet. Classification/regression share no parameters.
