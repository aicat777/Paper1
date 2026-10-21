"""LSConv in neck CSP blocks and projected three-scale output layers."""
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from torch import nn

from .layers import activation, dense_3x3, replace_csp_blocks
from .operator import ConvBN, LSConv


@MODELS.register_module()
class FullLSConvPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsconv_replacements = replace_csp_blocks(self)
        for level, previous in enumerate(self.out_layers):
            conv = dense_3x3(previous, equal_width=False)
            # Explicit value-channel mixing also remains at the equal-width P3.
            self.out_layers[level] = nn.Sequential(
                ConvBN(conv.in_channels, conv.out_channels),
                LSConv(conv.out_channels), activation(previous))
            self.lsconv_replacements.append(f'out_layers.{level}')
