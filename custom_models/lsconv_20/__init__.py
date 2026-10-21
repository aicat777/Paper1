"""Independent 20-site LSConv variant with the original neck output layers."""
from .backbone import LSConv20CSPNeXt
from .head import LSConv20SepBNHeadModule
from .neck import LSConv20PAFPN
from .operator import LSConv

__all__ = ['LSConv20CSPNeXt', 'LSConv20PAFPN',
           'LSConv20SepBNHeadModule', 'LSConv']
