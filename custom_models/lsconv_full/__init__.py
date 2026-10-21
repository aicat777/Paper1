"""Independent expanded LSConv variant; preserves the completed experiment."""
from .backbone import FullLSConvCSPNeXt
from .head import FullLSConvSepBNHeadModule
from .neck import FullLSConvPAFPN
from .operator import LSConv

__all__ = ['FullLSConvCSPNeXt', 'FullLSConvPAFPN',
           'FullLSConvSepBNHeadModule', 'LSConv']
