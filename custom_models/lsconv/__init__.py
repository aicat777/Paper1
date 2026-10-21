"""Independent LSConv migration for the CSPNeXt backbone."""
from .backbone import LSConvCSPNeXt
from .operator import LSConv

__all__ = ['LSConv', 'LSConvCSPNeXt']
