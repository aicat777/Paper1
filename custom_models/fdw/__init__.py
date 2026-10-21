"""Independent Fourier Disjoint Weight experiment."""
from .backbone import FDWCSPNeXt
from .operator import FDWConv2d

__all__ = ['FDWConv2d', 'FDWCSPNeXt']
