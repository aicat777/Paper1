"""P3 mechanism control A, independent of every other SCDown variant."""
from custom_models.scdown import SCDownPAFPN
from .backbone import SCDownP3DenseCSPNeXt
from .operator import P3DenseSpatial

__all__ = ['P3DenseSpatial', 'SCDownP3DenseCSPNeXt', 'SCDownPAFPN']
