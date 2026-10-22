"""SCDown3 plus seven groups=16 spatial convolutions, separate from G2."""
from .backbone import SCDownG16CSPNeXt
from .neck import SCDownG16PAFPN

__all__ = ['SCDownG16CSPNeXt', 'SCDownG16PAFPN']
