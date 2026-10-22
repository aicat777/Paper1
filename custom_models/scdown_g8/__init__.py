"""SCDown3 plus seven groups=8 spatial convolutions, separate from G2."""
from .backbone import SCDownG8CSPNeXt
from .neck import SCDownG8PAFPN

__all__ = ['SCDownG8CSPNeXt', 'SCDownG8PAFPN']
