"""SCDown3 plus seven groups=4 spatial convolutions, separate from G2."""
from .backbone import SCDownG4CSPNeXt
from .neck import SCDownG4PAFPN

__all__ = ['SCDownG4CSPNeXt', 'SCDownG4PAFPN']
