"""Three late SCDown transitions; otherwise the original RTMDet-tiny."""
from .backbone import SCDownCSPNeXt
from .neck import SCDownPAFPN
from .operator import SCDown

__all__ = ['SCDown', 'SCDownCSPNeXt', 'SCDownPAFPN']
