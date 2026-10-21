"""SCDown3 plus seven standard DW3x3 -> PW1x1 replacements."""
from .backbone import SCDownDSConvCSPNeXt
from .neck import SCDownDSConvPAFPN

__all__ = ['SCDownDSConvCSPNeXt', 'SCDownDSConvPAFPN']
