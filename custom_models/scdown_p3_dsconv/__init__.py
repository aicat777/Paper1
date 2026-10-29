"""Four SCDown transitions and the existing seven-site DSConv treatment."""
from custom_models.scdown_dsconv.neck import SCDownDSConvPAFPN

from .backbone import SCDownP3DSConvCSPNeXt

__all__ = ['SCDownP3DSConvCSPNeXt', 'SCDownDSConvPAFPN']
