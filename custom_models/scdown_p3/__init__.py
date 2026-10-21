"""Four SCDown transitions; the original three-scale neck/head remain."""
from custom_models.scdown import SCDownPAFPN
from .backbone import SCDownP3CSPNeXt

__all__ = ['SCDownP3CSPNeXt', 'SCDownPAFPN']
