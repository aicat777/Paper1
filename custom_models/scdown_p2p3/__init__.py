"""Five SCDown transitions; no P2 prediction head is added."""
from custom_models.scdown import SCDownPAFPN
from .backbone import SCDownP2P3CSPNeXt

__all__ = ['SCDownP2P3CSPNeXt', 'SCDownPAFPN']
