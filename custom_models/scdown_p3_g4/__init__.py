"""Four SCDown transitions and the existing seven-site G4 CSP treatment."""
from custom_models.scdown_g4.neck import SCDownG4PAFPN

from .backbone import SCDownP3G4CSPNeXt

__all__ = ['SCDownP3G4CSPNeXt', 'SCDownG4PAFPN']
