"""Four SCDown transitions and the existing seven-site G8 CSP treatment."""
from custom_models.scdown_g8.neck import SCDownG8PAFPN

from .backbone import SCDownP3G8CSPNeXt

__all__ = ['SCDownP3G8CSPNeXt', 'SCDownG8PAFPN']
