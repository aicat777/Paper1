"""Four SCDown transitions and the existing seven-site G2 CSP treatment."""
from custom_models.scdown_g2.neck import SCDownG2PAFPN

from .backbone import SCDownP3G2CSPNeXt

__all__ = ['SCDownP3G2CSPNeXt', 'SCDownG2PAFPN']
