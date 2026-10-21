"""Experiment 1: three SCDowns plus seven groups=2 CSP convolutions."""
from .backbone import SCDownG2CSPNeXt
from .neck import SCDownG2PAFPN

__all__ = ['SCDownG2CSPNeXt', 'SCDownG2PAFPN']
