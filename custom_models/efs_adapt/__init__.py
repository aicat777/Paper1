"""Independent EFS adaptation: preserve P3 width and separate the P2 towers."""
from .head import EFSP2IndependentHeadModule
from .neck import EFSPreserveP3PAFPN

__all__ = ['EFSP2IndependentHeadModule', 'EFSPreserveP3PAFPN']
