"""Independent CSP-only extension of the legacy three-site LSConv adapter."""
from .backbone import CSPOnlyLSConvBackbone
from .neck import CSPOnlyLSConvPAFPN

__all__ = ['CSPOnlyLSConvBackbone', 'CSPOnlyLSConvPAFPN']
