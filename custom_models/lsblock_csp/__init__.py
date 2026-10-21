"""Independent LSConv + residual FFN CSP feature-block candidate."""
from .backbone import LSBlockCSPBackbone
from .block import LSFeatureBlock
from .neck import LSBlockCSPPAFPN

__all__ = ['LSBlockCSPBackbone', 'LSBlockCSPPAFPN', 'LSFeatureBlock']
