"""Four-site SCDown + seven-site G4 + original all-GT ROI supervision.

Reuse the independently validated operators and auxiliary task unchanged.
The companion config assembles them into an isolated experiment.
"""
from custom_models.scdown_p3_g4 import SCDownP3G4CSPNeXt, SCDownG4PAFPN
from custom_models.instance_semantic import (
    InstanceSemanticRTMDetHead, SemanticFeatureHeadModule)

__all__ = ['SCDownP3G4CSPNeXt', 'SCDownG4PAFPN',
           'InstanceSemanticRTMDetHead', 'SemanticFeatureHeadModule']
