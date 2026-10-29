"""Compose the validated four-site SCDown structure and original all-GT ROI task.

No copied operators, feature-gradient gates, grouped CSPs, or new predictors.
The separate config below assembles these unchanged registered components.
"""
from custom_models.scdown_p3 import SCDownP3CSPNeXt, SCDownPAFPN
from custom_models.instance_semantic import InstanceSemanticRTMDetHead, SemanticFeatureHeadModule

__all__ = ['SCDownP3CSPNeXt', 'SCDownPAFPN',
           'InstanceSemanticRTMDetHead', 'SemanticFeatureHeadModule']
