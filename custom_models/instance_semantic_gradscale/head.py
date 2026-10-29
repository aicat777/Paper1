"""Retain the original detector head and replace only the ROI gradient route."""
from mmyolo.registry import MODELS

from custom_models.instance_semantic import InstanceSemanticRTMDetHead
from .roi import GradientScaledInstanceSemanticLoss, validate_gradient_scale


@MODELS.register_module()
class GradientScaledSemanticRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, semantic_feature_gradient_scale=.25, **kwargs):
        scale = validate_gradient_scale(semantic_feature_gradient_scale)
        super().__init__(*args, **kwargs)
        self.instance_semantic = GradientScaledInstanceSemanticLoss(
            self.instance_semantic, feature_gradient_scale=scale)
