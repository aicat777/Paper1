"""Keep the original cls-feature ROI path; replace only single-level pooling."""
from mmyolo.registry import MODELS

from custom_models.instance_semantic import InstanceSemanticRTMDetHead
from .roi import AllLevelInstanceSemanticLoss


@MODELS.register_module()
class AllLevelROISemanticRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, semantic_feature_gradient_scale=.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_semantic = AllLevelInstanceSemanticLoss(
            self.instance_semantic, semantic_feature_gradient_scale)
