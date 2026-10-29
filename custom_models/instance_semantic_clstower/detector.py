"""Reuse the validated route, excluding both backbone and neck parameters."""
from mmyolo.registry import MODELS

from custom_models.instance_semantic_neckhead import NeckHeadROISemanticDetector


@MODELS.register_module()
class ClassificationOnlyROISemanticDetector(NeckHeadROISemanticDetector):
    def roi_gradient_parameters(self):
        """Classification-tower convolutions/BN and the auxiliary classifier.

        Cross-scale convolution sharing and independent BN remain unchanged.
        Final detection predictors and regression parameters are not included,
        as in the original ROI. Main detection gradients remain unrestricted.
        """
        return tuple(self.bbox_head.head_module.cls_convs.parameters()) + tuple(
            self.bbox_head.instance_semantic.parameters())
