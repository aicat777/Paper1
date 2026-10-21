"""Change only ROI aggregation to a single bilinear center sample."""
import torch
from mmyolo.models.dense_heads.rtmdet_head import RTMDetHead
from mmyolo.registry import MODELS

from custom_models.instance_semantic.head import (
    InstanceSemanticRTMDetHead, SemanticFeatureHeadModule)
from .sampling import CenterInstanceSemanticLoss


@MODELS.register_module()
class CenterInstanceSemanticRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, semantic_loss_weight=.1, semantic_finest_scale=56, **kwargs):
        # Same construction as the original head, with its pooling subclass.
        RTMDetHead.__init__(self, *args, **kwargs)
        if not isinstance(self.head_module, SemanticFeatureHeadModule):
            raise TypeError('The auxiliary loss needs classification-tower features')
        if self.prior_generator.offset != 0:
            raise ValueError('Center coordinates require offset-zero RTMDet priors')
        with torch.random.fork_rng(devices=[]):
            self.instance_semantic = CenterInstanceSemanticLoss(
                self.head_module.feat_channels, self.num_classes, self.featmap_strides,
                semantic_loss_weight, semantic_finest_scale)
