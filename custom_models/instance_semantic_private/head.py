"""The main head runs once; auxiliary supervision reads neck, not cls tower."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS

from custom_models.instance_semantic import InstanceSemanticRTMDetHead
from .roi import PrivateInstanceSemanticLoss


@MODELS.register_module()
class PrivateROISemanticRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, semantic_feature_gradient_scale=.25,
                 semantic_private_hidden_channels=192, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_module.in_channels != self.head_module.feat_channels:
            raise ValueError('Private ROI expects the actual equal-width neck outputs')
        self.instance_semantic = PrivateInstanceSemanticLoss(
            self.instance_semantic, semantic_feature_gradient_scale,
            semantic_private_hidden_channels)

    def loss(self, x, batch_data_samples):
        # Ordinary detection, without evaluating the classification tower twice.
        cls_scores, bbox_preds = self.head_module(x)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses = self.loss_by_feat(cls_scores, bbox_preds, gt, metas, ignored)
        if isinstance(gt, torch.Tensor):
            rows = gt
        else:
            padded = gt_instances_preprocess(gt, len(metas))
            valid = (padded[..., 3:5] > padded[..., 1:3]).all(dim=-1)
            image, slot = valid.nonzero(as_tuple=True)
            rows = torch.cat((image[:, None].to(padded), padded[image, slot]), dim=1)
        losses['loss_instance_semantic'] = self.instance_semantic(x, rows)
        return losses
