"""Single main-head/assignment forward with two separately logged ROI losses."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS

from custom_models.instance_semantic import InstanceSemanticRTMDetHead
from .roi import ContrastiveInstanceSemanticLoss


@MODELS.register_module()
class ContrastiveROISemanticRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, semantic_feature_gradient_scale=.25,
                 semantic_contrast_weight=.1, semantic_contrast_temperature=.2,
                 semantic_contrast_channels=128, semantic_pair_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_semantic = ContrastiveInstanceSemanticLoss(
            self.instance_semantic, semantic_feature_gradient_scale,
            semantic_contrast_weight, semantic_contrast_temperature,
            semantic_contrast_channels, semantic_pair_chunk_size)

    def loss(self, x, batch_data_samples):
        cls_scores, bbox_preds, features = self.head_module(x, return_cls_features=True)
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
        losses.update(self.instance_semantic(features, rows))
        return losses
