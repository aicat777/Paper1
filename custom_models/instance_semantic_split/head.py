"""Split the original ROI task's classifier and shared-feature gradient paths."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmyolo.registry import MODELS

from custom_models.roi_supervision_common import SelectedROISemanticHead
from custom_models.roi_supervision_common.supervision import valid_rows_and_matches
from .supervision import MatchedFeatureGradientGate, split_roi_loss


@MODELS.register_module()
class SplitGradientInstanceSemanticRTMDetHead(SelectedROISemanticHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_gate = MatchedFeatureGradientGate()

    def loss(self, x, batch_data_samples):
        cls, reg, features = self.head_module(x, return_cls_features=True)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses, padded, assigned = self._detection_losses(cls, reg, gt, metas, ignored)
        rows, matched = valid_rows_and_matches(gt, padded, assigned['assigned_gt_indices'])
        losses['loss_instance_semantic'] = split_roi_loss(
            self.instance_semantic, self.gradient_gate, features, rows, matched)
        losses['semantic_all_gt'] = cls[0].new_tensor(len(rows), dtype=torch.float32)
        losses['semantic_matched_gt'] = matched.sum().float()
        losses['semantic_classifier_gt'] = losses['semantic_all_gt'].clone()
        losses['semantic_feature_gt'] = losses['semantic_matched_gt'].clone()
        return losses
