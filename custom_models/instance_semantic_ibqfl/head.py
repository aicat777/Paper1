"""One head/DSL pass; retain main losses and all-GT ROI CE unchanged."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmyolo.registry import MODELS

from custom_models.roi_supervision_common.head import SelectedROISemanticHead
from .loss import InstanceBalancedQFLLoss


@MODELS.register_module()
class InstanceBalancedSemanticRTMDetHead(SelectedROISemanticHead):
    def __init__(self, *args, instance_qfl_weight=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_qfl = InstanceBalancedQFLLoss(instance_qfl_weight)

    def loss(self, x, batch_data_samples):
        cls, reg, features = self.head_module(x, return_cls_features=True)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses, padded, assignment = self._detection_losses(cls, reg, gt, metas, ignored)
        valid = (padded[..., 3:5] > padded[..., 1:3]).all(-1)
        if isinstance(gt, torch.Tensor):
            rows = gt
        else:
            batch, slot = valid.nonzero(as_tuple=True)
            rows = torch.cat((batch[:, None].to(padded), padded[batch, slot]), dim=1)
        losses['loss_instance_semantic'] = self.instance_semantic(features, rows)
        value, stats = self.instance_qfl(cls, assignment, valid, self.loss_cls)
        losses['loss_instance_qfl'] = value
        losses.update(stats)
        return losses
