"""Official DSL assignment with its actual GT indices included in the result.

The forward below follows the local MMYOLO BatchDynamicSoftLabelAssigner;
only the extra assigned_gt_indices output is added. In particular, neither
matching costs nor dynamic_k_matching nor zero-IoU positives are changed.
Keeping indices from matching avoids guessing identity from labels/boxes.
"""
import torch
import torch.nn.functional as F
from mmdet.structures.bbox import BaseBoxes
from mmyolo.models.task_modules.assigners.batch_dsl_assigner import (
    EPS, INF, BatchDynamicSoftLabelAssigner, find_inside_points, get_box_center)
from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class IndexedDynamicSoftLabelAssigner(BatchDynamicSoftLabelAssigner):
    @torch.no_grad()
    def forward(self, pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes,
                pad_bbox_flag):
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()
        assigned_gt_indices = torch.full(pred_scores.shape[:2], -1,
                                         dtype=torch.long, device=pred_scores.device)
        if num_gt == 0 or num_bboxes == 0:
            return dict(
                assigned_labels=gt_labels.new_full(pred_scores[..., 0].shape,
                                                   self.num_classes, dtype=torch.long),
                assigned_labels_weights=gt_bboxes.new_full(pred_scores[..., 0].shape, 1),
                assigned_bboxes=gt_bboxes.new_full(pred_bboxes.shape, 0),
                assign_metrics=gt_bboxes.new_full(pred_scores[..., 0].shape, 0),
                assigned_gt_indices=assigned_gt_indices)

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(f'type of {type(gt_bboxes)} are not implemented !')
        is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        is_in_gts = is_in_gts.permute(1, 0, 2)
        valid_mask = is_in_gts.sum(dim=-1) > 0
        gt_center = get_box_center(gt_bboxes, box_dim)
        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] -
                    gt_center[:, None, :, :]).pow(2).sum(-1).sqrt() / strides[None, :, None]
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)
        if self.batch_iou:
            pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)
        else:
            pairwise_ious = torch.stack([
                self.iou_calculator(box, gt) for box, gt in zip(decoded_bboxes, gt_bboxes)], dim=0)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight
        pairwise_pred_scores = pred_scores.permute(0, 2, 1)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0], idx[1]].permute(0, 2, 1)
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious, reduction='none') * scale_factor.abs().pow(2.0)
        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior
        max_pad_value = torch.ones_like(cost_matrix) * INF
        cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt),
                                  cost_matrix, max_pad_value)
        matched_pred_ious, matched_gt_inds, fg_mask_inboxes = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, pad_bbox_flag)
        del pairwise_ious, cost_matrix
        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]
        assigned_labels = gt_labels.new_full(pred_scores[..., 0].shape, self.num_classes)
        assigned_labels[fg_mask_inboxes] = gt_labels[batch_index, matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()
        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape, 1)
        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index, matched_gt_inds]
        assign_metrics = gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious
        assigned_gt_indices[fg_mask_inboxes] = matched_gt_inds
        return dict(assigned_labels=assigned_labels,
                    assigned_labels_weights=assigned_labels_weights,
                    assigned_bboxes=assigned_bboxes, assign_metrics=assign_metrics,
                    assigned_gt_indices=assigned_gt_indices)
