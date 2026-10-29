"""Reweight the existing positive-point QFL without replacing its targets."""
import math

import torch
from torch import nn
from mmdet.utils import reduce_mean


class InstanceBalancedQFLLoss(nn.Module):
    """Mean point QFL within each GT, then sum over ALL valid GTs.

    Uses the detector's actual QualityFocalLoss object, category labels and
    detached IoU targets. No teacher, label filtering or additional parameters.
    Unmatched GTs contribute zero here and retain the original ROI task.
    Different images and duplicate boxes are distinguished by DSL slot indices.
    """
    def __init__(self, loss_weight=.1):
        super().__init__()
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('Instance QFL weight must be finite and nonnegative')
        self.loss_weight = loss_weight

    @staticmethod
    def point_inputs(cls_scores, assignment, valid_gt):
        if valid_gt.dtype != torch.bool or valid_gt.ndim != 2:
            raise ValueError('valid_gt must be a bool [batch, padded GT slot] mask')
        batch_size, max_gt = valid_gt.shape
        indices = assignment['assigned_gt_indices'].detach()
        if indices.shape[0] != batch_size:
            raise ValueError('GT masks and assignment must use the same batch')
        logits, labels, qualities, keys = [], [], [], []
        offset = 0
        for score in cls_scores:
            if score.ndim != 4 or score.shape[0] != batch_size:
                raise ValueError('Expected detector logits in [batch, classes, height, width] order')
            h, w = score.shape[-2:]
            slots = indices[:, offset:offset + h * w]
            batch, point = (slots >= 0).nonzero(as_tuple=True)
            if len(batch):
                gt = slots[batch, point]
                # Original DSL slots include degenerate boxes; exclude those
                # only from this auxiliary term, never edit the assignment.
                keep = valid_gt[batch, gt]
                batch, point, gt = batch[keep], point[keep], gt[keep]
                logits.append(score[batch, :, point // w, point % w])
                labels.append(assignment['assigned_labels'][batch, offset + point].detach())
                qualities.append(assignment['assign_metrics'][batch, offset + point].detach())
                keys.append(batch * max_gt + gt)
            offset += h * w
        if offset != indices.shape[1]:
            raise ValueError('Assignments must cover the entire level-major detection grid')
        if not logits:
            return None
        return tuple(torch.cat(values) for values in (logits, labels, qualities, keys))

    def forward(self, cls_scores, assignment, valid_gt, quality_loss):
        # Connected zero also handles an empty GT rank in DDP. Call reduce_mean
        # on every rank even when this rank has no matched or valid GTs.
        zero = sum(score.float().sum() * 0 for score in cls_scores)
        local_count = valid_gt.sum().to(dtype=torch.float32)
        mean_count = reduce_mean(local_count)
        denominator = torch.where(mean_count > 0, mean_count, mean_count.new_ones(()))
        inputs = self.point_inputs(cls_scores, assignment, valid_gt)
        group_count = valid_gt.numel()
        counts = torch.zeros(group_count, device=valid_gt.device, dtype=torch.long)
        loss_sum = zero
        positive_points = 0
        if inputs is not None:
            logits, labels, qualities, keys = inputs
            positive_points = len(keys)
            counts = torch.bincount(keys, minlength=group_count)
            if self.loss_weight:
                # Preserve the main loss's autocast behavior; only accumulation
                # and GT normalization are FP32. Do not softmax detector scores.
                per_point = quality_loss(logits, (labels, qualities), reduction_override='none')
                if per_point.shape != keys.shape:
                    raise ValueError('Quality loss must return one scalar per detection point')
                sums = per_point.new_zeros(group_count, dtype=torch.float32)
                sums.scatter_add_(0, keys, per_point.float())
                loss_sum = (sums / counts.clamp_min(1)).sum() + zero
        matched = (counts > 0).sum().to(dtype=torch.float32)
        loss = self.loss_weight * loss_sum / denominator
        stats = dict(ibqfl_all_gt=local_count, ibqfl_matched_gt=matched,
                     ibqfl_positive_points=local_count.new_tensor(positive_points),
                     ibqfl_points_per_matched_gt=local_count.new_tensor(positive_points)
                     / matched.clamp_min(1))
        return loss, {name: value.detach() for name, value in stats.items()}
