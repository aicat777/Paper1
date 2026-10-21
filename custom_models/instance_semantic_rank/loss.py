"""One softmax CE per matched GT, computed on mean real detector logits."""
import math

import torch
import torch.nn.functional as F
from mmdet.utils import reduce_mean
from torch import nn


class InstanceCategoryRankLoss(nn.Module):
    def __init__(self, loss_weight=.1):
        super().__init__()
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('Rank loss weight must be finite and nonnegative')
        self.loss_weight = loss_weight

    @staticmethod
    def pool_logits(logits, assigned_gt_indices, gt_labels):
        """Group by exact (image, GT index), across all prediction levels.

        Background index is -1. Different instances with equal labels or
        equal boxes retain distinct identities. No sigmoid, IoU weighting,
        extra matching, or predicted-score threshold is applied here.
        """
        if logits.ndim != 3 or assigned_gt_indices.shape != logits.shape[:2]:
            raise ValueError('Expected B x N x C logits and B x N GT indices')
        if gt_labels.ndim != 3 or gt_labels.shape[0] != logits.shape[0] or gt_labels.shape[2] != 1:
            raise ValueError('Expected B x G x 1 ground-truth labels')
        batch, prior = (assigned_gt_indices >= 0).nonzero(as_tuple=True)
        # FP32 accumulation also under AMP: crowded instances can have many priors.
        values = logits[batch, prior].float()
        keys = batch * gt_labels.shape[1] + assigned_gt_indices[batch, prior]
        keys, inverse, counts = torch.unique(keys, sorted=True, return_inverse=True,
                                             return_counts=True)
        pooled = values.new_zeros((len(keys), logits.shape[-1])).index_add(0, inverse, values)
        pooled = pooled / counts[:, None].clamp_min(1)
        targets = gt_labels.reshape(-1)[keys].long()
        return pooled, targets, keys, counts

    def forward(self, logits, assigned_gt_indices, gt_labels):
        if self.loss_weight == 0:
            return logits.float().sum() * 0
        pooled, targets, _, _ = self.pool_logits(logits, assigned_gt_indices, gt_labels)
        if len(targets):
            loss_sum = F.cross_entropy(pooled, targets, reduction='sum')
        else:
            loss_sum = logits.float().sum() * 0
        # Every DDP rank participates, including ranks without matched GTs.
        # DDP averages gradients; divide by mean count to obtain a global GT mean.
        mean_count = reduce_mean(logits.new_tensor(len(targets), dtype=torch.float32))
        mean_count = torch.where(mean_count > 0, mean_count, mean_count.new_ones(()))
        return loss_sum / mean_count * self.loss_weight
