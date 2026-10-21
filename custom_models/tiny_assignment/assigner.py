"""DSL with local rescue for unmatched sub-stride GTs, plus explicit GT IDs.

The ordinary matching cost and dynamic-k procedure are MMYOLO's DSL. Rescue
does not steal existing valid positives. An augmenting-path matching resolves
competition among new local matches; a GT stays unmatched if no local free
prior exists. Classification quality remains the actual predicted-box IoU.
"""
from collections import deque

import torch
from torch.nn import functional as F
from mmyolo.models.task_modules.assigners.batch_dsl_assigner import (
    BatchDynamicSoftLabelAssigner, EPS, INF, find_inside_points, get_box_center)
from mmyolo.registry import TASK_UTILS


def local_bipartite_matching(candidates):
    """Maximum-cardinality local matching; candidate order encodes DSL cost.

    Iterative augmenting paths avoid recursion-depth failures on dense images.
    The input contains ONLY initially free priors, hence existing matches cannot
    be displaced. This maximizes coverage, not the sum of matching costs.
    """
    prior_owner, gt_prior = {}, {}
    for root in candidates:
        queue, seen_gt, parent = deque([root]), {root}, {}
        free = None
        while queue and free is None:
            gt = queue.popleft()
            for prior in candidates[gt]:
                if prior in parent:
                    continue
                parent[prior] = gt
                if prior not in prior_owner:
                    free = prior
                    break
                owner = prior_owner[prior]
                if owner not in seen_gt:
                    seen_gt.add(owner)
                    queue.append(owner)
        while free is not None:
            gt = parent[free]
            previous = gt_prior.get(gt)
            prior_owner[free], gt_prior[gt] = gt, free
            free = previous
    return gt_prior


@TASK_UTILS.register_module()
class TinyRescueDSLAssigner(BatchDynamicSoftLabelAssigner):
    def __init__(self, *args, tiny_size=8., rescue_radius=1.5,
                 regression_floor=.05, enable_rescue=True, **kwargs):
        super().__init__(*args, **kwargs)
        if tiny_size < 0 or rescue_radius <= 0 or not 0 <= regression_floor <= 1:
            raise ValueError('Invalid tiny-object assignment parameters')
        self.tiny_size, self.rescue_radius = tiny_size, rescue_radius
        self.regression_floor, self.enable_rescue = regression_floor, enable_rescue

    @torch.no_grad()
    def forward(self, pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag):
        batch, points, _ = pred_bboxes.shape
        num_gt = gt_bboxes.shape[1]
        owner = torch.full((batch, points), -1, dtype=torch.long, device=pred_bboxes.device)
        fallback = torch.zeros_like(owner, dtype=torch.bool)
        if num_gt == 0 or points == 0:
            result = super().forward(pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes, pad_bbox_flag)
            result.update(assigned_gt_indices=owner, fallback_mask=fallback,
                          regression_weights=result['assign_metrics'].clone(),
                          unmatched_tiny=torch.zeros((batch, num_gt), dtype=torch.bool, device=owner.device))
            return result
        valid_gt = pad_bbox_flag[..., 0].bool()
        tiny = ((gt_bboxes[..., 2:] - gt_bboxes[..., :2]).amin(-1) < self.tiny_size) & valid_gt
        inside = find_inside_points(gt_bboxes, priors[:, :2], 4)
        valid_points = (inside * pad_bbox_flag[..., 0][None]).permute(1, 0, 2).sum(-1) > 0
        centers = get_box_center(gt_bboxes, 4)
        distance = ((priors[None, :, None, :2] - centers[:, None]).square().sum(-1).sqrt()
                    / priors[None, :, None, 2])
        # Exact baseline cost, including its inside-ANY-GT union eligibility.
        center_cost = torch.pow(10, distance * valid_points[..., None] - self.soft_center_radius)
        if self.batch_iou:
            ious = self.iou_calculator(pred_bboxes, gt_bboxes)
        else:
            ious = torch.stack([self.iou_calculator(p, g) for p, g in zip(pred_bboxes, gt_bboxes)])
        logits = pred_scores.gather(2, gt_labels.long().squeeze(-1)[:, None].expand(-1, points, -1))
        cls_cost = F.binary_cross_entropy_with_logits(logits, ious, reduction='none')
        cls_cost = cls_cost * (ious - logits.sigmoid()).abs().square()
        pair_cost = cls_cost - torch.log(ious + EPS) * self.iou_weight
        cost = torch.where(valid_points[..., None], pair_cost + center_cost, torch.full_like(pair_cost, INF))
        matched_quality, matched_gt, foreground = super().dynamic_k_matching(cost, ious, pad_bbox_flag)
        owner[foreground] = matched_gt

        if self.enable_rescue:
            # DSL can assign an INF-cost prior when no eligible point exists.
            # Clear only such tiny-object assignments; never equate an outside-
            # OWN-box point with an invalid union candidate.
            invalid_tiny = (owner >= 0) & tiny.gather(1, owner.clamp(min=0)) & ~valid_points
            owner[invalid_tiny] = -1
            finest = priors[:, 2] == priors[:, 2].min()
            for b in range(batch):
                counts = torch.bincount(owner[b][owner[b] >= 0], minlength=num_gt)
                missing = (tiny[b] & (counts == 0)).nonzero().flatten()
                candidates = {}
                for gt in missing.tolist():
                    delta = (priors[:, :2] - centers[b, gt]).abs() / priors[:, 2:4]
                    available = finest & (delta.amax(-1) <= self.rescue_radius) & (owner[b] < 0)
                    ids = available.nonzero().flatten()
                    local_cost = pair_cost[b, ids, gt] + torch.pow(10, distance[b, ids, gt] - self.soft_center_radius)
                    ids = ids[torch.argsort(local_cost, stable=True)]
                    candidates[gt] = ids.tolist()
                for gt, prior in local_bipartite_matching(candidates).items():
                    owner[b, prior] = gt
                    fallback[b, prior] = True

        foreground = owner >= 0
        b, p = foreground.nonzero(as_tuple=True)
        g = owner[b, p]
        labels = gt_labels.new_full((batch, points), self.num_classes, dtype=torch.long)
        boxes = gt_bboxes.new_zeros(pred_bboxes.shape)
        quality = gt_bboxes.new_zeros((batch, points))
        labels[b, p] = gt_labels[b, g, 0].long()
        boxes[b, p] = gt_bboxes[b, g]
        quality[b, p] = ious[b, p, g]
        regression_weights = quality.clone()
        tiny_positive = foreground & tiny.gather(1, owner.clamp(min=0))
        regression_weights[tiny_positive] = regression_weights[tiny_positive].clamp(min=self.regression_floor)
        unmatched = torch.zeros_like(tiny)
        for image in range(batch):
            counts = torch.bincount(owner[image][owner[image] >= 0], minlength=num_gt)
            unmatched[image] = tiny[image] & (counts == 0)
        return dict(assigned_labels=labels, assigned_labels_weights=quality.new_ones(quality.shape),
                    assigned_bboxes=boxes, assign_metrics=quality,
                    assigned_gt_indices=owner, fallback_mask=fallback,
                    regression_weights=regression_weights, unmatched_tiny=unmatched)
