"""All-GT ROI CE and a separate, parameter-free local semantic objective."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mmdet.utils import reduce_mean


def valid_rows_and_lookup(gt, padded):
    """Map actual DSL GT slots to valid ROI rows without matching box values.

    Raw tensor order, duplicate boxes, and degenerate GT slot numbers are
    preserved. DSL's padded GT indices are image-local, not global ROI indices.
    """
    batch_size, max_gt = padded.shape[:2]
    lookup = torch.full((batch_size, max_gt), -1, dtype=torch.long, device=padded.device)
    if isinstance(gt, torch.Tensor):
        valid = (gt[:, 4:6] > gt[:, 2:4]).all(-1)
        rows = gt[valid]
        raw_to_valid = torch.full((len(gt),), -1, dtype=torch.long, device=gt.device)
        raw_to_valid[valid] = torch.arange(len(rows), device=gt.device)
        for image in range(batch_size):
            indices = (gt[:, 0] == image).nonzero().flatten()
            lookup[image, :len(indices)] = raw_to_valid[indices]
    else:
        valid = (padded[..., 3:5] > padded[..., 1:3]).all(-1)
        batch, slot = valid.nonzero(as_tuple=True)
        rows = torch.cat((batch[:, None].to(padded), padded[batch, slot]), 1)
        lookup[batch, slot] = torch.arange(len(rows), device=padded.device)
    return rows, lookup


def original_roi_readout(module, features, rows):
    """Compute the original CE once and reuse its logits as detached targets.

    Pooling, coordinate correction, scale choice, classifier and all-GT
    normalization are unchanged. Returning logits avoids a second ROI pass.
    """
    with torch.autocast(device_type=features[0].device.type, enabled=False):
        count = reduce_mean(features[0].new_tensor(float(len(rows)), dtype=torch.float32)).clamp_min(1)
        if not len(rows):
            zero = (sum(x.float().sum() * 0 for x in features)
                    + sum(p.sum() * 0 for p in module.parameters()))
            return zero, module.classifier.weight.new_empty((0, module.classifier.out_features))
        rois = torch.cat((rows[:, :1], rows[:, 2:6]), 1)
        logits = module.classifier(module.pool_instances(features, rois))
        loss = module.loss_weight * F.cross_entropy(logits, rows[:, 1].long(), reduction='sum') / count
        return loss, logits


class RegionToPointTransfer(nn.Module):
    """Select reliable boxes with wrong detector classes and correct ROI classes.

    KL targets and the auxiliary classifier parameters are detached. Only the
    sampled classification features receive the transfer gradient. First mean
    over each GT's selected points across all levels, then over participating
    GTs. The original ROI CE still covers every valid GT independently.
    """
    def __init__(self, loss_weight=.1, min_iou=.5, temperature=1.):
        super().__init__()
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError('Transfer loss weight must be finite and nonnegative')
        if not math.isfinite(min_iou) or not 0 <= min_iou <= 1:
            raise ValueError('Transfer minimum IoU must be within [0, 1]')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Transfer temperature must be finite and positive')
        self.loss_weight, self.min_iou, self.temperature = loss_weight, min_iou, temperature

    def forward(self, features, cls_scores, assigned, rows, lookup, roi_logits, classifier):
        if len(features) != len(cls_scores):
            raise ValueError('Every feature level needs its actual detector scores')
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            return self._forward_float(features, cls_scores, assigned, rows, lookup,
                                       roi_logits, classifier)

    def _forward_float(self, features, cls_scores, assigned, rows, lookup, roi_logits, classifier):
        zero = sum(x.reshape(-1)[0].float() * 0 for x in features)
        point_features, roi_indices = [], []
        matched_count = rows.new_zeros((), dtype=torch.float32)
        quality_count = matched_count.clone()
        teacher = roi_logits.detach().float()
        offset = 0
        for feature, score in zip(features, cls_scores):
            height, width = feature.shape[-2:]
            size = height * width
            if score.shape[:1] != feature.shape[:1] or score.shape[-2:] != (height, width):
                raise ValueError('Detector scores and classification features must share their grid')
            slots = assigned['assigned_gt_indices'][:, offset:offset + size].detach()
            batch, point = (slots >= 0).nonzero(as_tuple=True)
            if len(batch):
                row = lookup[batch, slots[batch, point]]
                valid = row >= 0
                batch, point, row = batch[valid], point[valid], row[valid]
                matched_count += len(row)
                quality = assigned['assign_metrics'][batch, offset + point].detach()
                reliable = quality >= self.min_iou
                quality_count += reliable.sum()
                y, x = torch.div(point, width, rounding_mode='floor'), point % width
                detector_label = score.detach()[batch, :, y, x].argmax(-1)
                labels = rows[row, 1].long()
                selected = reliable & (teacher[row].argmax(-1) == labels) & (detector_label != labels)
                if self.loss_weight and bool(selected.any()):
                    point_features.append(feature[batch[selected], :, y[selected], x[selected]].float())
                    roi_indices.append(row[selected])
            offset += size
        if offset != assigned['assigned_gt_indices'].shape[1]:
            raise ValueError('Assignment indices must follow the full level-major prediction grid')
        counts = rows.new_zeros(len(rows), dtype=torch.float32)
        sums = counts.clone()
        if point_features:
            values, row = torch.cat(point_features), torch.cat(roi_indices)
            logits = F.linear(values, classifier.weight.detach().float(),
                              classifier.bias.detach().float() if classifier.bias is not None else None)
            temperature = self.temperature
            kl = F.kl_div(F.log_softmax(logits / temperature, -1),
                          F.log_softmax(teacher[row] / temperature, -1),
                          reduction='none', log_target=True).sum(-1) * temperature ** 2
            sums.scatter_add_(0, row, kl)
            counts.scatter_add_(0, row, torch.ones_like(kl))
        participating = (counts > 0).sum().float()
        normalizer = reduce_mean(participating).clamp_min(1)
        mean_kl = (sums / counts.clamp_min(1)).sum() / normalizer + zero
        stats = dict(semantic_all_gt=rows.new_tensor(len(rows), dtype=torch.float32),
                     transfer_matched_points=matched_count,
                     transfer_reliable_points=quality_count,
                     transfer_selected_points=counts.sum(),
                     transfer_selected_gt=participating,
                     transfer_mean_kl=mean_kl.detach())
        return self.loss_weight * mean_kl, {name: value.detach() for name, value in stats.items()}
