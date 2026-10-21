"""Preserve exact GT identity and the original all-valid-GT denominator."""
import torch
from mmdet.utils import reduce_mean
from torch.nn import functional as F


def valid_rows_and_matches(gt, padded, assigned_gt_indices):
    batch_size, max_gt = padded.shape[:2]
    matched = torch.zeros((batch_size, max_gt), dtype=torch.bool, device=padded.device)
    batch, prior = (assigned_gt_indices >= 0).nonzero(as_tuple=True)
    matched[batch, assigned_gt_indices[batch, prior]] = True
    if isinstance(gt, torch.Tensor):
        # Preserve raw row order, including interleaved images. Slot numbers
        # include degenerate GTs, just as gt_instances_preprocess does.
        rows = gt
        per_row = torch.zeros(len(rows), dtype=torch.bool, device=rows.device)
        for image in range(batch_size):
            indices = (rows[:, 0] == image).nonzero().flatten()
            per_row[indices] = matched[image, :len(indices)]
    else:
        valid = (padded[..., 3:5] > padded[..., 1:3]).all(-1)
        batch, index = valid.nonzero(as_tuple=True)
        rows = torch.cat((batch[:, None].to(padded), padded[batch, index]), dim=1)
        per_row = matched[batch, index]
    valid = (rows[:, 4:6] > rows[:, 2:4]).all(-1)
    return rows[valid], per_row[valid]


def selected_roi_loss(module, features, all_rows, selected):
    if selected.dtype != torch.bool or selected.shape != (len(all_rows),):
        raise ValueError('Selection must be one bool for each valid GT row')
    with torch.autocast(device_type=features[0].device.type, enabled=False):
        # Normalize by ALL valid GTs, even when none of them is selected.
        count = reduce_mean(features[0].new_tensor(float(len(all_rows)), dtype=torch.float32)).clamp_min(1)
        rows = all_rows[selected]
        if not len(rows):
            return (sum(x.float().sum() * 0 for x in features)
                    + sum(p.sum() * 0 for p in module.parameters()))
        rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
        logits = module.classifier(module.pool_instances(features, rois))
        return module.loss_weight * F.cross_entropy(logits, rows[:, 1].long(), reduction='sum') / count
