"""All-GT classifier learning; only matched instances backpropagate to features."""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.utils import reduce_mean


class MatchedFeatureGradientGate(nn.Module):
    """Keep every descriptor's value, stopping only unmatched contributions.

    Routing after ROI pooling preserves each GT's identity even when its ROI
    overlaps another GT. This never masks shared feature-map pixels.
    """

    def forward(self, pooled, matched):
        if pooled.ndim != 2:
            raise ValueError('Expected one feature vector per valid GT')
        if matched.dtype != torch.bool or matched.shape != (len(pooled),):
            raise ValueError('Expected one bool matched flag per valid GT')
        if matched.device != pooled.device:
            raise ValueError('Descriptors and matched flags must share a device')
        return torch.where(matched[:, None], pooled, pooled.detach())


def split_roi_loss(module, gate, features, rows, matched):
    """Preserve original all-GT forward loss and classifier parameter gradients."""
    if matched.dtype != torch.bool or matched.shape != (len(rows),):
        raise ValueError('Matched flags must correspond to the valid GT rows')
    if len(rows) == 0:
        # Preserve the original connected zero for features AND classifier.
        return module(features, rows)
    with torch.autocast(device_type=features[0].device.type, enabled=False):
        count = reduce_mean(features[0].new_tensor(float(len(rows)), dtype=torch.float32)).clamp_min(1)
        rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
        pooled = module.pool_instances(features, rois)
        logits = module.classifier(gate(pooled, matched))
        return module.loss_weight * F.cross_entropy(logits, rows[:, 1].long(), reduction='sum') / count
