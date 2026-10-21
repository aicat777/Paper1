"""Training-only category supervision on true resized object coordinates."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmdet.utils import reduce_mean


class CropSemanticLoss(nn.Module):
    def __init__(self, channels, num_classes, strides=(8, 16, 32),
                 loss_weight=.1, finest_scale=56, output_size=3):
        super().__init__()
        self.strides = tuple(strides)
        self.loss_weight, self.finest_scale, self.output_size = loss_weight, finest_scale, output_size
        self.classifier = nn.Linear(channels, num_classes)
        nn.init.normal_(self.classifier.weight, std=.01)
        nn.init.zeros_(self.classifier.bias)

    def levels(self, boxes):
        sizes = boxes[:, 2:] - boxes[:, :2]
        if not torch.isfinite(boxes).all() or not (sizes > 0).all():
            raise ValueError('Crop GT coordinates must be finite and positive-area')
        scale = sizes.float().prod(dim=1).sqrt()
        return torch.floor(torch.log2(scale / self.finest_scale + 1e-6)).clamp(0, len(self.strides) - 1).long()

    def pool_instances(self, features, rois, levels):
        first = next(iter(features.values()))
        pooled = first.new_zeros((len(rois), first.shape[1]), dtype=torch.float32)
        for level in levels.unique(sorted=True).tolist():
            indices = (levels == level).nonzero().flatten()
            selected = rois[indices].float().clone()
            selected[:, 1:] = selected[:, 1:] / self.strides[level] + .5
            values = roi_align(features[level].float(), selected, self.output_size,
                               spatial_scale=1., sampling_ratio=2, aligned=True)
            pooled[indices] = values.mean(dim=(-2, -1))
        return pooled

    def forward(self, features, rows, levels):
        with torch.autocast(device_type=rows.device.type, enabled=False):
            count = reduce_mean(rows.new_tensor(float(len(rows)))).clamp(min=1)
            if not len(rows):
                return sum(p.sum() * 0 for p in self.parameters())
            rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
            logits = self.classifier(self.pool_instances(features, rois, levels))
            return self.loss_weight * F.cross_entropy(logits, rows[:, 1].long(), reduction='sum') / count
