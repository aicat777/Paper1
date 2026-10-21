"""Instance-balanced, training-only supervision on classification features."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmdet.utils import reduce_mean


class InstanceSemanticLoss(nn.Module):
    def __init__(self, channels, num_classes, strides=(8, 16, 32),
                 loss_weight=.1, finest_scale=56, output_size=3):
        super().__init__()
        if loss_weight < 0 or finest_scale <= 0 or output_size < 1:
            raise ValueError('Invalid instance supervision settings')
        self.strides = tuple(strides)
        self.loss_weight = loss_weight
        self.finest_scale, self.output_size = finest_scale, output_size
        self.classifier = nn.Linear(channels, num_classes)
        nn.init.normal_(self.classifier.weight, std=.01)
        nn.init.zeros_(self.classifier.bias)

    def pool_instances(self, features, rois):
        """rois: [batch_index, x1, y1, x2, y2] in augmented input pixels.

        RTMDet priors have offset=0: feature index j corresponds to j*stride.
        torchvision aligned=True subtracts .5, so add .5 after dividing the
        coordinates by stride. This gives samples at x/stride without its
        legacy minimum-one-cell ROI width (which distorts sub-stride boxes).
        """
        if len(features) != len(self.strides):
            raise ValueError('A stride is required for every feature map')
        widths = rois[:, 3:5] - rois[:, 1:3]
        if not bool((widths > 0).all()):
            raise ValueError('Training ROIs must have positive area')
        scale = widths.prod(dim=1).sqrt()
        levels = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        levels = levels.clamp(0, len(features) - 1).long()
        pooled = features[0].new_zeros((len(rois), features[0].shape[1]), dtype=torch.float32)
        for level, (feature, stride) in enumerate(zip(features, self.strides)):
            indices = (levels == level).nonzero().flatten()
            if indices.numel() == 0:
                continue
            selected = rois[indices].float().clone()
            selected[:, 1:] = selected[:, 1:] / stride + .5
            values = roi_align(feature.float(), selected, self.output_size, spatial_scale=1.,
                               sampling_ratio=2, aligned=True)
            pooled[indices] = values.mean(dim=(-2, -1))
        return pooled

    def forward(self, features, rows):
        # No random instance cap: every valid, post-augmentation GT contributes
        # once, regardless of how many detector priors it receives.
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            # Clipping/flipping can leave an empty ROI in the augmented batch.
            # Exclude only nonpositive-area rows from this auxiliary objective;
            # the detector's GT and losses remain unchanged. Keep every
            # positive-area box, including boxes smaller than a feature cell.
            rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(dim=1)]
            normalizer = reduce_mean(features[0].new_tensor(float(len(rows)), dtype=torch.float32)).clamp(min=1)
            if len(rows) == 0:
                return (sum(x.float().sum() * 0 for x in features)
                        + sum(p.sum() * 0 for p in self.parameters()))
            rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
            logits = self.classifier(self.pool_instances(features, rois))
            return self.loss_weight * F.cross_entropy(logits, rows[:, 1].long(), reduction='sum') / normalizer
