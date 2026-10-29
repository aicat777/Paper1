"""Cross-level supervision, without fusing descriptors or tripling the loss."""
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmdet.utils import reduce_mean

from custom_models.instance_semantic_gradscale.roi import (
    GradientScaledInstanceSemanticLoss, scale_feature_gradient)


class AllLevelInstanceSemanticLoss(GradientScaledInstanceSemanticLoss):
    def pool_instances(self, features, rois):
        if len(features) != len(self.strides):
            raise ValueError('A stride is required for every feature map')
        if not bool((rois[:, 3:5] > rois[:, 1:3]).all()):
            raise ValueError('Training ROIs must have positive area')
        vectors = []
        for feature, stride in zip(features, self.strides):
            selected = rois.float().clone()
            selected[:, 1:] = selected[:, 1:] / stride + .5
            value = roi_align(feature.float(), selected, self.output_size,
                              spatial_scale=1., sampling_ratio=2, aligned=True)
            vectors.append(value.mean(dim=(-2, -1)))
        # [GT, level, channel]; classifier parameters are shared across levels.
        return scale_feature_gradient(torch.stack(vectors, dim=1),
                                      self.feature_gradient_scale)

    def forward(self, features, rows):
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(dim=1)]
            normalizer = reduce_mean(features[0].new_tensor(
                float(len(rows)), dtype=torch.float32)).clamp(min=1)
            if len(rows) == 0:
                return (sum(x.float().sum() * 0 for x in features)
                        + sum(p.sum() * 0 for p in self.parameters()))
            rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
            vectors = self.pool_instances(features, rois)
            levels = len(self.strides)
            logits = self.classifier(vectors.reshape(-1, vectors.shape[-1]))
            labels = rows[:, 1].long().repeat_interleave(levels)
            return self.loss_weight * F.cross_entropy(
                logits, labels, reduction='sum') / (normalizer * levels)
