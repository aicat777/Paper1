"""Keep all-GT ROI CE unchanged and add normalized instance relations."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mmdet.utils import reduce_mean

from custom_models.instance_semantic_gradscale.roi import GradientScaledInstanceSemanticLoss
from .contrastive import supervised_contrastive_sum


class ContrastiveInstanceSemanticLoss(GradientScaledInstanceSemanticLoss):
    def __init__(self, reference, feature_gradient_scale=.25, contrast_weight=.1,
                 temperature=.2, embedding_channels=128, pair_chunk_size=128):
        super().__init__(reference, feature_gradient_scale)
        if not math.isfinite(contrast_weight) or contrast_weight < 0:
            raise ValueError('Contrast weight must be finite and nonnegative')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Contrast temperature must be finite and positive')
        if not isinstance(embedding_channels, int) or embedding_channels < 1:
            raise ValueError('Contrast width must be a positive integer')
        if not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
            raise ValueError('Pair chunk size must be a positive integer')
        self.contrast_weight, self.temperature = contrast_weight, temperature
        self.pair_chunk_size = pair_chunk_size
        channels = self.classifier.in_features
        with torch.random.fork_rng(devices=[]):
            self.projector = nn.Sequential(
                nn.Linear(channels, channels), nn.ReLU(),
                nn.Linear(channels, embedding_channels))
            for layer in self.projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, features, rows):
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(dim=1)]
            normalizer = reduce_mean(features[0].new_tensor(
                float(len(rows)), dtype=torch.float32)).clamp(min=1)
            if len(rows) == 0:
                shared_zero = sum(x.float().sum() * 0 for x in features)
                ce = shared_zero + sum(p.sum() * 0 for p in self.classifier.parameters())
                contrast = shared_zero + sum(p.sum() * 0 for p in self.projector.parameters())
                anchors = normalizer.new_zeros(())
            else:
                rois = torch.cat((rows[:, :1], rows[:, 2:6]), dim=1)
                # One pooled descriptor and ONE gradient scaling shared by both
                # objectives. The classifier and projector themselves are unscaled.
                vectors = self.pool_instances(features, rois)
                labels = rows[:, 1].long()
                ce = self.loss_weight * F.cross_entropy(
                    self.classifier(vectors), labels, reduction='sum') / normalizer
                if self.contrast_weight:
                    embedding = F.normalize(self.projector(vectors), dim=1, eps=1e-12)
                    contrast = self.contrast_weight * supervised_contrastive_sum(
                        embedding, labels, self.temperature, self.pair_chunk_size) / normalizer
                else:
                    contrast = vectors.sum() * 0 + sum(p.sum() * 0 for p in self.projector.parameters())
                counts = torch.bincount(labels, minlength=self.classifier.out_features)
                anchors = (counts[labels] > 1).sum().to(normalizer)
            return dict(loss_instance_semantic=ce, loss_instance_contrast=contrast,
                        semantic_contrast_anchors=anchors.detach(),
                        semantic_all_gt=normalizer.new_tensor(float(len(rows))))
