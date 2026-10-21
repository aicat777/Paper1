"""One bilinear sample at each GT center; keep the original semantic task."""
import torch

from custom_models.instance_semantic.roi import InstanceSemanticLoss


class CenterInstanceSemanticLoss(InstanceSemanticLoss):
    def pool_instances(self, features, rois):
        if len(features) != len(self.strides):
            raise ValueError('A stride is required for every feature map')
        sizes = rois[:, 3:5] - rois[:, 1:3]
        if not bool((sizes > 0).all()):
            raise ValueError('Training ROIs must have positive area')
        levels = torch.floor(torch.log2(sizes.prod(1).sqrt() / self.finest_scale + 1e-6))
        levels = levels.clamp(0, len(features) - 1).long()
        pooled = features[0].new_zeros((len(rois), features[0].shape[1]), dtype=torch.float32)
        for level, (feature, stride) in enumerate(zip(features, self.strides)):
            selected = (levels == level).nonzero().flatten()
            if not len(selected):
                continue
            rows = rois[selected]
            # RTMDet offset-zero priors: feature index x is at input x*stride.
            xy = (rows[:, 1:3] + rows[:, 3:5]).float() / (2 * stride)
            _, channels, height, width = feature.shape
            x, y = xy[:, 0].clamp(0, width - 1), xy[:, 1].clamp(0, height - 1)
            x0, y0 = x.floor().long(), y.floor().long()
            x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
            dx, dy = (x - x0)[:, None], (y - y0)[:, None]
            offset = rows[:, 0].long() * height * width
            flat = feature.float().permute(0, 2, 3, 1).reshape(-1, channels)
            # Gather four vectors, never a whole feature map per GT. This is
            # important for dense batches (e.g. 36 images * 1200 GTs).
            pooled[selected] = (
                flat[offset + y0 * width + x0] * (1 - dx) * (1 - dy)
                + flat[offset + y0 * width + x1] * dx * (1 - dy)
                + flat[offset + y1 * width + x0] * (1 - dx) * dy
                + flat[offset + y1 * width + x1] * dx * dy)
        return pooled
