"""Separate QFL's IoU quality from tiny-object regression weighting."""
import torch
from torch.nn import functional as F
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import reduce_mean
from mmyolo.models.dense_heads.rtmdet_head import RTMDetHead
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS


@MODELS.register_module()
class TinyRescueRTMDetHead(RTMDetHead):
    def losses_from_assignment(self, cls_preds, bbox_preds, result, prior_strides):
        labels = result['assigned_labels'].reshape(-1)
        quality = result['assign_metrics'].reshape(-1)
        weights = result['regression_weights'].reshape(-1)
        targets = result['assigned_bboxes'].reshape(-1, 4)
        positive = ((labels >= 0) & (labels < self.num_classes)).nonzero().flatten()
        # Quality targets AND classification normalization remain the baseline's.
        # Regression uses its own new sum of weights; no accidental amplification
        # by normalizing the floor-weighted terms with the old near-zero IoUs.
        cls_normalizer = reduce_mean(quality.sum()).clamp(min=1).item()
        reg_normalizer = reduce_mean(weights.sum()).clamp(min=1).item()
        loss_cls = self.loss_cls(cls_preds, (labels, quality),
                                 result['assigned_labels_weights'].reshape(-1),
                                 avg_factor=cls_normalizer)
        deficit = weights - quality
        bootstrap = (deficit > 0).nonzero().flatten()
        # A linear ltrb head can initially predict reversed corners. GIoU is
        # not defined for those boxes (its loss can even become negative).
        # Preserve the original IoU-weighted GIoU and supply ONLY the missing
        # weight through SmoothL1 on signed edge coordinates in stride units.
        # Never sort/clamp the prediction: that would change what each edge
        # means and could send gradients in the wrong direction.
        giou_positive = (quality > 0).nonzero().flatten() if bootstrap.numel() else positive
        loss_bbox = bbox_preds.sum() * 0
        if giou_positive.numel():
            loss_bbox = self.loss_bbox(bbox_preds[giou_positive], targets[giou_positive],
                                       weight=quality[giou_positive], avg_factor=reg_normalizer)
        if bootstrap.numel():
            stride = prior_strides.reshape(-1, 1)[bootstrap]
            edge_error = F.smooth_l1_loss(bbox_preds[bootstrap] / stride,
                                          targets[bootstrap] / stride, beta=1., reduction='none').mean(-1)
            loss_bbox = loss_bbox + self.loss_bbox.loss_weight * (deficit[bootstrap] * edge_error).sum() / reg_normalizer
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def loss_by_feat(self, cls_scores, bbox_preds, batch_gt_instances,
                     batch_img_metas, batch_gt_instances_ignore=None):
        batch = len(batch_img_metas)
        shapes = [value.shape[-2:] for value in cls_scores]
        if len(shapes) != self.prior_generator.num_levels:
            raise ValueError('Prediction maps must match the prior strides')
        gt = gt_instances_preprocess(batch_gt_instances, batch)
        labels, boxes = gt[..., :1], gt[..., 1:]
        valid = (boxes.sum(-1, keepdim=True) > 0).float()
        if shapes != self.featmap_sizes_train:
            self.featmap_sizes_train = shapes
            self.flatten_priors_train = torch.cat(self.prior_generator.grid_priors(
                shapes, device=cls_scores[0].device, with_stride=True))
        logits = torch.cat([x.permute(0, 2, 3, 1).reshape(batch, -1, self.cls_out_channels)
                            for x in cls_scores], 1).contiguous()
        distances = torch.cat([x.permute(0, 2, 3, 1).reshape(batch, -1, 4) for x in bbox_preds], 1)
        distances = distances * self.flatten_priors_train[..., -1, None]
        decoded = distance2bbox(self.flatten_priors_train[..., :2], distances)
        result = self.assigner(decoded.detach(), logits.detach(), self.flatten_priors_train,
                               labels, boxes, valid)
        # Signed l/t/r/b are deliberately retained: rescued points can lie just
        # outside a sub-stride box. Clamping distances would make it unreachable.
        return self.losses_from_assignment(logits.reshape(-1, self.num_classes),
                                            decoded.reshape(-1, 4), result,
                                            self.flatten_priors_train[:, -1].expand(batch, -1))
