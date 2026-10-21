"""Original ROI semantic head, plus an independent loss on its real logits."""
import torch
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import reduce_mean
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS

from custom_models.instance_semantic.head import InstanceSemanticRTMDetHead
from .assigner import IndexedDynamicSoftLabelAssigner
from .loss import InstanceCategoryRankLoss


@MODELS.register_module()
class InstanceSemanticRankRTMDetHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, rank_loss_weight=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_cfg is not None and not isinstance(self.assigner, IndexedDynamicSoftLabelAssigner):
            raise TypeError('Rank supervision requires actual GT indices from the DSL assigner')
        self.instance_rank = InstanceCategoryRankLoss(rank_loss_weight)

    def loss_by_feat(self, cls_scores, bbox_preds, batch_gt_instances,
                     batch_img_metas, batch_gt_instances_ignore=None):
        # Same computation/order as the local official RTMDetHead. Assignment
        # runs exactly once; the added loss reuses its indices and live logits.
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        device = cls_scores[0].device
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
        flatten_cls_scores = torch.cat([
            score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for score in cls_scores], 1).contiguous()
        flatten_bboxes = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for pred in bbox_preds], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1, None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2], flatten_bboxes)
        assigned_result = self.assigner(
            flatten_bboxes.detach(), flatten_cls_scores.detach(),
            self.flatten_priors_train, gt_labels, gt_bboxes, pad_bbox_flag)
        labels = assigned_result['assigned_labels'].reshape(-1)
        label_weights = assigned_result['assigned_labels_weights'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 4)
        assign_metrics = assigned_result['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)
        bbox_preds = flatten_bboxes.reshape(-1, 4)
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(1)
        avg_factor = reduce_mean(assign_metrics.sum()).clamp_(min=1).item()
        loss_cls = self.loss_cls(cls_preds, (labels, assign_metrics), label_weights,
                                 avg_factor=avg_factor)
        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(bbox_preds[pos_inds], bbox_targets[pos_inds],
                                      weight=assign_metrics[pos_inds], avg_factor=avg_factor)
        else:
            loss_bbox = bbox_preds.sum() * 0
        loss_rank = self.instance_rank(flatten_cls_scores,
                                       assigned_result['assigned_gt_indices'], gt_labels)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_instance_rank=loss_rank)
