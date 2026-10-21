"""One official DSL assignment and one head pass; reuse ROI parameters unchanged."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import reduce_mean
from mmyolo.models.utils import gt_instances_preprocess

from custom_models.instance_semantic.head import InstanceSemanticRTMDetHead
from .assigner import ROISemanticIndexedAssigner
from .supervision import selected_roi_loss, valid_rows_and_matches


class SelectedROISemanticHead(InstanceSemanticRTMDetHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_cfg is not None and not isinstance(self.assigner, ROISemanticIndexedAssigner):
            raise TypeError('ROI selection requires actual GT indices from DSL')

    def loss(self, x, batch_data_samples):
        cls, reg, features = self.head_module(x, return_cls_features=True)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses, padded, assignment = self._detection_losses(cls, reg, gt, metas, ignored)
        rows, matched = valid_rows_and_matches(gt, padded, assignment['assigned_gt_indices'])
        selected = self.selection(matched)
        losses['loss_instance_semantic'] = selected_roi_loss(self.instance_semantic, features, rows, selected)
        # Detached counts expose coverage in formal logs without changing loss.
        losses['semantic_all_gt'] = cls[0].new_tensor(len(rows), dtype=torch.float32)
        losses['semantic_matched_gt'] = matched.sum().float()
        losses['semantic_selected_gt'] = selected.sum().float()
        return losses

    def loss_by_feat(self, cls_scores, bbox_preds, batch_gt_instances,
                     batch_img_metas, batch_gt_instances_ignore=None):
        return self._detection_losses(cls_scores, bbox_preds, batch_gt_instances,
                                       batch_img_metas, batch_gt_instances_ignore)[0]

    def _detection_losses(self, cls_scores, bbox_preds, batch_gt_instances,
                          batch_img_metas, batch_gt_instances_ignore=None):
        # Preserve the operations/order of the local official RTMDetHead.
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels, gt_bboxes = gt_info[:, :, :1], gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            priors = self.prior_generator.grid_priors(
                featmap_sizes, device=cls_scores[0].device, with_stride=True)
            self.flatten_priors_train = torch.cat(priors, dim=0)
        flatten_cls_scores = torch.cat([
            score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for score in cls_scores], 1).contiguous()
        flatten_bboxes = torch.cat([
            pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for pred in bbox_preds], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1, None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2], flatten_bboxes)
        assigned = self.assigner(flatten_bboxes.detach(), flatten_cls_scores.detach(),
                                  self.flatten_priors_train, gt_labels, gt_bboxes, pad_bbox_flag)
        labels = assigned['assigned_labels'].reshape(-1)
        label_weights = assigned['assigned_labels_weights'].reshape(-1)
        targets = assigned['assigned_bboxes'].reshape(-1, 4)
        metrics = assigned['assign_metrics'].reshape(-1)
        cls_preds = flatten_cls_scores.reshape(-1, self.num_classes)
        box_preds = flatten_bboxes.reshape(-1, 4)
        pos = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(1)
        avg_factor = reduce_mean(metrics.sum()).clamp_(min=1).item()
        loss_cls = self.loss_cls(cls_preds, (labels, metrics), label_weights, avg_factor=avg_factor)
        if len(pos) > 0:
            loss_bbox = self.loss_bbox(box_preds[pos], targets[pos], weight=metrics[pos], avg_factor=avg_factor)
        else:
            loss_bbox = box_preds.sum() * 0
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox), gt_info, assigned
