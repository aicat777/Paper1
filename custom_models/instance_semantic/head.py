"""Ordinary RTMDet inference; one extra ROI classification loss in training."""
import torch
from mmdet.models.utils import unpack_gt_instances
from mmyolo.models.dense_heads.rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS

from .roi import InstanceSemanticLoss


@MODELS.register_module()
class SemanticFeatureHeadModule(RTMDetSepBNHeadModule):
    def forward(self, feats, return_cls_features=False):
        if not return_cls_features:
            return super().forward(feats)
        cls_scores, bbox_preds, cls_features = [], [], []
        for idx, feature in enumerate(feats):
            cls_feature, reg_feature = feature, feature
            for layer in self.cls_convs[idx]:
                cls_feature = layer(cls_feature)
            for layer in self.reg_convs[idx]:
                reg_feature = layer(reg_feature)
            cls_features.append(cls_feature)
            cls_scores.append(self.rtm_cls[idx](cls_feature))
            bbox_preds.append(self.rtm_reg[idx](reg_feature))
        return tuple(cls_scores), tuple(bbox_preds), tuple(cls_features)


@MODELS.register_module()
class InstanceSemanticRTMDetHead(RTMDetHead):
    def __init__(self, *args, semantic_loss_weight=.1, semantic_finest_scale=56, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.head_module, SemanticFeatureHeadModule):
            raise TypeError('The auxiliary loss needs classification-tower features')
        if self.prior_generator.offset != 0:
            raise ValueError('ROI coordinates are defined for RTMDet offset-zero priors')
        with torch.random.fork_rng(devices=[]):
            self.instance_semantic = InstanceSemanticLoss(
                self.head_module.feat_channels, self.num_classes, self.featmap_strides,
                semantic_loss_weight, semantic_finest_scale)

    def loss(self, x, batch_data_samples):
        cls_scores, bbox_preds, features = self.head_module(x, return_cls_features=True)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses = self.loss_by_feat(cls_scores, bbox_preds, gt, metas, ignored)
        if isinstance(gt, torch.Tensor):
            rows = gt
        else:
            padded = gt_instances_preprocess(gt, len(metas))
            valid = (padded[..., 3:5] > padded[..., 1:3]).all(dim=-1)
            batch_index, gt_index = valid.nonzero(as_tuple=True)
            rows = torch.cat((batch_index[:, None].to(padded), padded[batch_index, gt_index]), dim=1)
        losses['loss_instance_semantic'] = self.instance_semantic(features, rows)
        return losses
