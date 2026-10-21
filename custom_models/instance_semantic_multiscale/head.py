"""Unchanged detection head; auxiliary views execute only selected cls towers."""
import torch
from mmyolo.models.dense_heads.rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from mmyolo.registry import MODELS

from .roi import CropSemanticLoss


@MODELS.register_module()
class CropSemanticHeadModule(RTMDetSepBNHeadModule):
    def classification_features(self, features, levels):
        result = {}
        for level in levels.unique(sorted=True).tolist():
            value = features[level]
            for layer in self.cls_convs[level]:
                value = layer(value)
            result[level] = value
        return result


@MODELS.register_module()
class CropSemanticRTMDetHead(RTMDetHead):
    def __init__(self, *args, semantic_loss_weight=.1, semantic_finest_scale=56, **kwargs):
        super().__init__(*args, **kwargs)
        if self.prior_generator.offset != 0:
            raise ValueError('The crop ROI adapter requires offset-zero RTMDet priors')
        with torch.random.fork_rng(devices=[]):
            self.instance_semantic = CropSemanticLoss(self.head_module.feat_channels,
                self.num_classes, self.featmap_strides, semantic_loss_weight, semantic_finest_scale)

    def crop_loss(self, features, rows):
        module = self.instance_semantic
        levels = module.levels(rows[:, 2:6])
        classification = self.head_module.classification_features(features, levels)
        return module(classification, rows, levels)
