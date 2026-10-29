"""One head/DSL pass, unchanged all-GT ROI CE, and selective feature transfer."""
from mmdet.models.utils import unpack_gt_instances
from mmyolo.registry import MODELS

from custom_models.roi_supervision_common.head import SelectedROISemanticHead
from .supervision import RegionToPointTransfer, original_roi_readout, valid_rows_and_lookup


@MODELS.register_module()
class TransferInstanceSemanticRTMDetHead(SelectedROISemanticHead):
    def __init__(self, *args, transfer_loss_weight=.1, transfer_min_iou=.5,
                 transfer_temperature=1., **kwargs):
        # Reuse only the validated indexed DSL/main-loss plumbing. This head
        # does not invoke SelectedROISemanticHead.loss or filter the ROI task.
        super().__init__(*args, **kwargs)
        self.local_transfer = RegionToPointTransfer(transfer_loss_weight,
                                                   transfer_min_iou, transfer_temperature)

    def loss(self, x, batch_data_samples):
        cls, reg, features = self.head_module(x, return_cls_features=True)
        if isinstance(batch_data_samples, list):
            gt, ignored, metas = unpack_gt_instances(batch_data_samples)
        else:
            gt, metas = batch_data_samples['bboxes_labels'], batch_data_samples['img_metas']
            ignored = None
        losses, padded, assigned = self._detection_losses(cls, reg, gt, metas, ignored)
        rows, lookup = valid_rows_and_lookup(gt, padded)
        losses['loss_instance_semantic'], roi_logits = original_roi_readout(
            self.instance_semantic, features, rows)
        loss, stats = self.local_transfer(features, cls, assigned, rows, lookup,
                                          roi_logits, self.instance_semantic.classifier)
        losses['loss_instance_transfer'] = loss
        losses.update(stats)
        return losses
