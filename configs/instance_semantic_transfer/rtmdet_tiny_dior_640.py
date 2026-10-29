"""Original all-GT ROI CE + selective region-to-point semantic transfer."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.instance_semantic_transfer'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(type='TransferInstanceSemanticRTMDetHead',
                   transfer_loss_weight=.1, transfer_min_iou=.5, transfer_temperature=1.),
    train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/instance_semantic_transfer_rtmdet_tiny_dior_640_bs36'
