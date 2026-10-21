"""All-GT ROI semantic loss + per-instance ranking on actual detector logits."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.instance_semantic_rank'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(type='InstanceSemanticRankRTMDetHead', rank_loss_weight=.1),
    train_cfg=dict(assigner=dict(type='IndexedDynamicSoftLabelAssigner')))

work_dir = _base_.root + '/work_dirs/instance_semantic_rank_rtmdet_tiny_dior_640_bs36'
