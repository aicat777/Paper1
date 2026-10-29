"""All-GT ROI CE + instance-balanced QFL on real detector outputs."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.instance_semantic_ibqfl'], allow_failed_imports=False)
model = dict(
    bbox_head=dict(type='InstanceBalancedSemanticRTMDetHead', instance_qfl_weight=.1),
    train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/instance_semantic_ibqfl_rtmdet_tiny_dior_640_bs36'
