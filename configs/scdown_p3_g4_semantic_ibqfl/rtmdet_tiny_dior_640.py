"""GSCE (SCDown4 + seven G4 CSPs) + unchanged ROI + instance-balanced QFL."""
_base_ = '../scdown_p3_g4_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.scdown_p3_g4', 'custom_models.instance_semantic_ibqfl'],
    allow_failed_imports=False)
model = dict(
    bbox_head=dict(type='InstanceBalancedSemanticRTMDetHead', instance_qfl_weight=.1),
    train_cfg=dict(assigner=dict(type='ROISemanticIndexedAssigner')))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_ibqfl_rtmdet_tiny_dior_640_bs36'
