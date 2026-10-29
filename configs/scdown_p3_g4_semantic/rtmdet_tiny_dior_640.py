"""Four SCDown transitions, seven G4 CSP convolutions, original all-GT ROI CE."""
_base_ = '../scdown_p3_g4/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.scdown_p3_g4_semantic'], allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='InstanceSemanticRTMDetHead', semantic_loss_weight=.1,
    semantic_finest_scale=56,
    head_module=dict(type='SemanticFeatureHeadModule')))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_rtmdet_tiny_dior_640_bs36'
