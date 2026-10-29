"""Original ROI: full classifier learning, one-quarter feature gradients."""
_base_ = '../instance_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.instance_semantic_gradscale'], allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='GradientScaledSemanticRTMDetHead', semantic_feature_gradient_scale=.25))
work_dir = _base_.root + '/work_dirs/instance_semantic_gradscale_rtmdet_tiny_dior_640_bs36'
