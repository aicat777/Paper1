"""GSCE + all-GT ROI with one-quarter auxiliary feature gradients."""
_base_ = '../scdown_p3_g4_semantic/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.scdown_p3_g4_semantic',
             'custom_models.instance_semantic_gradscale'], allow_failed_imports=False)
model = dict(bbox_head=dict(
    type='GradientScaledSemanticRTMDetHead', semantic_feature_gradient_scale=.25))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_gradscale_rtmdet_tiny_dior_640_bs36'
