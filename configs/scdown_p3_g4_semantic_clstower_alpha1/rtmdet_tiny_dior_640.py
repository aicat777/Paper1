"""GSCE ROI: full auxiliary gradient on the classification tower only."""
_base_ = '../scdown_p3_g4_semantic_clstower/rtmdet_tiny_dior_640.py'

model = dict(bbox_head=dict(semantic_feature_gradient_scale=1.0))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_clstower_alpha1_rtmdet_tiny_dior_640_bs36'
