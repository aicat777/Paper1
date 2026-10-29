"""GSCE + private neck-to-ROI MLP; the main classification tower stays task-specific."""
_base_ = '../scdown_p3_g4_semantic_gradscale/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_g4_semantic',
                             'custom_models.instance_semantic_private'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='PrivateROISemanticRTMDetHead',
                           semantic_private_hidden_channels=192))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_private_rtmdet_tiny_dior_640_bs36'
