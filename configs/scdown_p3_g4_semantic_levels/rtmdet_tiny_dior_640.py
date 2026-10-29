"""GSCE + the same GT classified separately at P3/P4/P5, averaged across levels."""
_base_ = '../scdown_p3_g4_semantic_gradscale/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_g4_semantic',
                             'custom_models.instance_semantic_levels'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='AllLevelROISemanticRTMDetHead'))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_levels_rtmdet_tiny_dior_640_bs36'
