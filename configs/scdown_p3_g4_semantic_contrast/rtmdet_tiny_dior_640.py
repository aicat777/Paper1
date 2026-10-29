"""GSCE + unchanged ROI CE plus all-GT-normalized supervised instance contrast."""
_base_ = '../scdown_p3_g4_semantic_gradscale/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_g4_semantic',
                             'custom_models.instance_semantic_contrast'], allow_failed_imports=False)
model = dict(bbox_head=dict(type='ContrastiveROISemanticRTMDetHead',
                           semantic_contrast_weight=.1, semantic_contrast_temperature=.2,
                           semantic_contrast_channels=128, semantic_pair_chunk_size=128))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_contrast_rtmdet_tiny_dior_640_bs36'
