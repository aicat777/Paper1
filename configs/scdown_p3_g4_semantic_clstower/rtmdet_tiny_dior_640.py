"""GSCE ROI: classification-tower auxiliary gradient .25, backbone/neck zero."""
_base_ = '../scdown_p3_g4_semantic_neckhead/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.scdown_p3_g4_semantic',
             'custom_models.instance_semantic_gradscale',
             'custom_models.instance_semantic_clstower'], allow_failed_imports=False)
model = dict(type='ClassificationOnlyROISemanticDetector')
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_clstower_rtmdet_tiny_dior_640_bs36'
