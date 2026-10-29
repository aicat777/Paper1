"""GSCE ROI: backbone auxiliary gradient zero; neck/classification stay .25."""
_base_ = '../scdown_p3_g4_semantic_gradscale/rtmdet_tiny_dior_640.py'

custom_imports = dict(
    imports=['custom_models.scdown_p3_g4_semantic',
             'custom_models.instance_semantic_gradscale',
             'custom_models.instance_semantic_neckhead'], allow_failed_imports=False)
model = dict(type='NeckHeadROISemanticDetector')
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_semantic_neckhead_rtmdet_tiny_dior_640_bs36'
