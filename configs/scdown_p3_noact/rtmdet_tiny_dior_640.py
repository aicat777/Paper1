"""Control B: remove the intermediate P3 SiLU, leaving both convolutions and BNs."""
_base_ = '../scdown_p3/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_noact'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP3NoActCSPNeXt'))
work_dir = _base_.root + '/work_dirs/scdown_p3_noact_rtmdet_tiny_dior_640_bs36'
