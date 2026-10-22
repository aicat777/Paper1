"""Four SCDown transitions + the same seven groups=4 CSP conv1 sites."""
_base_ = '../scdown_p3/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_g4'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP3G4CSPNeXt'), neck=dict(type='SCDownG4PAFPN'))
work_dir = _base_.root + '/work_dirs/scdown_p3_g4_rtmdet_tiny_dior_640_bs36'
