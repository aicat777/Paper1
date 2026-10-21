"""SCDown3 + groups=4 at the same seven CSP conv1 sites as the G2 run."""
_base_ = '../scdown/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_g4'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownG4CSPNeXt'), neck=dict(type='SCDownG4PAFPN'))
work_dir = _base_.root + '/work_dirs/scdown_g4_rtmdet_tiny_dior_640_bs36'
