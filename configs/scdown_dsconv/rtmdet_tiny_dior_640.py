"""SCDown3 + DW3x3/BN/SiLU -> PW1x1/BN/SiLU at seven CSP conv1 sites."""
_base_ = '../scdown/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_dsconv'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownDSConvCSPNeXt'), neck=dict(type='SCDownDSConvPAFPN'))
work_dir = _base_.root + '/work_dirs/scdown_dsconv_rtmdet_tiny_dior_640_bs36'
