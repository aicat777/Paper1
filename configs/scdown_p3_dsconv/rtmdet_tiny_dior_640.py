"""Four SCDown transitions + standard DW3x3 -> PW1x1 at seven CSP sites."""
_base_ = '../scdown_p3/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_dsconv'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP3DSConvCSPNeXt'), neck=dict(type='SCDownDSConvPAFPN'))
work_dir = _base_.root + '/work_dirs/scdown_p3_dsconv_rtmdet_tiny_dior_640_bs36'
