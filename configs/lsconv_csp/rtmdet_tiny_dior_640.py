"""Legacy LSConv conv1 adapter expanded to eight CSP blocks, real batch36."""
_base_ = '../lsconv/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.lsconv_csp'], allow_failed_imports=False)
model = dict(backbone=dict(type='CSPOnlyLSConvBackbone', lsconv_stages=(1, 2, 3, 4)),
             neck=dict(type='CSPOnlyLSConvPAFPN'))
work_dir = _base_.root + '/work_dirs/lsconv_csp_rtmdet_tiny_dior_640_bs36'
