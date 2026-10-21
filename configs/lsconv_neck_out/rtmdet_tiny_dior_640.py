"""Three-site LSConv backbone + LSConv-before-projection neck outputs."""
_base_ = '../lsconv/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.lsconv', 'custom_models.lsconv_neck_out'],
                      allow_failed_imports=False)
model = dict(neck=dict(type='LSConvOutputPAFPN'))
work_dir = _base_.root + '/work_dirs/lsconv_neck_out_rtmdet_tiny_dior_640_gpu1_bs36'
