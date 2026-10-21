"""Three-site LSConv backbone + four whole neck LSBlocks; scratch, batch36."""
_base_ = '../lsconv/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.lsconv', 'custom_models.lsconv_neck'],
                      allow_failed_imports=False)
model = dict(neck=dict(type='LSConvNeckPAFPN'))
work_dir = _base_.root + '/work_dirs/lsconv_neck_rtmdet_tiny_dior_640_gpu1_bs36'
