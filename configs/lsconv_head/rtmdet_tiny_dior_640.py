"""Three-site LSConv backbone + second head layer LSBlocks; scratch, batch36."""
_base_ = '../lsconv/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.lsconv', 'custom_models.lsconv_head'],
                      allow_failed_imports=False)
model = dict(bbox_head=dict(head_module=dict(type='LSConvSecondLayerHeadModule')))
work_dir = _base_.root + '/work_dirs/lsconv_head_rtmdet_tiny_dior_640_gpu1_bs36'
