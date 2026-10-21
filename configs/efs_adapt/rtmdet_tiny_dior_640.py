"""EFS with a preserved P3 lateral and independent P2 detection towers."""

_base_ = '../efs/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.efs_adapt'],
                      allow_failed_imports=False)

model = dict(
    neck=dict(type='EFSPreserveP3PAFPN'),
    bbox_head=dict(head_module=dict(type='EFSP2IndependentHeadModule')))

# Same scratch protocol and microbatch9 x accumulation4 as corrected EFS.
work_dir = _base_.root + '/work_dirs/efs_adapt_rtmdet_tiny_dior_640_gpu1_bs9x4'
