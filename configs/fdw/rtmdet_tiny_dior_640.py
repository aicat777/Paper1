"""Independent FDW-only backbone experiment, DIOR 640, scratch, batch36."""
_base_ = '../rtmdet_tiny_dior_300e.py'

custom_imports = dict(imports=['custom_models.fdw'], allow_failed_imports=False)
model = dict(backbone=dict(type='FDWCSPNeXt', fdw_stages=(2, 3, 4), fdw_kernel_num=16))

train_batch_size_per_gpu = 36
base_lr = 0.004 * train_batch_size_per_gpu / 256
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True,
         convert_to_iter_based=True)]
work_dir = _base_.root + '/work_dirs/fdw_rtmdet_tiny_dior_640_gpu1_bs36'
