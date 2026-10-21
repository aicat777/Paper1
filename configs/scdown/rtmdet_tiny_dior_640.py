"""SCDown only at two late backbone transitions and the final PAN transition."""
_base_ = '../rtmdet_tiny_dior_300e.py'

custom_imports = dict(imports=['custom_models.scdown'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownCSPNeXt'), neck=dict(type='SCDownPAFPN'))

train_batch_size_per_gpu = 36
base_lr = 0.004 * train_batch_size_per_gpu / 256
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True, convert_to_iter_based=True)]
work_dir = _base_.root + '/work_dirs/scdown_rtmdet_tiny_dior_640_bs36'
