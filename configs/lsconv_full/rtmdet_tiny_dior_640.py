"""Expanded LSConv, DIOR 640, scratch, microbatch12 x accumulation3."""
_base_ = '../rtmdet_tiny_dior_300e.py'

custom_imports = dict(imports=['custom_models.lsconv_full'], allow_failed_imports=False)
model = dict(
    backbone=dict(type='FullLSConvCSPNeXt'),
    neck=dict(type='FullLSConvPAFPN'),
    bbox_head=dict(head_module=dict(type='FullLSConvSepBNHeadModule')))

train_batch_size_per_gpu = 12
accumulation = 3
effective_batch_size = train_batch_size_per_gpu * accumulation
base_lr = 0.004 * effective_batch_size / 256
train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(accumulative_counts=accumulation, optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False,
         begin=0, end=1000 * accumulation),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True,
         convert_to_iter_based=True)]
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         interval=accumulation, gamma=2000 * accumulation,
         update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=280,
         switch_pipeline=_base_.train_pipeline_stage2)]
work_dir = _base_.root + '/work_dirs/lsconv_full_rtmdet_tiny_dior_640_gpu1_bs12x3'
