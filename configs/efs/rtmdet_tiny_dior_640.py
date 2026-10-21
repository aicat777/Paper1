"""EFS migration: expose C2 and predict jointly on P2/P3/P4/P5.

Retains DM-EFS's expanded-feature pathway without its DFM controller.
Uses the official CSPNeXtPAFPN and RTMDet head; no other candidate modules.
"""

_base_ = '../rtmdet_tiny_dior_300e.py'

# P2 matching needs more memory. Accumulate four microbatches to retain the
# completed GPU1 baseline's effective batch and learning-rate budget.
train_batch_size_per_gpu = 9
accumulative_counts = 4
effective_batch_size = train_batch_size_per_gpu * accumulative_counts
base_lr = 0.004 * effective_batch_size / 256
strides = [4, 8, 16, 32]

model = dict(
    backbone=dict(out_indices=(1, 2, 3, 4)),
    # CSPNeXt-tiny's 0.375 width multiplier gives [48, 96, 192, 384].
    neck=dict(in_channels=[128, 256, 512, 1024]),
    bbox_head=dict(
        head_module=dict(featmap_strides=strides),
        prior_generator=dict(strides=strides)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
optim_wrapper = dict(accumulative_counts=accumulative_counts,
                     optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0,
         end=1000 * accumulative_counts),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=150, end=300, T_max=150, by_epoch=True,
         convert_to_iter_based=True)]

# ExpMomentumEMA counts microiterations. Match its update interval and early
# momentum schedule to the accumulation factor (MMYOLO default gamma=2000).
custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         gamma=2000 * accumulative_counts, interval=accumulative_counts,
         update_buffers=True, strict_load=False, priority=49),
    dict(type='mmdet.PipelineSwitchHook', switch_epoch=280,
         switch_pipeline=_base_.train_pipeline_stage2)
]

work_dir = _base_.root + '/work_dirs/efs_rtmdet_tiny_dior_640_gpu1_bs9x4'
