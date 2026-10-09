"""One-epoch training-only speed trial from scratch."""
_base_ = './rtmdet_tiny_dior_300e.py'

max_epochs = 1
train_cfg = dict(max_epochs=1, val_interval=2, dynamic_intervals=[])
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
param_scheduler = [dict(type='LinearLR', start_factor=1.0e-5,
                       by_epoch=False, begin=0, end=1000)]
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
                    update_buffers=True, strict_load=False, priority=49)]
default_hooks = dict(logger=dict(interval=20),
                     checkpoint=dict(interval=1, max_keep_ckpts=1,
                                     save_best=None, save_last=True))
