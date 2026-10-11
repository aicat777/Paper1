"""A ablation: compressed context with the original nearest upsampling."""
_base_ = './full.py'
model = dict(neck=dict(use_upsample=False))
work_dir = _base_.root + '/work_dirs/state_guided_context_only'
