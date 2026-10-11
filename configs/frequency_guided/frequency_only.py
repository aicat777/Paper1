"""C ablation: spatial band selection with original nearest upsampling."""
_base_ = './full.py'
model = dict(neck=dict(use_upsample=False))
work_dir = _base_.root + '/work_dirs/frequency_guided_frequency_only_640'
