"""C ablation: lateral-guided upsampling without frequency conditioning."""
_base_ = './full.py'
model = dict(neck=dict(use_frequency=False))
work_dir = _base_.root + '/work_dirs/frequency_guided_upsample_only_640'
