"""A ablation: lateral-guided local upsampling without hidden states."""
_base_ = './full.py'
model = dict(neck=dict(use_context=False))
work_dir = _base_.root + '/work_dirs/state_guided_upsample_only_640'
