"""B ablation: gated shallow detail, without global magnitude attention."""
_base_ = './full.py'
model = dict(neck=dict(use_context=False))
work_dir = _base_.root + '/work_dirs/magnitude_detail_detail_only_640'
