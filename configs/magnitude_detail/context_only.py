"""B ablation: magnitude-aware context; C2 is returned but not consumed."""
_base_ = './full.py'
model = dict(neck=dict(use_detail=False))
work_dir = _base_.root + '/work_dirs/magnitude_detail_context_only'
