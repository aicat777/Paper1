"""B control: ordinary linear attention with the same context parameters."""
_base_ = './context_only.py'
model = dict(neck=dict(magnitude_aware=False))
work_dir = _base_.root + '/work_dirs/magnitude_detail_linear_context_only'
