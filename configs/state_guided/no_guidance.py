"""A control: both modules, without the direct state-to-upsampler connection."""
_base_ = './full.py'
model = dict(neck=dict(use_state_guidance=False))
work_dir = _base_.root + '/work_dirs/state_guided_no_guidance_640'
