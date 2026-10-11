"""C control: both modules, without direct band statistics in the upsampler."""
_base_ = './full.py'
model = dict(neck=dict(use_frequency_guidance=False))
work_dir = _base_.root + '/work_dirs/frequency_guided_no_guidance_640'
