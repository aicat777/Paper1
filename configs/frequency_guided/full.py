"""C: spatial band selection + frequency-conditioned local upsampling."""
_base_ = '../rtmdet_tiny_dior_300e.py'
custom_imports = dict(imports=['custom_models.frequency_guided'], allow_failed_imports=False)
model = dict(neck=dict(type='FrequencyGuidedPAFPN', use_frequency=True,
                      use_upsample=True, frequency_groups=4))
work_dir = _base_.root + '/work_dirs/frequency_guided_full'
