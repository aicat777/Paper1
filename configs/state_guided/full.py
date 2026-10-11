"""A: compressed hidden states + state-guided local upsampling."""
_base_ = '../rtmdet_tiny_dior_300e.py'
custom_imports = dict(imports=['custom_models.state_guided'], allow_failed_imports=False)
model = dict(neck=dict(type='StateGuidedPAFPN', use_context=True,
                      use_upsample=True, state_channels=48, num_states=16))
work_dir = _base_.root + '/work_dirs/state_guided_full'
