"""B: magnitude-aware context + gated shallow detail transport."""
_base_ = '../rtmdet_tiny_dior_300e.py'
custom_imports = dict(imports=['custom_models.magnitude_detail'], allow_failed_imports=False)
model = dict(backbone=dict(out_indices=(1, 2, 3, 4)),
             neck=dict(type='MagnitudeDetailPAFPN', use_context=True,
                       use_detail=True, shallow_channels=128, context_channels=48,
                       num_heads=4, detail_channels=24))
work_dir = _base_.root + '/work_dirs/magnitude_detail_full_640'
