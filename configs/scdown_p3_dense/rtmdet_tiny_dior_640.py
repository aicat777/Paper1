"""Control A: only the new P3 spatial kernel becomes dense, preserving PW/BN/SiLU."""
_base_ = '../scdown_p3/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3_dense'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP3DenseCSPNeXt'))
work_dir = _base_.root + '/work_dirs/scdown_p3_dense_rtmdet_tiny_dior_640_bs36'
