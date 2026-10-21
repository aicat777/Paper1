"""Experiment 2: SCDown3 + the backbone P3 entry; no grouped CSPs."""
_base_ = '../scdown/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p3'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP3CSPNeXt'))
work_dir = _base_.root + '/work_dirs/scdown_p3_rtmdet_tiny_dior_640_bs36'
