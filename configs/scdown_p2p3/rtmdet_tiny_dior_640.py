"""Experiment 3: SCDown3 + backbone P3/P2 entries; no grouped CSPs."""
_base_ = '../scdown/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_p2p3'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownP2P3CSPNeXt'))
work_dir = _base_.root + '/work_dirs/scdown_p2p3_rtmdet_tiny_dior_640_bs36'
