"""Experiment 1: SCDown3 + seven P3-P5/neck CSP conv1 groups=2."""
_base_ = '../scdown/rtmdet_tiny_dior_640.py'

custom_imports = dict(imports=['custom_models.scdown_g2'], allow_failed_imports=False)
model = dict(backbone=dict(type='SCDownG2CSPNeXt'), neck=dict(type='SCDownG2PAFPN'))
work_dir = _base_.root + '/work_dirs/scdown_g2_rtmdet_tiny_dior_640_bs36'
