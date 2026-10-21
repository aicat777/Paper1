"""Experiment 2: extend the three-SCDown control to the P3 entry."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown import SCDown, SCDownCSPNeXt


@MODELS.register_module()
class SCDownP3CSPNeXt(SCDownCSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = 'stage2.0'
        original = self.get_submodule(path)
        with torch.random.fork_rng(devices=[]):
            replacement = SCDown.from_conv_module(original)
        self.stage2[0] = replacement
        self.scdown_replacements += (path,)
        self._scdown_initialization_slots += ((path, original, replacement),)
