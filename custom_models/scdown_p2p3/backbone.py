"""Experiment 3: extend the three-SCDown control to P3 and P2 entries."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown import SCDown, SCDownCSPNeXt


@MODELS.register_module()
class SCDownP2P3CSPNeXt(SCDownCSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        slots = []
        # P3 is constructed first, matching experiment 2's initialization.
        with torch.random.fork_rng(devices=[]):
            for stage in (2, 1):
                path = f'stage{stage}.0'
                original = self.get_submodule(path)
                replacement = SCDown.from_conv_module(original)
                getattr(self, f'stage{stage}')[0] = replacement
                slots.append((path, original, replacement))
        self.scdown_replacements += tuple(path for path, _, _ in slots)
        self._scdown_initialization_slots += tuple(slots)
