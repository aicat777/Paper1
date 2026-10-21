"""Keep P3 construction and replace only backbone P4/P5 downsampling."""
import torch
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS

from .initialization import initialize_scdown
from .operator import SCDown


@MODELS.register_module()
class SCDownCSPNeXt(CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_stages != 4:
            raise ValueError('This experiment uses the four-stage P5 backbone')
        slots = []
        with torch.random.fork_rng(devices=[]):
            for stage in (3, 4):
                path = f'stage{stage}.0'
                original = self.get_submodule(path)
                replacement = SCDown.from_conv_module(original)
                getattr(self, f'stage{stage}')[0] = replacement
                slots.append((path, original, replacement))
        self.scdown_replacements = tuple(path for path, _, _ in slots)
        self._scdown_initialization_slots = tuple(slots)

    def init_weights(self):
        initialize_scdown(self, super().init_weights)
