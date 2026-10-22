"""Keep the four-SCDown control weights; remove the P3 intermediate SiLU."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown_p3 import SCDownP3CSPNeXt
from .operator import P3NoActivation


@MODELS.register_module()
class SCDownP3NoActCSPNeXt(SCDownP3CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = 'stage2.0'
        reference = self.get_submodule(path)
        with torch.random.fork_rng(devices=[]):
            replacement = P3NoActivation(reference)
        self.stage2[0] = replacement
        self._scdown_initialization_slots = tuple(
            (p, original, replacement if p == path else module)
            for p, original, module in self._scdown_initialization_slots)
        self.p3_control = 'no_intermediate_activation'
