"""Change only the spatial convolution at the experiment-2 P3 entry."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown_p3 import SCDownP3CSPNeXt
from .operator import P3DenseSpatial


@MODELS.register_module()
class SCDownP3DenseCSPNeXt(SCDownP3CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = 'stage2.0'
        reference = self.get_submodule(path)
        with torch.random.fork_rng(devices=[]):
            replacement = P3DenseSpatial(reference)
        self.stage2[0] = replacement
        # Replace the saved slot too: parent initialization must restore this
        # control, not the SCDown object which has just been removed.
        self._scdown_initialization_slots = tuple(
            (p, original, replacement if p == path else module)
            for p, original, module in self._scdown_initialization_slots)
        self.p3_control = 'dense_spatial'
