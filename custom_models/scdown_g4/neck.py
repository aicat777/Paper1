"""SCDown3 control plus groups=4 in the four internal PAFPN CSP conv1s."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown.neck import SCDownPAFPN
from .operator import grouped_conv_from


@MODELS.register_module()
class SCDownG4PAFPN(SCDownPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        paths = ('top_down_layers.0.0.blocks.0.conv1',
                 'top_down_layers.1.blocks.0.conv1',
                 'bottom_up_layers.0.blocks.0.conv1',
                 'bottom_up_layers.1.blocks.0.conv1')
        slots = []
        with torch.random.fork_rng(devices=[]):
            for path in paths:
                original = self.get_submodule(path)
                replacement = grouped_conv_from(original)
                parent, _, name = path.rpartition('.')
                self.get_submodule(parent)._modules[name] = replacement
                slots.append((path, original, replacement))
        self.grouped_replacements = paths
        self._scdown_initialization_slots += tuple(slots)
