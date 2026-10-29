"""Add the established G8 treatment to the four-SCDown backbone."""
import torch
from mmyolo.registry import MODELS

from custom_models.scdown_g8.operator import grouped_conv_from
from custom_models.scdown_p3.backbone import SCDownP3CSPNeXt


@MODELS.register_module()
class SCDownP3G8CSPNeXt(SCDownP3CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        paths = ('stage2.1.blocks.0.conv1', 'stage3.1.blocks.0.conv1',
                 'stage4.2.blocks.0.conv1')
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
