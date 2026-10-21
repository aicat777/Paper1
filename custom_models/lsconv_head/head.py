"""Keep tower depth0; replace only depth1 by LSConv + residual channel FFN."""
import torch
from torch import nn
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS

from .block import LSFeatureBlock
from .initialization import original_initialization_slots


def share_convolutions(source, target):
    """Share Conv2d objects (including bias), never BN/GN or their statistics."""
    for path, layer in list(source.named_modules()):
        if isinstance(layer, nn.Conv2d):
            parent_path, _, name = path.rpartition('.')
            parent = target.get_submodule(parent_path) if parent_path else target
            parent._modules[name] = layer


@MODELS.register_module()
class LSConvSecondLayerHeadModule(RTMDetSepBNHeadModule):
    def _init_layers(self):
        super()._init_layers()
        if self.stacked_convs != 2 or not self.share_conv or len(self.featmap_strides) != 3:
            raise ValueError('This experiment requires two tower layers and three shared-conv scales')
        slots = []
        with torch.random.fork_rng(devices=[]):
            for branch in ('cls_convs', 'reg_convs'):
                towers = getattr(self, branch)
                for level, tower in enumerate(towers):
                    original = tower[1]
                    replacement = LSFeatureBlock(self.feat_channels)
                    tower[1] = replacement
                    slots.append((f'{branch}.{level}.1', original, replacement))
                for level in (1, 2):
                    share_convolutions(towers[0][1], towers[level][1])
        self.lsblock_replacements = tuple(path for path, _, _ in slots)
        self._initialization_slots = tuple(slots)

    def init_weights(self):
        if self.is_init:
            return
        # Parent init applies normal(.01) to every Conv and resets every norm.
        # Run it on the original second layers, preserving random draws for
        # untouched first layers/predictors and preserving the complete LSBlock.
        with original_initialization_slots(self, self._initialization_slots):
            super().init_weights()
        self._initialization_slots = ()
