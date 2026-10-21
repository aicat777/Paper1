"""Keep the first PAN transition and replace only its last downsampling."""
import torch
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .initialization import initialize_scdown
from .operator import SCDown


@MODELS.register_module()
class SCDownPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.in_channels) != 3 or len(self.downsample_layers) != 2:
            raise ValueError('This SCDown adaptation uses the three-level PAFPN')
        original = self.downsample_layers[1]
        with torch.random.fork_rng(devices=[]):
            replacement = SCDown.from_conv_module(original)
        self.downsample_layers[1] = replacement
        self.scdown_replacements = ('downsample_layers.1',)
        self._scdown_initialization_slots = (('downsample_layers.1', original, replacement),)

    def init_weights(self):
        initialize_scdown(self, super().init_weights)
