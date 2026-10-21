"""LSConv at the original output width, followed by a pointwise projection."""
from copy import deepcopy

import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.model import initialize
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .initialization import original_initialization_slots
from .operator import LSConv


class LSConvOutput(nn.Module):
    """Keep the value width through spatial aggregation; project afterwards.

    LSConv includes its own BN and residual. The projection retains the original
    output's norm/activation configuration. No extra activation, residual or
    FFN surrounds the spatial mixer.
    """
    def __init__(self, original):
        super().__init__()
        conv = original.conv
        if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
                or conv.stride != (1, 1) or conv.padding != (1, 1)
                or conv.dilation != (1, 1) or conv.groups != 1):
            raise ValueError('Output replacement requires a dense, stride-one padded 3x3')
        self.mixer = LSConv(conv.in_channels)
        self.projection = ConvModule(
            conv.in_channels, conv.out_channels, 1,
            conv_cfg=deepcopy(original.conv_cfg),
            norm_cfg=deepcopy(original.norm_cfg) if original.with_norm else None,
            act_cfg=deepcopy(original.act_cfg) if original.with_activation else None)

    def forward(self, x):
        return self.projection(self.mixer(x))


@MODELS.register_module()
class LSConvOutputPAFPN(CSPNeXtPAFPN):
    """Preserve all FPN/PAN internals and replace only three final output maps."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.out_layers) != 3:
            raise ValueError('This experiment requires three neck outputs')
        slots = []
        # Parent construction consumes the same random draws as the foundation.
        with torch.random.fork_rng(devices=[]):
            for level, original in enumerate(self.out_layers):
                replacement = LSConvOutput(original)
                self.out_layers[level] = replacement
                slots.append((f'out_layers.{level}', original, replacement))
        self.lsconv_replacements = tuple(path for path, _, _ in slots)
        self._initialization_slots = tuple(slots)

    def init_weights(self):
        if self.is_init:
            return
        # Initialize the original layout so untouched layers and the later head
        # retain the foundation's exact initialization and random-draw order.
        # New LSConv cores retain the author's constructor state.
        with original_initialization_slots(self, self._initialization_slots):
            super().init_weights()
        try:
            # New 1x1 maps use the official neck initializer, rather than the
            # different ConvModule constructor default. Keep their draws local.
            with torch.random.fork_rng(devices=[]):
                if self.init_cfg is not None:
                    for output in self.out_layers:
                        initialize(output.projection, deepcopy(self.init_cfg))
        except Exception:
            self._is_init = False
            raise
        self._initialization_slots = ()
