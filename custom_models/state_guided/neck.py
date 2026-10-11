"""Candidate A: compressed context plus state-guided local upsampling."""
import torch
from torch.nn import functional as F
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .context import CompressedStateContext
from .upsample import StateGuidedUpsample


@MODELS.register_module()
class StateGuidedPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, use_context=True, use_upsample=True,
                 state_channels=48, num_states=16, use_state_guidance=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.in_channels) != 3:
            raise ValueError('StateGuidedPAFPN expects the P5 three-level pyramid')
        self.context = (CompressedStateContext(self.in_channels[1],
                        state_channels, num_states) if use_context else None)
        self.local_upsample = (StateGuidedUpsample(
            self.in_channels[0], state_channels=state_channels
            if use_context and use_state_guidance else None,
            num_states=num_states) if use_upsample else None)

    def forward(self, inputs):
        if len(inputs) != 3:
            raise ValueError('Expected C3, C4, C5')
        features = list(inputs)
        states = None
        if self.context is not None:
            features[1], states = self.context(features[1])
        reduced = [layer(x) for layer, x in zip(self.reduce_layers, features)]
        inner = [reduced[-1]]
        for idx in (2, 1):
            high, low = inner[0], reduced[idx - 1]
            if idx == 1 and self.local_upsample is not None:
                up = self.local_upsample(high, low, states)
            else:
                up = F.interpolate(high, size=low.shape[-2:], mode='nearest')
            pair = (up, low) if self.upsample_feats_cat_first else (low, up)
            inner.insert(0, self.top_down_layers[2 - idx](torch.cat(pair, 1)))
        outputs = [inner[0]]
        for idx in range(2):
            down = self.downsample_layers[idx](outputs[-1])
            outputs.append(self.bottom_up_layers[idx](
                torch.cat((down, inner[idx + 1]), 1)))
        return tuple(layer(x) for layer, x in zip(self.out_layers, outputs))
