"""Candidate B: shallow detail injection plus magnitude-aware context."""
import torch
from torch.nn import functional as F
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .context import MagnitudeContext
from .detail import GatedShallowDetail


@MODELS.register_module()
class MagnitudeDetailPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, use_context=True, use_detail=True,
                 shallow_channels=128, context_channels=48, num_heads=4,
                 detail_channels=24, magnitude_aware=True, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.in_channels) != 3:
            raise ValueError('MagnitudeDetailPAFPN expects a three-level neck')
        self.context = (MagnitudeContext(self.in_channels[1], context_channels,
                        num_heads, magnitude_aware) if use_context else None)
        self.detail = (GatedShallowDetail(int(shallow_channels * self.widen_factor),
                       self.in_channels[0], detail_channels) if use_detail else None)

    def forward(self, inputs):
        if len(inputs) != 4:
            raise ValueError('Expected C2, C3, C4, C5; use out_indices=(1,2,3,4)')
        shallow, *features = inputs
        if self.detail is not None:
            features[0] = self.detail(shallow, features[0])
        if self.context is not None:
            features[1] = self.context(features[1])
        reduced = [layer(x) for layer, x in zip(self.reduce_layers, features)]
        inner = [reduced[-1]]
        for idx in (2, 1):
            high, low = inner[0], reduced[idx - 1]
            up = F.interpolate(high, size=low.shape[-2:], mode='nearest')
            pair = (up, low) if self.upsample_feats_cat_first else (low, up)
            inner.insert(0, self.top_down_layers[2 - idx](torch.cat(pair, 1)))
        outputs = [inner[0]]
        for idx in range(2):
            outputs.append(self.bottom_up_layers[idx](torch.cat((
                self.downsample_layers[idx](outputs[-1]), inner[idx + 1]), 1)))
        return tuple(layer(x) for layer, x in zip(self.out_layers, outputs))
