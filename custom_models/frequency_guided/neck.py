"""Candidate C: spatial frequency selection plus band-guided upsampling."""
import torch
from torch.nn import functional as F
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .frequency import SpatialFrequencySelector
from .upsample import FrequencyGuidedUpsample


@MODELS.register_module()
class FrequencyGuidedPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, use_frequency=True, use_upsample=True,
                 frequency_groups=4, use_frequency_guidance=True, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.in_channels) != 3:
            raise ValueError('FrequencyGuidedPAFPN expects C3, C4, C5')
        self.frequency = (SpatialFrequencySelector(self.in_channels[0],
                          frequency_groups) if use_frequency else None)
        self.use_frequency_guidance = (use_frequency and use_upsample
                                       and use_frequency_guidance)
        self.local_upsample = (FrequencyGuidedUpsample(self.in_channels[0],
                              use_frequency=self.use_frequency_guidance)
                              if use_upsample else None)

    def init_weights(self):
        super().init_weights()
        # The inherited neck initialization also traverses our Conv2d layers.
        if self.frequency is not None:
            self.frequency.reset_band_weights()

    def forward(self, inputs):
        if len(inputs) != 3:
            raise ValueError('Expected C3, C4, C5')
        features = list(inputs)
        energy = None
        if self.frequency is not None:
            features[0], energy = self.frequency(features[0],
                                                self.use_frequency_guidance)
        reduced = [layer(x) for layer, x in zip(self.reduce_layers, features)]
        inner = [reduced[-1]]
        for idx in (2, 1):
            high, low = inner[0], reduced[idx - 1]
            if idx == 1 and self.local_upsample is not None:
                up = self.local_upsample(high, low, energy)
            else:
                up = F.interpolate(high, size=low.shape[-2:], mode='nearest')
            pair = (up, low) if self.upsample_feats_cat_first else (low, up)
            inner.insert(0, self.top_down_layers[2 - idx](torch.cat(pair, 1)))
        outputs = [inner[0]]
        for idx in range(2):
            outputs.append(self.bottom_up_layers[idx](torch.cat((
                self.downsample_layers[idx](outputs[-1]), inner[idx + 1]), 1)))
        return tuple(layer(x) for layer, x in zip(self.out_layers, outputs))
