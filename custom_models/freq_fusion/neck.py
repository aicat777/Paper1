"""Replace the two top-down fusion operations; preserve all CSP/PAN widths."""
import torch
from torch import nn
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .operator import FreqFusion


@MODELS.register_module()
class FreqFusionPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, compress_ratio=8, resample_groups=4, **kwargs):
        super().__init__(*args, **kwargs)
        # Guidance compression follows the author's detection config; it does
        # not project either value stream or change the existing CSP inputs.
        with torch.random.fork_rng(devices=[]):
            self.freq_fusions = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1, 0, -1):
                width = self.in_channels[idx - 1]
                if 2 * width % compress_ratio:
                    raise ValueError('Compression must divide the combined guidance width')
                self.freq_fusions.append(FreqFusion(width, width, 2 * width // compress_ratio,
                                                   groups=resample_groups))

    def init_weights(self):
        if self.is_init:
            return
        # Exclude new modules from the baseline neck's recursive Kaiming init.
        # This preserves both the baseline random draws and the author's custom
        # initialization. Reattach in finally so an initialization error cannot
        # silently delete registered parameters.
        fusion = self._modules.pop('freq_fusions')
        try:
            super().init_weights()
        finally:
            self.add_module('freq_fusions', fusion)
        with torch.random.fork_rng(devices=[]):
            for layer in fusion:
                layer.init_weights()

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError('Unexpected number of backbone scales')
        reduced = [layer(value) for layer, value in zip(self.reduce_layers, inputs)]
        inner = [reduced[-1]]
        for module_idx, idx in enumerate(range(len(inputs) - 1, 0, -1)):
            _, low_level, upsampled = self.freq_fusions[module_idx](reduced[idx - 1], inner[0])
            pair = (upsampled, low_level) if self.upsample_feats_cat_first else (low_level, upsampled)
            inner.insert(0, self.top_down_layers[module_idx](torch.cat(pair, dim=1)))
        outs = [inner[0]]
        for idx, layer in enumerate(self.bottom_up_layers):
            outs.append(layer(torch.cat((self.downsample_layers[idx](outs[-1]), inner[idx + 1]), 1)))
        return tuple(layer(value) for layer, value in zip(self.out_layers, outs))
