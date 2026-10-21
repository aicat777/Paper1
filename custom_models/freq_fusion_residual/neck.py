"""Fuse semantics through CSP; add the separately projected detail afterwards.

At each top-down site: Y = CSP_and_original_projection(cat(U, H)) + P(D).
U and D come from the full FreqFusion generators, with D = H - smooth(H).
The HR identity enters CSP once; its detail enters only the residual route.
"""
import torch
from torch import nn
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .operator import SplitFreqFusion


@MODELS.register_module()
class ResidualFreqFusionPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, compress_ratio=8, resample_groups=4, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep all baseline random draws and the original FreqFusion core's
        # initialization aligned with the existing standalone experiment.
        with torch.random.fork_rng(devices=[]):
            self.freq_fusions = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1, 0, -1):
                width = self.in_channels[idx - 1]
                if 2 * width % compress_ratio:
                    raise ValueError('Compression must divide the combined guidance width')
                self.freq_fusions.append(SplitFreqFusion(
                    width, width, 2 * width // compress_ratio, groups=resample_groups))
            self.detail_projections = nn.ModuleList()
            for idx in range(len(self.in_channels) - 1, 0, -1):
                width = self.in_channels[idx - 1]
                # CSPNeXtPAFPN's first top-down block includes a following
                # 192->96 projection; inject AFTER that complete block.
                fused_width = self.in_channels[0] if idx == 1 else self.in_channels[idx - 2]
                self.detail_projections.append(nn.Conv2d(width, fused_width, 1, bias=False))

    def init_weights(self):
        if self.is_init:
            return
        fusion = self._modules.pop('freq_fusions')
        projections = self._modules.pop('detail_projections')
        try:
            super().init_weights()
        finally:
            # Preserve registered modules and their order even if the parent
            # initialization fails. No new parameters are silently discarded.
            self.add_module('freq_fusions', fusion)
            self.add_module('detail_projections', projections)
        with torch.random.fork_rng(devices=[]):
            for layer in fusion:
                layer.init_weights()
            for layer in projections:
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError('Unexpected number of backbone scales')
        reduced = [layer(value) for layer, value in zip(self.reduce_layers, inputs)]
        inner = [reduced[-1]]
        for module_idx, idx in enumerate(range(len(inputs) - 1, 0, -1)):
            high = reduced[idx - 1]
            _, detail, upsampled = self.freq_fusions[module_idx](high, inner[0])
            pair = (upsampled, high) if self.upsample_feats_cat_first else (high, upsampled)
            main = self.top_down_layers[module_idx](torch.cat(pair, dim=1))
            residual = self.detail_projections[module_idx](detail)
            if residual.shape != main.shape:
                raise ValueError('Projected detail must match the entire top-down block output')
            inner.insert(0, main + residual)
        outs = [inner[0]]
        for idx, layer in enumerate(self.bottom_up_layers):
            outs.append(layer(torch.cat((self.downsample_layers[idx](outs[-1]), inner[idx + 1]), 1)))
        return tuple(layer(value) for layer, value in zip(self.out_layers, outs))
