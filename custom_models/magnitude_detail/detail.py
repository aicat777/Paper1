"""Gated C2-to-C3 detail transport, motivated by DM-EFS (ICCV 2025).

Uses its expanded shallow-feature idea, not its size codebook, controller or
hard inference routing. Space-to-depth moves four subpixels into channels;
learned compression and a C3-conditioned gate select the injected information.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GatedShallowDetail(nn.Module):
    def __init__(self, shallow_channels, channels, detail_channels=24):
        super().__init__()
        self.project = nn.Conv2d(shallow_channels, detail_channels, 1)
        packed = 4 * detail_channels
        self.mix = nn.Sequential(
            nn.Conv2d(packed, packed, 3, padding=1, groups=packed, bias=False),
            nn.BatchNorm2d(packed), nn.SiLU(), nn.Conv2d(packed, channels, 1))
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, shallow, lateral):
        detail = self.project(shallow)
        # Stride-2 convolutions use ceil-sized outputs for odd input shapes.
        h, w = detail.shape[-2:]
        detail = F.pad(detail, (0, w % 2, 0, h % 2), mode='replicate')
        detail = self.mix(F.pixel_unshuffle(detail, 2))
        if detail.shape[-2:] != lateral.shape[-2:]:
            detail = F.interpolate(detail, size=lateral.shape[-2:], mode='nearest')
        gate = self.gate(torch.cat((lateral, detail), dim=1)).sigmoid()
        return lateral + self.residual_scale * gate * detail
