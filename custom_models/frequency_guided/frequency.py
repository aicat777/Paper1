"""Spatial band modulation inspired by FDConv's FBM (CVPR 2025).

This is a feature-domain adapter, without FDW or KSM weight generation. Bands
are disjoint rectangular Fourier regions; their sum reconstructs the input.
Source: https://github.com/Linwei-Chen/FDConv
"""
import torch
from torch import nn


def split_frequency_bands(x):
    """Return four real bands in FP32; supports odd and rectangular shapes."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        h, w = x.shape[-2:]
        spectrum = torch.fft.rfft2(x, norm='ortho')
        fy = torch.fft.fftfreq(h, device=x.device).abs()[:, None]
        fx = torch.fft.rfftfreq(w, device=x.device).abs()[None, :]
        radius = torch.maximum(fy, fx)
        previous = torch.zeros_like(x)
        bands = []
        for cutoff in (1 / 16, 1 / 8, 1 / 4):
            low = torch.fft.irfft2(spectrum * (radius < cutoff), s=(h, w),
                                   norm='ortho')
            bands.append(low - previous)
            previous = low
        bands.append(x - previous)
        return bands


class SpatialFrequencySelector(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % groups:
            raise ValueError('channels must be divisible by frequency groups')
        self.groups = groups
        self.weight_features = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.SiLU())
        self.band_logits = nn.Conv2d(channels, 4 * groups, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.reset_band_weights()

    def reset_band_weights(self):
        nn.init.zeros_(self.band_logits.weight)
        nn.init.zeros_(self.band_logits.bias)

    def forward(self, x, return_energy=True):
        b, c, h, w = x.shape
        weights = 2 * self.band_logits(self.weight_features(x)).float().sigmoid()
        weights = weights.reshape(b, 4, self.groups, 1, h, w)
        bands = split_frequency_bands(x)
        result = torch.zeros_like(bands[0])
        energy = []
        for idx, band in enumerate(bands):
            weighted = band.reshape(b, self.groups, c // self.groups, h, w)
            weighted = (weighted * weights[:, idx]).reshape(b, c, h, w)
            result = result + weighted
            if return_energy:
                energy.append(weighted.abs().mean(dim=1, keepdim=True))
        output = x + self.residual_scale * (result.to(x.dtype) - x)
        if not return_energy:
            return output, None
        # Relative mean absolute activation, not squared spectral energy.
        energy = torch.cat(energy, dim=1)
        energy = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return output, energy.to(x.dtype)
