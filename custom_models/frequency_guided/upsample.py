"""Frequency-conditioned local upsampling inspired by UPLiFT (CVPR 2026).

This candidate owns its operator independently of state_guided. Per-band local
mean absolute activation conditions the kernel; it does not label high frequency as
foreground, and uses no offset regression or external teacher.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FrequencyGuidedUpsample(nn.Module):
    def __init__(self, channels, guide_channels=32, use_frequency=True):
        super().__init__()
        self.guide = nn.Sequential(nn.Conv2d(channels * 2, guide_channels, 1),
                                   nn.SiLU())
        self.energy_projection = (nn.Conv2d(4, guide_channels, 1)
                                  if use_frequency else None)
        self.kernel_logits = nn.Conv2d(guide_channels, 9, 1)
        self.mix_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, value, lateral, energy=None):
        size = lateral.shape[-2:]
        nearest = F.interpolate(value, size=size, mode='nearest')
        guide = self.guide(torch.cat((lateral, nearest), dim=1))
        if self.energy_projection is not None:
            if energy is None:
                raise ValueError('Frequency-conditioned upsampling needs band energy')
            guide = guide + self.energy_projection(energy)
        weights = self.kernel_logits(guide).float().softmax(dim=1).to(value.dtype)
        h, w = value.shape[-2:]
        padded = F.pad(value, (1, 1, 1, 1), mode='replicate')
        result = torch.zeros_like(nearest)
        for i in range(9):
            dy, dx = divmod(i, 3)
            neighbor = padded[:, :, dy:dy + h, dx:dx + w]
            neighbor = F.interpolate(neighbor, size=size, mode='nearest')
            result = result + weights[:, i:i + 1] * neighbor
        return nearest + self.mix_logit.sigmoid() * (result - nearest)
