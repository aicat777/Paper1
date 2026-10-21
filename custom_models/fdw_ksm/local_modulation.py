"""FDConv's local KSM: channel-neighborhood descriptors modulate kernel entries.

Adapted from KernelSpatialModulation_Local at upstream commit
bf7a33d4f39ac7ea8172158f5628107fcaaba469 (MIT; license in frequency.py).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalKernelModulation(nn.Module):
    """Return one [Cout, Cin, k, k] modulation tensor per image.

    "Local" refers to neighboring channels in the pooled descriptor, not to
    positions in the image. Each generated kernel is shared over image space.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        neighborhood = round(math.log2(in_channels) / 2 + 0.5) // 2 * 2 + 1
        self.norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv1d(1, out_channels * kernel_size ** 2, neighborhood,
                              padding=(neighborhood - 1) // 2, bias=False)
        # Author initialization: approximately neutral 2 * sigmoid(0) = 1.
        nn.init.constant_(self.conv.weight, 1e-6)

    def forward(self, x):
        descriptor = F.adaptive_avg_pool2d(x, 1).flatten(2).transpose(1, 2)
        logits = self.conv(self.norm(descriptor))  # B, Cout*k*k, Cin
        # Preserve the author's Cin -> Cout -> kernel ordering before permuting.
        logits = logits.transpose(1, 2).reshape(
            x.shape[0], self.in_channels, self.out_channels,
            self.kernel_size, self.kernel_size).permute(0, 2, 1, 3, 4)
        return 2 * torch.sigmoid(logits)
