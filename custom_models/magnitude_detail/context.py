"""Magnitude-aware low-dimensional context, inspired by MALA (ICCV 2025).

Retains paired magnitude scaling and mean-value compensation. This CNN adapter
uses a local depthwise path instead of the original transformer's rotary setup.
Source: https://github.com/qhfan/MALA
"""
import torch
from torch import nn
from torch.nn import functional as F


def magnitude_linear_attention(q, k, v, eps=1e-6, magnitude_aware=True):
    """Positive q/k, [B, heads, tokens, channels], without an NxN matrix.

    The ordinary normalized variant is a parameter-matched control: only the
    normalization changes, with identical projections, heads and local path.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        scale = q.shape[-1] ** -0.5
        z = (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)) * scale
        kv = (k.transpose(-2, -1) @ v) * (scale / k.shape[-2])
        if not magnitude_aware:
            return (q @ kv) / z.clamp_min(eps)
        return (q @ kv) * (1.0 + 1.0 / z.clamp_min(eps)) - z * v.mean(
            dim=-2, keepdim=True)


class MagnitudeContext(nn.Module):
    def __init__(self, channels, hidden_channels=48, num_heads=4,
                 magnitude_aware=True):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError('hidden_channels must be divisible by num_heads')
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.magnitude_aware = magnitude_aware
        self.project = nn.Conv2d(channels, hidden_channels, 1)
        self.norm = nn.LayerNorm(hidden_channels)
        self.qkv = nn.Conv2d(hidden_channels, hidden_channels * 3, 1)
        self.local = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                               groups=hidden_channels)
        self.restore = nn.Conv2d(hidden_channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        b, _, h, w = x.shape
        hidden = self.project(x)
        hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        qkv = self.qkv(hidden).reshape(b, 3, self.num_heads,
                                      self.hidden_channels // self.num_heads,
                                      h * w).permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)
        context = magnitude_linear_attention(F.elu(q.float()) + 1,
                                              F.elu(k.float()) + 1, v,
                                              magnitude_aware=self.magnitude_aware)
        context = context.transpose(-2, -1).reshape(b, self.hidden_channels, h, w)
        local = self.local(v.transpose(-2, -1).reshape_as(hidden))
        return x + self.residual_scale * self.restore(context.to(hidden.dtype) + local)
