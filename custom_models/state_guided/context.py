"""Low-dimensional hidden-state mixing inspired by EfficientViM (CVPR 2025).

This adaptation uses normalized state assignment/readout and exports the states
to the neck's upsampler. It is not the original HSM-SSD implementation.
Source: https://github.com/mlvlab/EfficientViM
"""
import torch
from torch import nn


class CompressedStateContext(nn.Module):
    def __init__(self, channels, hidden_channels=48, num_states=16):
        super().__init__()
        self.project = nn.Conv2d(channels, hidden_channels, 1)
        self.norm = nn.LayerNorm(hidden_channels)
        # Softmax runs over positions, so a per-state bias would cancel exactly.
        self.assignment = nn.Linear(hidden_channels, num_states, bias=False)
        self.readout = nn.Linear(hidden_channels, num_states)
        self.state_gate = nn.Linear(hidden_channels, hidden_channels * 2)
        self.state_out = nn.Linear(hidden_channels, hidden_channels)
        self.local = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                               groups=hidden_channels)
        self.restore = nn.Conv2d(hidden_channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        b, _, h, w = x.shape
        projected = self.project(x)
        tokens = self.norm(projected.flatten(2).transpose(1, 2))
        # Reductions stay FP32 under AMP; expensive channel mixing acts on S
        # states rather than H*W tokens. No spatial sequence scan is needed.
        with torch.autocast(device_type=x.device.type, enabled=False):
            assignment = self.assignment(tokens.float()).softmax(dim=1)
            states = assignment.transpose(1, 2) @ tokens.float()
            content, gate = self.state_gate(states).chunk(2, dim=-1)
            states = self.state_out(content * torch.nn.functional.silu(gate))
            readout = self.readout(tokens.float()).softmax(dim=-1)
            context = readout @ states
        context = context.transpose(1, 2).reshape(b, -1, h, w)
        context = context.to(projected.dtype) + self.local(projected)
        return x + self.residual_scale * self.restore(context), states
