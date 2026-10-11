"""Local reassembly inspired by UPLiFT, with cross-scale state conditioning.

Source: https://github.com/mwalmer-umd/UPLiFT (Local Attender, CVPR 2026).
Fixed 3x3 coarse-grid neighbors are reweighted using lateral features and the
states produced by this method's context block. No pretrained image encoder.
"""
import torch
from torch import nn
from torch.nn import functional as F


class StateGuidedUpsample(nn.Module):
    def __init__(self, channels, guide_channels=48, state_channels=None,
                 num_states=16):
        super().__init__()
        self.guide = nn.Sequential(nn.Conv2d(channels * 2, guide_channels, 1),
                                   nn.SiLU())
        self.state_query = (nn.Conv2d(guide_channels, num_states, 1)
                            if state_channels is not None else None)
        self.state_value = (nn.Linear(state_channels, guide_channels)
                            if state_channels is not None else None)
        self.kernel_logits = nn.Conv2d(guide_channels, 9, 1)
        self.mix_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, value, lateral, states=None):
        size = lateral.shape[-2:]
        nearest = F.interpolate(value, size=size, mode='nearest')
        guide = self.guide(torch.cat((lateral, nearest), dim=1))
        if self.state_query is not None:
            if states is None:
                raise ValueError('State-conditioned upsampling requires states')
            with torch.autocast(device_type=value.device.type, enabled=False):
                query = self.state_query(guide.float()).flatten(2)
                query = query.transpose(1, 2).softmax(dim=-1)
                context = query @ self.state_value(states.float())
            context = context.transpose(1, 2).reshape_as(guide)
            guide = guide + context.to(guide.dtype)
        weights = self.kernel_logits(guide).float().softmax(dim=1).to(value.dtype)
        # Accumulate neighbors separately instead of materializing a
        # [batch, channels, 9, output_height, output_width] product tensor.
        h, w = value.shape[-2:]
        padded = F.pad(value, (1, 1, 1, 1), mode='replicate')
        result = torch.zeros_like(nearest)
        for i in range(9):
            dy, dx = divmod(i, 3)
            neighbor = padded[:, :, dy:dy + h, dx:dx + w]
            neighbor = F.interpolate(neighbor, size=size, mode='nearest')
            result = result + weights[:, i:i + 1] * neighbor
        return nearest + self.mix_logit.sigmoid() * (result - nearest)
