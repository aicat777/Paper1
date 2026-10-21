"""Similarity-guided sampler from FreqFusion (Chen et al., TPAMI 2024).

Reference: Linwei-Chen/FreqFusion, commit
3fb0c70637a3c194fb74294d3ce4681958b26241, mmdetection/.../FreqFusion.py.
The four groups, dilation-2 cosine neighborhood, pixel ordering and half-pixel
grid are retained. The sampled value is ALREADY at high resolution.
"""
import torch
from torch import nn
from torch.nn import functional as F


def compute_similarity(x, kernel=3, dilation=2):
    b, c, h, w = x.shape
    patches = F.unfold(x, kernel, padding=kernel // 2 * dilation,
                       dilation=dilation).reshape(b, c, kernel * kernel, h, w)
    middle = kernel * kernel // 2
    similarity = F.cosine_similarity(patches[:, :, middle:middle + 1], patches, dim=1)
    return torch.cat((similarity[:, :middle], similarity[:, middle + 1:]), dim=1)


class LocalSimGuidedSampler(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % 8 or channels % groups:
            raise ValueError('Guidance channels must support the original GN and sampler groups')
        self.groups = groups
        self.offset = nn.Conv2d(channels + 8, 8 * groups, 3, padding=1)
        self.hr_offset = nn.Conv2d(channels + 8, 2 * groups, 3, padding=1)
        self.direct_scale = nn.Conv2d(channels + 8, 8 * groups, 3, padding=1)
        self.hr_direct_scale = nn.Conv2d(channels + 8, 2 * groups, 3, padding=1)
        self.norm_hr = nn.GroupNorm(channels // 8, channels)
        self.norm_lr = nn.GroupNorm(channels // 8, channels)
        h = torch.tensor([-.25, .25])
        self.register_buffer('init_pos', torch.stack(torch.meshgrid(h, h, indexing='ij'))
                             .transpose(1, 2).repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def sample(self, value, offset):
        b, _, h, w = offset.shape
        if value.shape[-2:] != (2 * h, 2 * w) or value.shape[1] % self.groups:
            raise ValueError('Expected an already upsampled value with divisible channels')
        offset = offset.reshape(b, 2, -1, h, w)
        x = torch.arange(w, device=value.device, dtype=value.dtype) + .5
        y = torch.arange(h, device=value.device, dtype=value.dtype) + .5
        coords = torch.stack(torch.meshgrid(x, y, indexing='ij')).transpose(1, 2)[None, :, None]
        normalizer = value.new_tensor([w, h]).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), 2)
        coords = coords.reshape(b, 2, self.groups, 2 * h, 2 * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        sampled = F.grid_sample(value.reshape(b * self.groups, -1, 2 * h, 2 * w),
                                coords, mode='bilinear', padding_mode='border',
                                align_corners=False)
        return sampled.reshape(b, -1, 2 * h, 2 * w)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x, lr_x = self.norm_hr(hr_x), self.norm_lr(lr_x)
        hr_x = torch.cat((hr_x, compute_similarity(hr_x)), dim=1)
        lr_x = torch.cat((lr_x, compute_similarity(lr_x)), dim=1)
        direction = self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), 2)
        scale = (self.direct_scale(lr_x)
                 + F.pixel_unshuffle(self.hr_direct_scale(hr_x), 2)).sigmoid()
        return self.sample(feat2sample, direction * scale + self.init_pos)
