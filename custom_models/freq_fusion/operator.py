"""Complete detection setting of FreqFusion, without an external checkpoint.

Adapted from Linwei-Chen/FreqFusion at 3fb0c70637a3c194fb74294d3ce4681958b26241
(mmdetection/mmdet/models/necks/FreqFusion.py, SHA256 886e2d0068d2c1ce8f054ad12b4337e990a43a64220b29c978c4eec8057c12d7).
All three generators AND enhanced initial fusion are active. CUDA uses MMCV
CARAFE, including its zero padding; the CPU reference implements the same math.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv.ops import carafe as cuda_carafe

from .resample import LocalSimGuidedSampler


def reassemble(value, mask, kernel, scale=1):
    b, c, h, w = value.shape
    if mask.shape != (b, kernel * kernel, h * scale, w * scale):
        raise ValueError('CARAFE mask must index each output location and kernel tap')
    if value.dtype != mask.dtype:
        raise TypeError('CARAFE values and kernels must use the same precision')
    if value.is_cuda:
        return cuda_carafe(value.contiguous(), mask.contiguous(), kernel, 1, scale)
    # Reference/CPU execution only. Do not use reflect padding: it changes the
    # boundary behavior of the paper's MMCV operator.
    patches = F.unfold(value, kernel, padding=kernel // 2)
    patches = patches.reshape(b, c * kernel * kernel, h, w)
    patches = F.interpolate(patches, scale_factor=scale, mode='nearest')
    patches = patches.reshape(b, c, kernel * kernel, h * scale, w * scale)
    return (patches * mask[:, None]).sum(dim=2)


class FreqFusion(nn.Module):
    def __init__(self, hr_channels, lr_channels, compressed_channels, groups=4):
        super().__init__()
        if lr_channels % groups:
            raise ValueError('Value channels must be divisible by offset groups')
        self.hr_channels, self.lr_channels = hr_channels, lr_channels
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, compressed_channels, 1)
        self.content_encoder = nn.Conv2d(compressed_channels, 25, 3, padding=1)
        self.content_encoder2 = nn.Conv2d(compressed_channels, 9, 3, padding=1)
        self.dysampler = LocalSimGuidedSampler(compressed_channels, groups)
        for name, kernel in (('lowpass', 5), ('highpass', 3)):
            window = np.outer(np.hamming(kernel), np.hamming(kernel))
            self.register_buffer('hamming_' + name, torch.tensor(window, dtype=torch.float32)[None, None])
        self.init_weights()

    def init_weights(self):
        # Preserve the upstream FreqFusion.init_weights outcome: it applies
        # Xavier to ALL convolutions, including the sampler, and then gives only
        # the two filter generators std=.001. Do not silently keep the sampler
        # constructor's small/zero initialization, which upstream overwrites.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in (self.content_encoder, self.content_encoder2):
            nn.init.normal_(layer.weight, std=.001)

    @staticmethod
    def kernel_normalizer(logits, kernel, hamming):
        b, _, h, w = logits.shape
        mask = logits.reshape(b, 1, kernel * kernel, h, w).softmax(dim=2)
        mask = mask.reshape(b, 1, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).reshape(b, -1, kernel, kernel)
        mask = mask * hamming
        mask = mask / mask.sum(dim=(-1, -2), keepdim=True)
        return mask.reshape(b, 1, h, w, -1).permute(0, 1, 4, 2, 3).reshape(b, -1, h, w).contiguous()

    def _forward(self, hr, lr):
        high = self.hr_channel_compressor(hr)
        low = self.lr_channel_compressor(lr)
        hp_hr_logits = self.content_encoder2(high)
        hp_initial = self.kernel_normalizer(hp_hr_logits, 3, self.hamming_highpass)
        high = high + high - reassemble(high, hp_initial, 3)
        lp_hr_logits = self.content_encoder(high)
        lp_initial = self.kernel_normalizer(lp_hr_logits, 5, self.hamming_lowpass)
        lp_logits = lp_hr_logits + reassemble(self.content_encoder(low), lp_initial, 5, 2)
        lp_mask = self.kernel_normalizer(lp_logits, 5, self.hamming_lowpass)
        hp_logits = hp_hr_logits + reassemble(self.content_encoder2(low), lp_mask, 5, 2)
        # Upstream normalizes lp_logits again here; this is the same tensor.
        lr = reassemble(lr, lp_mask, 5, 2)
        hp_mask = self.kernel_normalizer(hp_logits, 3, self.hamming_highpass)
        hr = (hr - reassemble(hr, hp_mask, 3)) + hr
        lr = self.dysampler(high, low, lr)
        return lp_mask, hr, lr

    def forward(self, hr_feat, lr_feat):
        if (hr_feat.ndim != 4 or lr_feat.ndim != 4
                or hr_feat.shape[:2] != (lr_feat.shape[0], self.hr_channels)
                or lr_feat.shape[1] != self.lr_channels
                or hr_feat.shape[-2:] != tuple(2 * s for s in lr_feat.shape[-2:])):
            raise ValueError('FreqFusion requires the declared widths and exactly 2x spatial scales')
        # MMCV 2.0 CARAFE is not autocast-aware. Keep kernels, features and grids
        # in FP32 throughout this module, rather than mixing half values with
        # float masks. Cast only the returned value streams back to their input
        # types. No image/feature resize is used to conceal a shape mismatch.
        hr_dtype, lr_dtype = hr_feat.dtype, lr_feat.dtype
        with torch.autocast(device_type=hr_feat.device.type, enabled=False):
            hr = hr_feat.float() if hr_dtype in (torch.float16, torch.bfloat16) else hr_feat
            lr = lr_feat.float() if lr_dtype in (torch.float16, torch.bfloat16) else lr_feat
            mask, hr, lr = self._forward(hr, lr)
        return mask, hr.to(hr_dtype), lr.to(lr_dtype)
