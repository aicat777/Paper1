"""FDW and input-conditioned kernel mixing from FDConv (CVPR 2025).

Source: https://github.com/Linwei-Chen/FDConv
Reference: bf7a33d4f39ac7ea8172158f5628107fcaaba469.
Retains the author's half-spectrum grouping, scaling, StarReLU router and
sigmoid kernel attention. Additional KSM modulation and FBM are disabled.

MIT License
Copyright (c) 2025 陈林蔚

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn
from torch.nn import functional as F


class StarReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * F.relu(x).square() + self.bias


class KernelRouter(nn.Module):
    def __init__(self, channels, kernel_num, reduction=0.0625):
        super().__init__()
        hidden = max(int(channels * reduction), 16)
        self.kernel_num = kernel_num
        self.fc = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.relu = StarReLU()
        self.kernel_fc = nn.Conv2d(hidden, kernel_num, 1)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.kernel_fc.weight, std=1e-6)
        nn.init.zeros_(self.kernel_fc.bias)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1)
        logits = self.kernel_fc(self.relu(self.bn(self.fc(pooled))))
        return (2.0 / self.kernel_num) * logits.flatten(1).sigmoid()


class FDWConv2d(nn.Module):
    """Convert one dense stride-one 3x3 convolution into the FDW operator.

    The supplied convolution provides only random initialization in our config.
    Spectrum values remain two real parameters per complex coefficient, so the
    standard AdamW, AMP and EMA paths can operate on ordinary floating tensors.
    """
    def __init__(self, conv, kernel_num=16):
        super().__init__()
        if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
                or conv.groups != 1 or conv.stride != (1, 1)
                or conv.dilation != (1, 1) or conv.padding_mode != 'zeros'):
            raise ValueError('FDW replacement requires a dense stride-one 3x3 convolution')
        if not isinstance(kernel_num, int) or kernel_num < 2 or conv.in_channels < 2:
            raise ValueError('FDW needs at least two kernel groups and input channels')
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size, self.stride, self.padding = conv.kernel_size, conv.stride, conv.padding
        self.kernel_num = kernel_num
        self.spectrum_shape = (self.out_channels * 3, self.in_channels * 3)
        rows, columns = self.spectrum_shape
        half_columns = columns // 2 + 1
        if rows * half_columns % kernel_num:
            raise ValueError('The half-spectrum coefficient count must divide into kernel groups')
        dtype = torch.float64 if conv.weight.dtype == torch.float64 else torch.float32
        device = conv.weight.device
        frequency_h = torch.fft.fftfreq(rows)
        frequency_w = torch.fft.rfftfreq(columns)
        grid = torch.stack(torch.meshgrid(frequency_h, frequency_w, indexing='ij'), dim=-1)
        order = grid.norm(dim=-1).flatten().argsort()
        indices = torch.stack((order // half_columns, order % half_columns))
        self.register_buffer('indices', indices.reshape(2, kernel_num, -1).to(device),
                             persistent=False)
        matrix = conv.weight.detach().to(dtype).permute(0, 2, 1, 3).reshape(rows, columns)
        # Same normalization and its compensating alpha as the official operator.
        spectrum = torch.fft.rfft2(matrix) / (self.in_channels // 2)
        self.dft_weight = nn.Parameter(torch.view_as_real(spectrum).unsqueeze(0).contiguous())
        self.alpha = (self.in_channels // 2) * kernel_num
        self.router = KernelRouter(self.in_channels, kernel_num).to(device=device, dtype=dtype)
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None

    def aggregate_weight(self, attention):
        """Synthesize one convolution kernel per sample, retaining FFT gradients."""
        dtype = torch.float64 if self.dft_weight.dtype == torch.float64 else torch.float32
        # CUDA half FFT cannot handle the non-power-of-two 144/288/576 matrices.
        with torch.autocast(device_type=attention.device.type, enabled=False):
            spectral = self.dft_weight[0].to(dtype)
            rows, columns = self.spectrum_shape
            h_indices, w_indices = self.indices
            grouped = spectral[h_indices, w_indices] * self.alpha
            mixed = grouped.unsqueeze(0) * attention.to(dtype)[:, :, None, None]
            packed = torch.zeros(attention.shape[0], rows, columns // 2 + 1, 2,
                                 device=attention.device, dtype=dtype)
            packed[:, h_indices, w_indices] = mixed
            matrix = torch.fft.irfft2(torch.view_as_complex(packed),
                                     s=self.spectrum_shape, dim=(-2, -1))
            return matrix.reshape(-1, self.out_channels, 3, self.in_channels, 3).permute(
                0, 1, 3, 2, 4).contiguous()

    def forward(self, x):
        batch, channels, height, width = x.shape
        weight = self.aggregate_weight(self.router(x))
        result = F.conv2d(x.reshape(1, batch * channels, height, width),
                          weight.to(x.dtype).reshape(batch * self.out_channels, self.in_channels, 3, 3),
                          stride=self.stride, padding=self.padding, groups=batch)
        result = result.reshape(batch, self.out_channels, *result.shape[-2:])
        if self.bias is not None:
            result = result + self.bias[None, :, None, None]
        return result
