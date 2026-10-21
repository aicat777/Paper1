"""FDW synthesis followed by the original local KSM multiplicative branch."""
from torch.nn import functional as F

from .frequency import FDWConv2d
from .local_modulation import LocalKernelModulation


class FDWKSMConv2d(FDWConv2d):
    def __init__(self, conv, kernel_num=16):
        super().__init__(conv, kernel_num)
        self.local = LocalKernelModulation(self.in_channels, self.out_channels, 3).to(
            device=self.dft_weight.device, dtype=self.dft_weight.dtype)

    def forward(self, x):
        batch, channels, height, width = x.shape
        weight = self.aggregate_weight(self.router(x)) * self.local(x)
        result = F.conv2d(
            x.reshape(1, batch * channels, height, width),
            weight.to(x.dtype).reshape(batch * self.out_channels, self.in_channels, 3, 3),
            stride=self.stride, padding=self.padding, groups=batch)
        result = result.reshape(batch, self.out_channels, *result.shape[-2:])
        if self.bias is not None:
            result = result + self.bias[None, :, None, None]
        return result
