"""P3 control A: retain PW/BN/SiLU, use a dense spatial convolution."""
from copy import deepcopy

from mmcv.cnn import ConvModule
from torch import nn

from custom_models.scdown import SCDown


class P3DenseSpatial(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if not isinstance(reference, SCDown):
            raise TypeError('This control replaces the P3 SCDown, not a raw dense convolution')
        pw, dw = reference.cv1, reference.cv2
        if (pw.conv.kernel_size != (1, 1) or pw.conv.stride != (1, 1)
                or pw.conv.groups != 1 or not pw.with_activation
                or not isinstance(pw.activate, nn.SiLU)
                or dw.conv.kernel_size != (3, 3) or dw.conv.stride != (2, 2)
                or dw.conv.padding != (1, 1) or dw.conv.dilation != (1, 1)
                or dw.conv.groups != dw.conv.in_channels
                or dw.conv.in_channels != dw.conv.out_channels
                or pw.conv.out_channels != dw.conv.in_channels
                or dw.with_activation or not pw.with_norm or not dw.with_norm
                or not isinstance(pw.norm, nn.BatchNorm2d) or not isinstance(dw.norm, nn.BatchNorm2d)
                or dw.conv.bias is not None or dw.conv.padding_mode != 'zeros'
                or pw.order != ('conv', 'norm', 'act') or dw.order != ('conv', 'norm', 'act')):
            raise ValueError('Expected PW/BN/SiLU -> DW3x3 stride2/BN')
        # Keep the reference pointwise weights and both normalization states.
        # Only the spatial kernel gains off-diagonal channel connections.
        self.cv1 = deepcopy(pw)
        width = dw.conv.in_channels
        self.cv2 = ConvModule(width, width, 3, stride=2, padding=1, groups=1,
                              norm_cfg=deepcopy(dw.norm_cfg), act_cfg=None)
        self.cv2.norm.load_state_dict(dw.norm.state_dict(), strict=True)

    def forward(self, x):
        return self.cv2(self.cv1(x))
