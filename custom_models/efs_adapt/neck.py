"""Four-scale PAN with an uncompressed P3 lateral and a separate P2 branch."""
from mmdet.models.layers import CSPLayer
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from torch import nn


@MODELS.register_module()
class EFSPreserveP3PAFPN(CSPNeXtPAFPN):
    """Keep the P3 FPN lateral before projecting a separate branch toward P2.

    With tiny widths, the first PAN concatenation contains 48 downsampled P2
    channels and all 96 P3 lateral channels. All other PAN stages and the four
    output projections retain the official CSPNeXtPAFPN implementation.
    """

    def __init__(self, in_channels, *args, **kwargs):
        if len(in_channels) != 4:
            raise ValueError('EFSPreserveP3PAFPN requires C2/C3/C4/C5 inputs')
        super().__init__(in_channels, *args, **kwargs)

    def _csp(self, in_channels, out_channels):
        return CSPLayer(
            in_channels, out_channels, num_blocks=self.num_csp_blocks,
            add_identity=False, use_cspnext_block=True,
            expand_ratio=self.expand_ratio, conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def build_top_down_layer(self, idx):
        if idx == 2:
            # F3 itself remains at the backbone C3 width (96 for tiny).
            return self._csp(self.in_channels[1] * 2, self.in_channels[1])
        return super().build_top_down_layer(idx)

    def build_upsample_layer(self, idx):
        if idx == 1:
            # Only this branch reduces F3 to the C2 width before upsampling.
            return nn.Sequential(
                self.conv(self.in_channels[1], self.in_channels[0], 1,
                          norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
                nn.Upsample(**self.upsample_cfg))
        return super().build_upsample_layer(idx)

    def build_bottom_up_layer(self, idx):
        if idx == 0:
            return self._csp(self.in_channels[0] + self.in_channels[1],
                             self.in_channels[1])
        return super().build_bottom_up_layer(idx)
