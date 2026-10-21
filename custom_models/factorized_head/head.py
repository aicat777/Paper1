"""Independent task towers with PW-DW-PW blocks and scale-specific BN.

This is a detector-head design using spatial/channel factorization, not an
implementation of the multi-branch InceptionNeXt block. No cls/reg sharing is
introduced. Each former 3x3 Conv/BN/SiLU becomes three Conv/BN/SiLU stages.
"""
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS


class FactorizedTowerBlock(nn.Module):
    def __init__(self, channels_in, channels_out, expansion, norm_cfg, act_cfg):
        super().__init__()
        hidden = int(channels_in * expansion)
        if hidden < channels_in or hidden != channels_in * expansion:
            raise ValueError('Expansion must produce an integral non-compressing width')
        self.expand = ConvModule(channels_in, hidden, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.spatial = ConvModule(hidden, hidden, 3, padding=1, groups=hidden,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.project = ConvModule(hidden, channels_out, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, value):
        return self.project(self.spatial(self.expand(value)))


@MODELS.register_module()
class FactorizedSepBNHeadModule(RTMDetSepBNHeadModule):
    def __init__(self, *args, expansion=2, **kwargs):
        self.expansion = expansion
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        with torch.random.fork_rng(devices=[]):
            for name in ('cls_convs', 'reg_convs'):
                towers = getattr(self, name)
                for depth in range(self.stacked_convs):
                    reference = None
                    for level, tower in enumerate(towers):
                        old = tower[depth].conv
                        block = FactorizedTowerBlock(old.in_channels, old.out_channels,
                                                     self.expansion, self.norm_cfg, self.act_cfg)
                        if self.share_conv and reference is not None:
                            for part in ('expand', 'spatial', 'project'):
                                getattr(block, part).conv = getattr(reference, part).conv
                        if reference is None:
                            reference = block
                        tower[depth] = block
        # The original forward and initialization cover the new blocks. All
        # Conv weights are initialized as RTMDet weights, and all BN scales to 1.
