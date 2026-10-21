"""Extend the legacy stage-wise conv1 replacement to all four backbone stages."""
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS
from torch import nn

from .layers import replace_conv1


@MODELS.register_module()
class CSPOnlyLSConvBackbone(CSPNeXt):
    def __init__(self, *args, lsconv_stages=(1, 2, 3, 4), **kwargs):
        self.lsconv_stages = tuple(lsconv_stages)
        self.lsconv_replacements = []
        super().__init__(*args, **kwargs)
        if any(not isinstance(stage, int) or not 1 <= stage <= self.num_stages
               for stage in self.lsconv_stages):
            raise ValueError('Invalid LSConv backbone stage')

    def build_stage_layer(self, stage_idx, setting):
        stage = super().build_stage_layer(stage_idx, setting)
        if stage_idx + 1 in self.lsconv_stages:
            paths = replace_conv1(nn.Sequential(*stage))
            self.lsconv_replacements.extend(f'stage{stage_idx + 1}.{path}' for path in paths)
        return stage
