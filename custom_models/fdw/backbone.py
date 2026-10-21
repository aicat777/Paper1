"""Independent FDW replacement of C3/C4/C5 CSPNeXt-block dense convolutions."""
import torch
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS

from .operator import FDWConv2d


@MODELS.register_module()
class FDWCSPNeXt(CSPNeXt):
    def __init__(self, *args, fdw_stages=(2, 3, 4), fdw_kernel_num=16, **kwargs):
        self.fdw_stages = tuple(fdw_stages)
        self.fdw_kernel_num = fdw_kernel_num
        self.fdw_replacements = []
        super().__init__(*args, **kwargs)
        if any(not isinstance(stage, int) or not 1 <= stage <= self.num_stages
               for stage in self.fdw_stages):
            raise ValueError('FDW stage numbers must identify existing backbone stages')

    def build_stage_layer(self, stage_idx, setting):
        stage = super().build_stage_layer(stage_idx, setting)
        if stage_idx + 1 in self.fdw_stages:
            # Router initialization must not change subsequent baseline weights.
            with torch.random.fork_rng(devices=[]):
                for layer_idx, layer in enumerate(stage):
                    for path, block in layer.named_modules():
                        if isinstance(block, CSPNeXtBlock):
                            block.conv1.conv = FDWConv2d(block.conv1.conv, self.fdw_kernel_num)
                            self.fdw_replacements.append(
                                f'stage{stage_idx + 1}.{layer_idx}.{path}.conv1.conv')
        return stage
