"""Replace C3/C4/C5 CSPNeXt-block 3x3 convolutions with complete LSConv."""
import copy

from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS
from torch import nn

from .operator import LSConv


@MODELS.register_module()
class LSConvCSPNeXt(CSPNeXt):
    def __init__(self, *args, lsconv_stages=(2, 3, 4), **kwargs):
        self.lsconv_stages = tuple(lsconv_stages)
        self.lsconv_replacements = []
        super().__init__(*args, **kwargs)
        if any(not isinstance(stage, int) or not 1 <= stage <= self.num_stages
               for stage in self.lsconv_stages):
            raise ValueError('LSConv stage numbers must identify existing backbone stages')

    def build_stage_layer(self, stage_idx, setting):
        stage = super().build_stage_layer(stage_idx, setting)
        if stage_idx + 1 not in self.lsconv_stages:
            return stage
        for layer_idx, layer in enumerate(stage):
            for path, block in layer.named_modules():
                if not isinstance(block, CSPNeXtBlock):
                    continue
                previous = block.conv1
                conv = getattr(previous, 'conv', None)
                if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
                        or conv.in_channels != conv.out_channels or conv.groups != 1):
                    raise ValueError('LSConv replacement requires an equal-width dense 3x3')
                activation = copy.deepcopy(previous.activate) if previous.with_activation else nn.Identity()
                block.conv1 = nn.Sequential(LSConv(conv.in_channels), activation)
                self.lsconv_replacements.append(
                    f'stage{stage_idx + 1}.{layer_idx}.{path}.conv1')
        return stage
