"""Replace complete internal CSPNeXt blocks, preserving backbone stage layout."""
import torch
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS

from .block import replace_blocks


@MODELS.register_module()
class LSBlockCSPBackbone(CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.lsblock_replacements = replace_blocks(self)
