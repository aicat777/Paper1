"""Four whole neck blocks, on top of the unchanged three-site backbone."""
import torch
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .block import replace_blocks
from .initialization import original_initialization_slots


@MODELS.register_module()
class LSConvNeckPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        originals = {path: child for path, child in self.named_modules()
                     if isinstance(child, CSPNeXtBlock)}
        with torch.random.fork_rng(devices=[]):
            self.lsblock_replacements = replace_blocks(self)
        if len(self.lsblock_replacements) != 4:
            raise ValueError('This experiment requires exactly four internal neck blocks')
        self._initialization_slots = tuple(
            (path, originals[path], self.get_submodule(path)) for path in self.lsblock_replacements)

    def init_weights(self):
        if self.is_init:
            return
        with original_initialization_slots(self, self._initialization_slots):
            super().init_weights()
        self._initialization_slots = ()
