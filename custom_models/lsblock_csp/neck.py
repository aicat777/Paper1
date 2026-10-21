"""Replace only internal CSP blocks; preserve FPN/PAN channels and outputs."""
import torch
from torch import nn
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .block import replace_blocks


@MODELS.register_module()
class LSBlockCSPPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.lsblock_replacements = replace_blocks(self)

    def init_weights(self):
        if self.is_init:
            return
        removed = []
        # Preserve author's LKP biases and the FFN terminal BN zero scale.
        # Every surrounding projection/output keeps the standard neck init.
        for path in self.lsblock_replacements:
            parent_path, name = path.rsplit('.', 1)
            parent = self.get_submodule(parent_path)
            removed.append((parent, name, parent._modules[name]))
            parent._modules[name] = nn.Identity()
        try:
            super().init_weights()
        finally:
            for parent, name, module in removed:
                parent._modules[name] = module
