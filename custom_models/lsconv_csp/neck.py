"""Change only neck CSP conv1; retain all projections and output layers."""
import torch
from torch import nn
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .layers import replace_conv1


@MODELS.register_module()
class CSPOnlyLSConvPAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, enable_lsconv=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Do not consume random draws used to construct the baseline head.
        with torch.random.fork_rng(devices=[]):
            self.lsconv_replacements = replace_conv1(self) if enable_lsconv else []

    def init_weights(self):
        if self.is_init:
            return
        removed = []
        # The neck's recursive Kaiming initializer also zeros Conv biases.
        # Preserve the complete LSConv constructor initialization, as in the
        # legacy backbone. Keep original conv2 and every outer CSP layer under
        # the official neck initializer.
        for path in self.lsconv_replacements:
            parent_path, name = path.rsplit('.', 1)
            parent = self.get_submodule(parent_path)
            removed.append((parent, name, parent._modules[name]))
            parent._modules[name] = nn.Identity()
        try:
            super().init_weights()
        finally:
            for parent, name, module in removed:
                parent._modules[name] = module
