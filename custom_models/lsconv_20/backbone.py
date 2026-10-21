"""LSConv in the four backbone CSP blocks; preserve the original conv2."""
from mmyolo.models.backbones import CSPNeXt
from mmyolo.registry import MODELS

from .layers import replace_csp_blocks


@MODELS.register_module()
class LSConv20CSPNeXt(CSPNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsconv_replacements = replace_csp_blocks(self)
