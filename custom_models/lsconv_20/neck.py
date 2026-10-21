"""Four internal CSP replacements; all three dense output layers stay intact."""
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS

from .initialization import preserve_lsconv_initialization
from .layers import replace_csp_blocks


@MODELS.register_module()
class LSConv20PAFPN(CSPNeXtPAFPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lsconv_replacements = replace_csp_blocks(self)

    def init_weights(self):
        if self.is_init:
            return
        with preserve_lsconv_initialization(self):
            super().init_weights()
