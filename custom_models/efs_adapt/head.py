"""Keep baseline P3-P5 convolution sharing and give P2 separate towers."""
from copy import deepcopy

from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.registry import MODELS


@MODELS.register_module()
class EFSP2IndependentHeadModule(RTMDetSepBNHeadModule):
    """P2 has independent cls/reg convolutions; P3-P5 keep their shared group.

    BN and final predictions remain independent per scale, as in RTMDet. The
    parent RTMDetHead still performs joint assignment, loss and box decoding.
    """

    def _init_layers(self):
        if tuple(self.featmap_strides) != (4, 8, 16, 32) or not self.share_conv:
            raise ValueError('P2 independence requires strides 4/8/16/32 and '
                             'the baseline shared P3-P5 towers')
        super()._init_layers()
        # The parent aliases every scale to P2. P3-P5 retain those original
        # convolutions; detach only P2 into a distinct set of parameters.
        for towers in (self.cls_convs, self.reg_convs):
            for layer_idx in range(self.stacked_convs):
                towers[0][layer_idx].conv = deepcopy(towers[0][layer_idx].conv)
