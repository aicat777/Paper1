"""P3 control B: remove only the SiLU between pointwise and depthwise."""
from copy import deepcopy

from torch import nn

from custom_models.scdown import SCDown


class P3NoActivation(nn.Module):
    def __init__(self, reference):
        super().__init__()
        if not isinstance(reference, SCDown):
            raise TypeError('This control replaces the P3 SCDown')
        if (not reference.cv1.with_activation or not isinstance(reference.cv1.activate, nn.SiLU)
                or reference.cv2.with_activation):
            raise ValueError('Expected exactly one SiLU after the pointwise BN')
        self.cv1 = deepcopy(reference.cv1)
        self.cv2 = deepcopy(reference.cv2)
        del self.cv1.activate
        self.cv1.with_activation = False
        self.cv1.act_cfg = None

    def forward(self, x):
        return self.cv2(self.cv1(x))
