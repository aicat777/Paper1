"""The original three-site adapter, extended only to CSP internal conv1."""
from copy import deepcopy

from mmdet.models.layers.csp_layer import CSPNeXtBlock
from torch import nn

from .operator import LSConv


def replace_conv1(container):
    paths = []
    for path, block in list(container.named_modules()):
        if not isinstance(block, CSPNeXtBlock):
            continue
        previous = block.conv1
        conv = getattr(previous, 'conv', None)
        if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
                or conv.stride != (1, 1) or conv.groups != 1
                or conv.in_channels != conv.out_channels):
            raise ValueError('CSP-only LSConv requires an equal-width dense stride-one 3x3')
        act = deepcopy(previous.activate) if previous.with_activation else nn.Identity()
        block.conv1 = nn.Sequential(LSConv(conv.in_channels), act)
        paths.append(path + '.conv1')
    return paths
