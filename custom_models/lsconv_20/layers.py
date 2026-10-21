"""Equal-width CSP adapters; preserve all other convolutions and connections."""
from copy import deepcopy

from mmdet.models.layers.csp_layer import CSPNeXtBlock
from torch import nn

from .operator import LSConv


def activation(previous):
    return deepcopy(previous.activate) if previous.with_activation else nn.Identity()


def dense_3x3(previous):
    conv = getattr(previous, 'conv', None)
    if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
            or conv.stride != (1, 1) or conv.groups != 1
            or conv.in_channels != conv.out_channels):
        raise ValueError('LSConv20 requires an equal-width, stride-one dense 3x3')
    return conv


def replace_csp_blocks(container):
    paths = []
    for path, block in list(container.named_modules()):
        if isinstance(block, CSPNeXtBlock):
            previous = block.conv1
            conv = dense_3x3(previous)
            block.conv1 = nn.Sequential(LSConv(conv.in_channels), activation(previous))
            paths.append(path + '.conv1')
    return paths
