"""Structural adapters for the expanded LSConv experiment."""
from copy import deepcopy

from mmdet.models.layers.csp_layer import CSPNeXtBlock
from torch import nn

from .operator import LSConv


def activation(previous):
    return deepcopy(previous.activate) if previous.with_activation else nn.Identity()


def dense_3x3(previous, equal_width=True):
    conv = getattr(previous, 'conv', None)
    if (not isinstance(conv, nn.Conv2d) or conv.kernel_size != (3, 3)
            or conv.stride != (1, 1) or conv.groups != 1
            or (equal_width and conv.in_channels != conv.out_channels)):
        raise ValueError('Expanded LSConv requires a stride-one dense 3x3 convolution')
    return conv


def replace_csp_blocks(container):
    """Replace only conv1; retain the second convolution and CSP connections."""
    paths = []
    for path, block in list(container.named_modules()):
        if isinstance(block, CSPNeXtBlock):
            previous = block.conv1
            conv = dense_3x3(previous)
            block.conv1 = nn.Sequential(LSConv(conv.in_channels), activation(previous))
            paths.append(path + '.conv1')
    return paths
