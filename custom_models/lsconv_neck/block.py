"""LSNet's LSConv-containing block: spatial residual, then channel residual."""
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from torch import nn

from .operator import ConvBN, LSConv


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def forward(self, x):
        return x + self.m(x)


class ChannelFFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pw1 = ConvBN(channels, 2 * channels)
        self.act = nn.ReLU()
        self.pw2 = ConvBN(2 * channels, channels)
        nn.init.zeros_(self.pw2.bn.weight)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class LSFeatureBlock(nn.Module):
    """Same LSConv + residual FFN as the author's stage<3, odd-depth Block.

    The mixer and FFN each have their own residual. No extra activation or
    outer residual is applied; the adapter decides which original block or
    tower layer this complete feature block replaces.
    """
    def __init__(self, channels):
        super().__init__()
        self.mixer = LSConv(channels)
        self.ffn = Residual(ChannelFFN(channels))

    def forward(self, x):
        return self.ffn(self.mixer(x))


def replace_blocks(container):
    paths = []
    for path, block in list(container.named_modules()):
        if not isinstance(block, CSPNeXtBlock):
            continue
        conv = block.conv1.conv
        if (not isinstance(conv, nn.Conv2d) or conv.in_channels != conv.out_channels
                or conv.groups != 1 or conv.stride != (1, 1)
                or block.conv2.pointwise_conv.conv.out_channels != conv.in_channels):
            raise ValueError('An LS feature block requires equal CSP internal widths')
        parent_path, name = path.rsplit('.', 1)
        container.get_submodule(parent_path).add_module(name, LSFeatureBlock(conv.in_channels))
        paths.append(path)
    return paths
