"""Crop batches use their own statistics without changing inference buffers."""
from contextlib import contextmanager
from types import MethodType

from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def crop_batch_norm(module, inputs):
    # A single crop at a 1x1 deepest map has no defined batch variance. Only
    # that degenerate case uses the main running estimates; affine gradients
    # remain enabled. No input, parameter, or loss is detached.
    batch_statistics = inputs.numel() // inputs.shape[1] > 1
    return F.batch_norm(inputs,
        None if batch_statistics else module.running_mean,
        None if batch_statistics else module.running_var,
        module.weight, module.bias, training=batch_statistics, momentum=0., eps=module.eps)


@contextmanager
def crop_normalization(model):
    originals = []
    try:
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                originals.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))
                module.forward = MethodType(crop_batch_norm, module)
        yield
    finally:
        for module, existed, forward in originals:
            if existed:
                module.forward = forward
            else:
                module.__dict__.pop('forward', None)
