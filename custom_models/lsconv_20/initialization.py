"""Keep LSConv constructor initialization outside recursive framework init."""
from contextlib import contextmanager

from torch import nn

from .operator import LSConv


@contextmanager
def preserve_lsconv_initialization(module):
    # Only this variant's LSConv children are excluded. All surrounding layers
    # keep their normal framework initialization. Replacing slots in place also
    # preserves module order, parameter identities and scale-sharing aliases.
    slots = []
    for path, child in module.named_modules():
        if isinstance(child, LSConv):
            parent_path, _, name = path.rpartition('.')
            parent = module.get_submodule(parent_path) if parent_path else module
            slots.append((parent, name, child))
    try:
        for parent, name, _ in slots:
            parent._modules[name] = nn.Identity()
        yield
    finally:
        for parent, name, child in slots:
            parent._modules[name] = child
