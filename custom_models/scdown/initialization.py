"""Preserve the original constructor and initializer random-draw order."""
from contextlib import contextmanager
from copy import deepcopy

import torch
from mmengine.model import initialize


@contextmanager
def original_slots(owner, slots):
    """Temporary originals stay unregistered outside parent initialization."""
    info = getattr(owner, '_params_init_info', None)
    added, annotated = [], []
    try:
        for path, original, replacement in slots:
            parameter = next(replacement.parameters())
            original.to(device=parameter.device, dtype=parameter.dtype)
            if info is not None:
                for child in original.modules():
                    annotated.append((child, getattr(child, '_params_init_info', None)))
                    child._params_init_info = info
                for name, value in original.named_parameters():
                    if value not in info:
                        info[value] = dict(init_info='Original SCDown transition: ' + path + '.' + name,
                                           tmp_mean_value=value.data.mean().cpu())
                        added.append(value)
            parent_path, _, name = path.rpartition('.')
            parent = owner.get_submodule(parent_path) if parent_path else owner
            parent._modules[name] = original
        yield
    finally:
        for path, _, replacement in slots:
            parent_path, _, name = path.rpartition('.')
            parent = owner.get_submodule(parent_path) if parent_path else owner
            parent._modules[name] = replacement
        for child, previous in reversed(annotated):
            if previous is None:
                child.__dict__.pop('_params_init_info', None)
            else:
                child._params_init_info = previous
        if info is not None:
            for value in added:
                info.pop(value, None)


def initialize_scdown(owner, parent_init):
    if owner.is_init:
        return
    slots = owner._scdown_initialization_slots
    with original_slots(owner, slots):
        parent_init()
    try:
        # Match each owner's existing Kaiming policy. An init_cfg=None
        # backbone already uses ConvModule's constructor initialization.
        with torch.random.fork_rng(devices=[]):
            if owner.init_cfg is not None:
                for _, _, replacement in slots:
                    initialize(replacement, deepcopy(owner.init_cfg))
    except Exception:
        owner._is_init = False
        raise
    owner._scdown_initialization_slots = ()
