"""Initialize surviving layers exactly as the original architecture does."""
from contextlib import contextmanager


@contextmanager
def original_initialization_slots(owner, slots):
    """Temporarily use original blocks for framework init, then restore LS blocks.

    Original blocks are held in an unregistered tuple only until initialization.
    This preserves both the original random-draw order and LSNet's constructor
    initialization, including its zero-scale final FFN BN. MMEngine's transient
    initialization metadata also needs entries for the temporary originals.
    """
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
                        info[value] = dict(init_info='Temporary original layer: ' + path + '.' + name,
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
