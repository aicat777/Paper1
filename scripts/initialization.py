"""Reject pretrained initialization in the effective, fully merged config."""
from collections.abc import Mapping


def _reject_checkpoint_initializers(value, path):
    if isinstance(value, Mapping):
        kind = value.get('type')
        if isinstance(kind, str) and kind.rsplit('.', 1)[-1].lower() == 'pretrained':
            raise ValueError(f'Pretrained initialization found at {path}')
        if value.get('checkpoint') is not None:
            raise ValueError(f'Checkpoint initialization found at {path}')
        for key, item in value.items():
            _reject_checkpoint_initializers(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_checkpoint_initializers(item, f'{path}[{index}]')


def assert_random_initialization(config):
    if config.get('load_from') is not None or config.get('resume', False):
        raise ValueError('Fresh training requires load_from=None and resume=False')
    _reject_checkpoint_initializers(config.model, 'model')


def assert_model_random_initialization(model):
    # Also inspect constructor defaults, which need not appear in the config.
    for name, module in model.named_modules():
        _reject_checkpoint_initializers(getattr(module, 'init_cfg', None),
                                       f'model.{name}.init_cfg')
