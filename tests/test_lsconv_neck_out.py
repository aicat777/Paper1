"""Output replacement: actual routing, preserved initialization and protocol."""
from copy import deepcopy
import hashlib
import logging
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from mmengine.config import Config
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.lsconv_neck_out import LSConvOutput
from custom_models.lsconv_neck_out.operator import LSConv

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method='lsconv_neck_out'):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method='lsconv_neck_out', initialize=True):
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(config(method).model))
    if initialize:
        model.init_weights()
    return model


class LSConvNeckOutputs(unittest.TestCase):
    def test_operator_is_verified_core_and_protocol_matches_foundation(self):
        for name in ('operator.py', 'aggregation.py'):
            self.assertEqual(hashlib.sha256((ROOT / 'custom_models/lsconv' / name).read_bytes()).digest(),
                             hashlib.sha256((ROOT / 'custom_models/lsconv_neck_out' / name).read_bytes()).digest())
        old, new = config('lsconv'), config()
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader', 'train_cfg', 'val_cfg',
                    'test_cfg', 'val_evaluator', 'test_evaluator', 'custom_hooks', 'optim_wrapper',
                    'param_scheduler', 'default_hooks', 'randomness'):
            self.assertEqual(old[key], new[key], key)
        expected = deepcopy(old.model)
        expected.neck.type = 'LSConvOutputPAFPN'
        self.assertEqual(expected, new.model)
        self.assertEqual(new.train_dataloader.batch_size, 36)
        self.assertEqual(new.optim_wrapper.get('accumulative_counts', 1), 1)
        self.assertIsNone(new.load_from)
        self.assertFalse(new.resume)

    def test_exact_untouched_state_and_random_draw_order(self):
        old = build('lsconv')
        old_rng = torch.get_rng_state().clone()
        new = build()
        torch.testing.assert_close(torch.get_rng_state(), old_rng, atol=0, rtol=0)
        old_state = old.state_dict()
        new_state = new.state_dict()
        for name, value in old_state.items():
            if not name.startswith('neck.out_layers.'):
                torch.testing.assert_close(new_state[name], value, atol=0, rtol=0, msg=name)
        self.assertFalse(any(hasattr(m, '_params_init_info') for m in new.modules()))
        self.assertFalse(new.neck._initialization_slots)

    def test_author_mixer_init_and_official_projection_init_survive_framework(self):
        model = build(initialize=False)
        before = [deepcopy(output.mixer.state_dict()) for output in model.neck.out_layers]
        model.init_weights()
        for output, state in zip(model.neck.out_layers, before):
            for name, value in output.mixer.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            weight = output.projection.conv.weight
            bound = 1 / math.sqrt(weight.shape[1])
            self.assertLessEqual(float(weight.abs().max()), bound + 1e-7)
            self.assertGreater(float(weight.std()), bound * .4)
            torch.testing.assert_close(output.projection.bn.weight, torch.ones(96), atol=0, rtol=0)
            torch.testing.assert_close(output.projection.bn.bias, torch.zeros(96), atol=0, rtol=0)
            self.assertIsNone(output.projection.conv.bias)
        state = deepcopy(model.state_dict())
        model.init_weights()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_only_outputs_replaced_and_aggregation_precedes_projection(self):
        old, new = build('lsconv'), build()
        self.assertEqual(new.neck.lsconv_replacements, ('out_layers.0', 'out_layers.1', 'out_layers.2'))
        self.assertEqual(sum(isinstance(m, LSConv) for m in new.modules()), 3)
        for output, width in zip(new.neck.out_layers, (96, 192, 384)):
            self.assertIsInstance(output, LSConvOutput)
            self.assertEqual(output.mixer.bn.num_features, width)
            self.assertEqual((output.projection.conv.in_channels, output.projection.conv.out_channels), (width, 96))
            self.assertEqual(output.projection.conv.kernel_size, (1, 1))
            self.assertEqual(list(output._modules), ['mixer', 'projection'])
            self.assertFalse(any('ffn' in name for name, _ in output.named_modules()))
        # The dynamic cores at different scales have independent parameters.
        ids = [{id(p) for p in m.parameters()} for m in new.neck.out_layers]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        old_inputs, new_inputs, projection_inputs, handles = [], [], [], []
        def capture(target):
            def hook(module, inputs):
                target.append(inputs[0].detach().clone())
            return hook
        for output in old.neck.out_layers:
            handles.append(output.register_forward_pre_hook(capture(old_inputs)))
        for output in new.neck.out_layers:
            handles.append(output.mixer.register_forward_pre_hook(capture(new_inputs)))
            handles.append(output.projection.register_forward_pre_hook(capture(projection_inputs)))
        old.eval()
        new.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 128, 160)
            old._forward(image)
            cls, reg = new._forward(image)
        for handle in handles:
            handle.remove()
        for original, actual, after in zip(old_inputs, new_inputs, projection_inputs):
            torch.testing.assert_close(original, actual, atol=0, rtol=0)
            self.assertEqual(after.shape, actual.shape)
        self.assertEqual([tuple(x.shape) for x in cls], [(1, 20, h, w) for h, w in ((16, 20), (8, 10), (4, 5))])
        self.assertEqual([tuple(x.shape) for x in reg], [(1, 4, h, w) for h, w in ((16, 20), (8, 10), (4, 5))])

    def test_init_failures_restore_modules_and_can_retry(self):
        model = build(initialize=False)
        slots = model.neck._initialization_slots
        with patch.object(CSPNeXtPAFPN, 'init_weights', side_effect=RuntimeError('parent failure')):
            with self.assertRaisesRegex(RuntimeError, 'parent failure'):
                model.neck.init_weights()
        for path, _, replacement in slots:
            self.assertIs(model.neck.get_submodule(path), replacement)
        self.assertFalse(model.neck.is_init)
        with patch('custom_models.lsconv_neck_out.neck.initialize', side_effect=RuntimeError('projection failure')):
            with self.assertRaisesRegex(RuntimeError, 'projection failure'):
                model.neck.init_weights()
        self.assertFalse(model.neck.is_init)
        for path, _, replacement in slots:
            self.assertIs(model.neck.get_submodule(path), replacement)
        model.neck.init_weights()
        self.assertTrue(model.neck.is_init)
        self.assertFalse(model.neck._initialization_slots)


if __name__ == '__main__':
    unittest.main()
