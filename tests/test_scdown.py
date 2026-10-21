"""Audit the three transitions, baseline parity, coordinates and gradients."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.scdown.initialization import original_slots

ROOT = Path(__file__).resolve().parents[1]
SITES = ('backbone.stage3.0', 'backbone.stage4.0', 'neck.downsample_layers.1')
register_all_modules(); torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build_model(candidate=True, initialize=True):
    path = 'configs/scdown/rtmdet_tiny_dior_640.py' if candidate else 'configs/rtmdet_tiny_dior_300e.py'
    cfg = Config.fromfile(str(ROOT / path))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


def data():
    torch.manual_seed(31)
    images = torch.randn(2, 3, 128, 128)
    targets = dict(bboxes_labels=torch.tensor(
        [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112],
         [1, 3, 40, 36, 80, 76]], dtype=torch.float32),
        img_metas=[dict(img_shape=(128, 128))] * 2)
    return images, targets


class SCDownTests(unittest.TestCase):
    def test_channel_mix_before_decimation_odd_grid_and_signed_output(self):
        torch.manual_seed(11)
        for height, width in ((7, 9), (8, 10)):
            layer = SCDown(3, 5).eval()
            matrix = torch.randn(5, 3)
            with torch.no_grad():
                layer.cv1.conv.weight.copy_(matrix[:, :, None, None])
                layer.cv2.conv.weight.zero_()
                layer.cv2.conv.weight[:, 0, 1, 1] = -1
                for part in (layer.cv1, layer.cv2):
                    part.bn.running_mean.zero_(); part.bn.running_var.fill_(1)
                    part.bn.weight.fill_((1 + part.bn.eps) ** .5)
                    part.bn.bias.zero_()
            x = torch.randn(2, 3, height, width)
            x[:, :, 0, 0] += 10  # Include the top-left sampling center/border.
            expected = -F.silu(torch.einsum('oc,bchw->bohw', matrix, x))[:, :, ::2, ::2]
            actual = layer(x)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            self.assertTrue((actual < 0).any())
            self.assertFalse(layer.cv2.with_activation)
            self.assertEqual(layer.cv2.conv.groups, 5)

    def test_untouched_initialization_constructor_and_init_rng_exact(self):
        for initialize in (False, True):
            original = build_model(False, initialize)
            expected_rng = torch.get_rng_state().clone()
            model = build_model(True, initialize)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
            original_state, actual_state = original.state_dict(), model.state_dict()
            for name, value in original_state.items():
                if not any(name.startswith(path + '.') for path in SITES):
                    torch.testing.assert_close(value, actual_state[name], atol=0, rtol=0, msg=name)
            if initialize:
                before = {n: v.clone() for n, v in actual_state.items()}
                rng = torch.get_rng_state().clone()
                model.init_weights()
                torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                for name, value in before.items():
                    torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
                self.assertEqual(model.backbone._scdown_initialization_slots, ())
                self.assertEqual(model.neck._scdown_initialization_slots, ())

    def test_exact_scope_actual_widths_outputs_priors_and_head_sharing(self):
        model = build_model().eval()
        self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), SITES)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3854964)
        trace, handles = [], []
        for path in SITES:
            handles.append(model.get_submodule(path).register_forward_hook(
                lambda mod, inputs, output, path=path: trace.append((path, list(inputs[0].shape), list(output.shape)))))
        with torch.no_grad():
            cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        self.assertEqual(trace, [(SITES[0], [1, 96, 80, 80], [1, 192, 40, 40]),
                                 (SITES[1], [1, 192, 40, 40], [1, 384, 20, 20]),
                                 (SITES[2], [1, 192, 40, 40], [1, 192, 20, 20])])
        self.assertEqual([list(v.shape) for v in cls], [[1, 20, s, s] for s in (80, 40, 20)])
        self.assertEqual([list(v.shape) for v in reg], [[1, 4, s, s] for s in (80, 40, 20)])
        priors = model.bbox_head.prior_generator.grid_priors(
            [v.shape[-2:] for v in cls], device='cpu', with_stride=True)
        self.assertEqual(sum(len(p) for p in priors), 8400)
        for i, output in enumerate(model.neck.out_layers):
            self.assertEqual(output.conv.kernel_size, (3, 3))
            self.assertEqual(output.conv.in_channels, (96, 192, 384)[i])
        self.assertEqual(model.neck.downsample_layers[0].conv.kernel_size, (3, 3))
        head = model.bbox_head.head_module
        for towers in (head.cls_convs, head.reg_convs):
            for depth in range(2):
                self.assertIs(towers[0][depth].conv, towers[1][depth].conv)
                self.assertIsNot(towers[0][depth].bn, towers[1][depth].bn)

    def test_restoring_three_originals_recovers_predictions_losses_gradients(self):
        original, model = build_model(False), build_model(True)
        for path in SITES:
            parent, _, leaf = path.rpartition('.')
            model.get_submodule(parent)._modules[leaf] = deepcopy(original.get_submodule(path))
        images, targets = data()
        original.eval(); model.eval()
        with torch.no_grad():
            for first, second in zip(original._forward(images), model._forward(images)):
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        original.train(); model.train()
        first, second = original.loss(images, targets), model.loss(images, targets)
        self.assertEqual(set(first), {'loss_cls', 'loss_bbox'})
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0, msg=key)
        sum(first.values()).backward(); sum(second.values()).backward()
        parameters = dict(original.named_parameters())
        for name, p in model.named_parameters():
            self.assertEqual(p.grad is None, parameters[name].grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_real_detection_gradients_and_one_bn_update_per_main_forward(self):
        model = build_model().train()
        images, targets = data()
        losses = model.loss(images, targets)
        self.assertEqual(set(losses), {'loss_cls', 'loss_bbox'})
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        sum(losses.values()).backward()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        initial = {}
        for path in SITES:
            for name, p in model.get_submodule(path).named_parameters():
                self.assertIsNotNone(p.grad, path + '.' + name)
                self.assertTrue(torch.isfinite(p.grad).all(), path + '.' + name)
                self.assertGreater(p.grad.abs().max().item(), 0, path + '.' + name)
                initial[path + '.' + name] = p.detach().clone()
        optimizer.step()
        for name, value in initial.items():
            self.assertGreater((model.get_parameter(name) - value).abs().max().item(), 0, name)
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_state_reload_keeps_prediction_and_scale_sharing(self):
        model = build_model().eval()
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = build_model(initialize=False).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        images, _ = data()
        with torch.no_grad():
            for first, second in zip(model._forward(images), restored._forward(images)):
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        head = restored.bbox_head.head_module
        self.assertIs(head.cls_convs[0][0].conv, head.cls_convs[2][0].conv)

    def test_initialization_exception_restores_modules_and_allows_retry(self):
        model = build_model(initialize=False)
        replacements = tuple(model.get_submodule(path) for path in SITES)
        with patch('custom_models.scdown.initialization.initialize', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                model.init_weights()
        for path, module in zip(SITES, replacements):
            self.assertIs(model.get_submodule(path), module)
        self.assertFalse(model.neck.is_init)
        model.init_weights()
        self.assertTrue(model.neck.is_init)
        self.assertEqual(model.neck._scdown_initialization_slots, ())
        # Also exercise failure inside the temporary-original context itself.
        original = model.neck.downsample_layers[0]
        replacement = model.neck.downsample_layers[1]
        with self.assertRaisesRegex(RuntimeError, 'context'):
            with original_slots(model.neck, (('downsample_layers.1', original, replacement),)):
                self.assertIs(model.neck.downsample_layers[1], original)
                raise RuntimeError('context')
        self.assertIs(model.neck.downsample_layers[1], replacement)

    def test_protocol_and_invalid_replacement_contract(self):
        original = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        cfg = Config.fromfile(str(ROOT / 'configs/scdown/rtmdet_tiny_dior_640.py'))
        expected = deepcopy(original.model)
        expected.backbone.type = 'SCDownCSPNeXt'; expected.neck.type = 'SCDownPAFPN'
        self.assertEqual(cfg.model, expected)
        for key in ('train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'val_dataloader',
                    'val_evaluator', 'test_dataloader', 'test_evaluator', 'train_cfg', 'custom_hooks',
                    'default_hooks', 'randomness'):
            self.assertEqual(cfg[key], original[key], key)
        self.assertEqual(cfg.train_dataloader.batch_size, 36)
        self.assertEqual(cfg.optim_wrapper.get('accumulative_counts', 1), 1)
        self.assertAlmostEqual(cfg.optim_wrapper.optimizer.lr, .0005625)
        self.assertEqual(cfg.param_scheduler[0].end, 1000)
        self.assertIsNone(cfg.load_from); self.assertFalse(cfg.resume)
        self.assertIsNone(cfg.model.backbone.init_cfg)
        self.assertEqual(cfg.img_scale, (640, 640))
        self.assertEqual(cfg.train_cfg.max_epochs, 300)
        self.assertEqual(cfg.train_cfg.val_interval, 10)
        model = build_model(False)
        with self.assertRaises(ValueError):
            SCDown.from_conv_module(model.neck.out_layers[0])
        with self.assertRaises(ValueError):
            SCDown(4, 8, act_cfg=dict(type='ReLU'))

    def test_inference_bn_fusion_handles_distinct_stats_on_shared_head_convs(self):
        from scripts.scdown_probe import fuse_for_latency
        model = build_model().eval()
        for towers in (model.bbox_head.head_module.cls_convs, model.bbox_head.head_module.reg_convs):
            for level, tower in enumerate(towers):
                for layer in tower:
                    layer.bn.running_mean.fill_(.3 * level)
                    layer.bn.running_var.fill_(.5 + level)
        images, _ = data()
        with torch.no_grad():
            expected = model._forward(images)
            self.assertEqual(fuse_for_latency(model), 8)
            actual = model._forward(images)
        for first, second in zip(actual, expected):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        head = model.bbox_head.head_module
        self.assertIsNot(head.cls_convs[0][0].conv, head.cls_convs[1][0].conv)


if __name__ == '__main__':
    unittest.main()
