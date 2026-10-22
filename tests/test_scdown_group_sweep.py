"""G8/G16 change only the established seven spatial convolutions."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import torch
from mmcv.cnn import ConvModule
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.scdown_g8.operator import grouped_conv_from as g8_from
from custom_models.scdown_g16.operator import grouped_conv_from as g16_from
from scripts.scdown_group_sweep_probe import CONTROL_SITES, GROUP_SITES, SPECS, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build(method='scdown', initialize=True):
    cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


def detection_data():
    torch.manual_seed(31)
    return torch.randn(2, 3, 128, 128), dict(
        bboxes_labels=torch.tensor(
            [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
            dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)


class SCDownGroupSweepTests(unittest.TestCase):
    def test_scope_dimensions_channels_and_head_sharing(self):
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                model = build(method).eval()
                control = build().eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
                self.assertEqual(len(watched_parameters(model, method)), 39)
                self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), CONTROL_SITES)
                self.assertEqual(tuple(n.removesuffix('.conv') for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Conv2d) and m.groups == spec['group_count']), GROUP_SITES)
                for site in GROUP_SITES:
                    layer = model.get_submodule(site)
                    width = layer.conv.in_channels
                    self.assertEqual(layer.conv.weight.shape, (width, width // spec['group_count'], 3, 3))
                    self.assertEqual(layer.conv.stride, (1, 1))
                    self.assertEqual(layer.conv.padding, (1, 1))
                    original = control.get_submodule(site)
                    self.assertEqual(layer.norm_cfg, original.norm_cfg)
                    self.assertEqual(layer.norm.eps, original.norm.eps)
                    self.assertEqual(layer.norm.momentum, original.norm.momentum)
                    self.assertIsInstance(layer.activate, torch.nn.SiLU)
                    block = model.get_submodule(site.rpartition('.')[0])
                    self.assertEqual(block.conv2.depthwise_conv.conv.groups, width)
                    self.assertEqual(block.conv2.pointwise_conv.conv.groups, 1)
                self.assertEqual(model.backbone.stage2[0].conv.groups, 1)
                self.assertEqual(model.backbone.stage1[1].blocks[0].conv1.conv.groups, 1)
                self.assertEqual(model.neck.downsample_layers[0].conv.groups, 1)
                for layer in model.neck.out_layers:
                    self.assertEqual(layer.conv.kernel_size, (3, 3))
                    self.assertEqual(layer.conv.groups, 1)
                head = model.bbox_head.head_module
                for towers in (head.cls_convs, head.reg_convs):
                    for depth in range(2):
                        for scale in (1, 2):
                            self.assertIs(towers[0][depth].conv, towers[scale][depth].conv)
                            self.assertIsNot(towers[0][depth].bn, towers[scale][depth].bn)
                with torch.no_grad():
                    cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
                self.assertEqual([list(v.shape) for v in cls], [[1, 20, s, s] for s in (80, 40, 20)])
                self.assertEqual([list(v.shape) for v in reg], [[1, 4, s, s] for s in (80, 40, 20)])
                priors = model.bbox_head.prior_generator.grid_priors([v.shape[-2:] for v in cls])
                self.assertEqual(sum(len(p) for p in priors), 8400)

    def test_protocol_matches_existing_group_controls(self):
        def normalized(method):
            cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py')).to_dict()
            for key in ('custom_imports', 'work_dir'):
                cfg.pop(key)
            for component in ('backbone', 'neck'):
                cfg['model'][component].pop('type')
            return cfg
        expected = normalized('scdown')
        for method in (*SPECS, 'scdown_g2', 'scdown_g4'):
            self.assertEqual(normalized(method), expected, method)

    def test_untouched_initialization_random_state_and_idempotence(self):
        for method in SPECS:
            for initialize in (False, True):
                with self.subTest(method=method, initialize=initialize):
                    control = build(initialize=initialize)
                    expected_rng = torch.get_rng_state().clone()
                    model = build(method, initialize)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    actual = model.state_dict()
                    for name, value in control.state_dict().items():
                        if not any(name.startswith(site + '.') for site in GROUP_SITES):
                            torch.testing.assert_close(value, actual[name], atol=0, rtol=0, msg=name)
                    if initialize:
                        self.assertEqual(model.backbone._scdown_initialization_slots, ())
                        self.assertEqual(model.neck._scdown_initialization_slots, ())
                        state = {name: value.clone() for name, value in actual.items()}
                        model.init_weights()
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(value, state[name], atol=0, rtol=0, msg=name)

    def test_restoring_seven_sites_recovers_full_predictions_losses_and_gradients(self):
        images, targets = detection_data()
        for method in SPECS:
            with self.subTest(method=method):
                control, model = build(), build(method)
                for site in GROUP_SITES:
                    parent, _, name = site.rpartition('.')
                    model.get_submodule(parent)._modules[name] = deepcopy(control.get_submodule(site))
                control.eval()
                model.eval()
                with torch.no_grad():
                    for expected, actual in zip(control._forward(images), model._forward(images)):
                        for a, b in zip(expected, actual):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                control.train()
                model.train()
                expected, actual = control.loss(images, targets), model.loss(images, targets)
                self.assertEqual(set(actual), {'loss_cls', 'loss_bbox'})
                for key in expected:
                    torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
                sum(expected.values()).backward()
                sum(actual.values()).backward()
                parameters = dict(control.named_parameters())
                for name, p in model.named_parameters():
                    self.assertEqual(p.grad is None, parameters[name].grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_group_partitions_match_block_diagonal_dense_forward_and_backward(self):
        for groups, factory in ((8, g8_from), (16, g16_from)):
            for width in (48, 96, 192):
                for training in (False, True):
                    with self.subTest(groups=groups, width=width, training=training):
                        torch.manual_seed(71)
                        dense = ConvModule(width, width, 3, padding=1,
                            norm_cfg=dict(type='BN', eps=.001, momentum=.03),
                            act_cfg=dict(type='SiLU', inplace=True)).train(training)
                        grouped = factory(dense).train(training)
                        chunk = width // groups
                        with torch.no_grad():
                            dense.conv.weight.zero_()
                            for group in range(groups):
                                section = slice(group * chunk, (group + 1) * chunk)
                                dense.conv.weight[section, section].copy_(grouped.conv.weight[section])
                        x = torch.randn(2, width, 9, 13, requires_grad=True)
                        reference_x = x.detach().clone().requires_grad_()
                        actual, expected = grouped(x), dense(reference_x)
                        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                        seed = torch.randn_like(actual) / actual.numel() ** .5
                        gradients = torch.autograd.grad(actual, (x, *grouped.parameters()), seed)
                        reference = torch.autograd.grad(expected, (reference_x, *dense.parameters()), seed)
                        blocks = []
                        for group in range(groups):
                            section = slice(group * chunk, (group + 1) * chunk)
                            blocks.append(reference[1][section, section])
                        aligned = (reference[0], torch.cat(blocks), *reference[2:])
                        for a, b in zip(gradients, aligned):
                            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                        for a, b in zip(grouped.norm.buffers(), dense.norm.buffers()):
                            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)

    def test_detection_gradients_updates_bn_and_checkpoint_restore(self):
        images, targets = detection_data()
        for method in SPECS:
            with self.subTest(method=method):
                model = build(method).train()
                watched = watched_parameters(model, method)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
                included = [p for group in optimizer.param_groups for p in group['params']]
                self.assertEqual(len(included), len({id(p) for p in included}))
                self.assertTrue({id(p) for p in watched.values()} <= {id(p) for p in included})
                initial = {name: p.detach().clone() for name, p in watched.items()}
                counters = {name: value.clone() for name, value in model.named_buffers()
                    if name.endswith('num_batches_tracked') and
                    any(name.startswith(site + '.') for site in (*CONTROL_SITES, *GROUP_SITES))}
                gradients = set()
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    losses = model.loss(images, targets)
                    self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                    sum(losses.values()).backward()
                    for name, p in watched.items():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                        if p.grad.abs().max() > 0:
                            gradients.add(name)
                    optimizer.step()
                    for name, value in model.named_buffers():
                        if name in counters:
                            self.assertEqual(int(value), int(counters[name]) + step + 1, name)
                self.assertEqual(gradients, set(watched))
                for name, p in watched.items():
                    self.assertGreater(float((p.detach() - initial[name]).abs().max()), 0, name)
                stream = io.BytesIO()
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
                stream.seek(0)
                saved = torch.load(stream, map_location='cpu')
                restored = build(method)
                restored.load_state_dict(saved['model'], strict=True)
                restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.0005625)
                restored_optimizer.load_state_dict(saved['optimizer'])
                for name, p in restored.named_parameters():
                    torch.testing.assert_close(p, dict(model.named_parameters())[name], atol=0, rtol=0)
                    self.assertEqual(int(restored_optimizer.state[p]['step']), 2)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    for expected, actual in zip(model._forward(images), restored._forward(images)):
                        for a, b in zip(expected, actual):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
