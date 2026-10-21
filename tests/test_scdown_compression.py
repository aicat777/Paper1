"""G4 and DSConv: independent numerical references and full-model isolation."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.scdown_g4.operator import grouped_conv_from
from custom_models.scdown_dsconv.operator import depthwise_separable_from
from scripts.scdown_compression_probe import SPECS, GROUP_SITES, CONTROL_SITES, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
register_all_modules(); torch.set_num_threads(2); logging.disable(logging.CRITICAL)


def build(method='scdown', initialize=True):
    cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


def original_conv(width=48, **kwargs):
    options = dict(in_channels=width, out_channels=width, kernel_size=3, padding=1,
                   norm_cfg=dict(type='BN', eps=.001, momentum=.03),
                   act_cfg=dict(type='SiLU', inplace=True))
    options.update(kwargs)
    return ConvModule(**options)


def data():
    torch.manual_seed(31)
    images = torch.randn(2, 3, 128, 128)
    targets = dict(bboxes_labels=torch.tensor(
        [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
        dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)
    return images, targets


def plain_torch_dsconv_reference(ours):
    """Independent standard PyTorch stack, explicitly spatial then channel."""
    width = ours.depthwise_conv.conv.in_channels
    dw_bn = ours.depthwise_conv.bn
    pw_bn = ours.pointwise_conv.bn
    reference = torch.nn.Sequential(
        torch.nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
        torch.nn.BatchNorm2d(width, eps=dw_bn.eps, momentum=dw_bn.momentum),
        torch.nn.SiLU(), torch.nn.Conv2d(width, width, 1, bias=False),
        torch.nn.BatchNorm2d(width, eps=pw_bn.eps, momentum=pw_bn.momentum), torch.nn.SiLU())
    reference[0].load_state_dict(ours.depthwise_conv.conv.state_dict())
    reference[1].load_state_dict(dw_bn.state_dict())
    reference[3].load_state_dict(ours.pointwise_conv.conv.state_dict())
    reference[4].load_state_dict(pw_bn.state_dict())
    return reference


class SCDownCompressionTests(unittest.TestCase):
    def test_exact_scope_operator_order_dimensions_and_sharing(self):
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                model, control = build(method).eval(), build().eval()
                self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), CONTROL_SITES)
                self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
                self.assertEqual(len(watched_parameters(model, method)), spec['watched'])
                groups = tuple(n.removesuffix('.conv') for n, m in model.named_modules()
                               if isinstance(m, torch.nn.Conv2d) and m.groups == 4)
                self.assertEqual(groups, GROUP_SITES if method == 'scdown_g4' else ())
                for path in GROUP_SITES:
                    layer = model.get_submodule(path)
                    original = control.get_submodule(path)
                    width = original.conv.in_channels
                    if method == 'scdown_g4':
                        self.assertEqual(layer.conv.weight.shape, (width, width // 4, 3, 3))
                        parts = (layer,)
                    else:
                        self.assertIsInstance(layer, DepthwiseSeparableConvModule)
                        dw, pw = layer.depthwise_conv, layer.pointwise_conv
                        self.assertEqual(dw.conv.weight.shape, (width, 1, 3, 3))
                        self.assertEqual(dw.conv.groups, width)
                        self.assertEqual(pw.conv.weight.shape, (width, width, 1, 1))
                        self.assertEqual(pw.conv.groups, 1)
                        parts = (dw, pw)
                    for part in parts:
                        self.assertEqual(part.norm_cfg, original.norm_cfg)
                        self.assertEqual(part.act_cfg, original.act_cfg)
                        self.assertEqual(part.order, ('conv', 'norm', 'act'))
                        self.assertIsNone(part.conv.bias)
                    block = model.get_submodule(path.rpartition('.')[0])
                    self.assertEqual(block.conv2.depthwise_conv.conv.kernel_size, (5, 5))
                    self.assertEqual(block.conv2.pointwise_conv.conv.groups, 1)
                for name, layer in control.named_modules():
                    if not name or any(name == p or name.startswith(p + '.') for p in GROUP_SITES):
                        continue
                    # Backbone/neck containers intentionally have a new type.
                    if name not in ('backbone', 'neck'):
                        self.assertIs(type(model.get_submodule(name)), type(layer), name)
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
                priors = model.bbox_head.prior_generator.grid_priors(
                    [v.shape[-2:] for v in cls], device='cpu', with_stride=True)
                self.assertEqual(sum(len(p) for p in priors), 8400)

    def test_unmodified_initialization_rng_and_control_scdowns_are_exact(self):
        for method in SPECS:
            for initialize in (False, True):
                with self.subTest(method=method, initialized=initialize):
                    control = build(initialize=initialize)
                    expected_rng = torch.get_rng_state().clone()
                    model = build(method, initialize)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    for name, value in control.state_dict().items():
                        if not any(name.startswith(site + '.') for site in GROUP_SITES):
                            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
                    if initialize:
                        state = {n: v.clone() for n, v in model.state_dict().items()}
                        model.init_weights()
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                        for name, value in state.items():
                            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
                        self.assertEqual(model.backbone._scdown_initialization_slots, ())
                        self.assertEqual(model.neck._scdown_initialization_slots, ())

    def test_restoring_seven_sites_recovers_predictions_losses_and_gradients_exactly(self):
        images, targets = data()
        for method in SPECS:
            with self.subTest(method=method):
                control, model = build(), build(method)
                for path in GROUP_SITES:
                    parent, _, name = path.rpartition('.')
                    model.get_submodule(parent)._modules[name] = deepcopy(control.get_submodule(path))
                control.eval(); model.eval()
                with torch.no_grad():
                    for aa, bb in zip(control._forward(images), model._forward(images)):
                        for a, b in zip(aa, bb):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                control.train(); model.train()
                original, actual = control.loss(images, targets), model.loss(images, targets)
                self.assertEqual(set(actual), {'loss_cls', 'loss_bbox'})
                for key in original:
                    torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0, msg=key)
                sum(original.values()).backward(); sum(actual.values()).backward()
                parameters = dict(control.named_parameters())
                for name, p in model.named_parameters():
                    self.assertEqual(p.grad is None, parameters[name].grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_g4_forward_and_gradients_match_dense_block_diagonal_reference(self):
        for width in (48, 96, 192):
            for training in (False, True):
                with self.subTest(width=width, training=training):
                    torch.manual_seed(71)
                    dense = original_conv(width).train(training)
                    grouped = grouped_conv_from(dense).train(training)
                    group_width = width // 4
                    with torch.no_grad():
                        dense.conv.weight.zero_()
                        for group in range(4):
                            span = slice(group * group_width, (group + 1) * group_width)
                            dense.conv.weight[span, span].copy_(grouped.conv.weight[span])
                    x = torch.randn(2, width, 9, 13, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    actual, expected = grouped(x), dense(reference_x)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                    seed = torch.randn_like(actual) / actual.numel() ** .5
                    gradients = torch.autograd.grad(actual, (x, *grouped.parameters()), seed)
                    reference = torch.autograd.grad(expected, (reference_x, *dense.parameters()), seed)
                    blocks = []
                    for group in range(4):
                        span = slice(group * group_width, (group + 1) * group_width)
                        blocks.append(reference[1][span, span])
                    for a, b in zip(gradients, (reference[0], torch.cat(blocks), *reference[2:])):
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                    for a, b in zip(grouped.norm.buffers(), dense.norm.buffers()):
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                    with torch.no_grad():
                        isolated = torch.zeros(1, width, 7, 11)
                        isolated[:, :group_width] = torch.randn_like(isolated[:, :group_width])
                        self.assertEqual(float(grouped.conv(isolated)[:, group_width:].abs().max()), 0)

    def test_dsconv_forward_and_gradients_match_plain_torch_ordered_reference(self):
        for width in (48, 96, 192):
            for training in (False, True):
                for h, w in ((9, 13), (10, 14)):
                    with self.subTest(width=width, training=training, hw=(h, w)):
                        torch.manual_seed(19)
                        ours = depthwise_separable_from(original_conv(width)).train(training)
                        reference = plain_torch_dsconv_reference(ours).train(training)
                        x = torch.randn(2, width, h, w, requires_grad=True)
                        reference_x = x.detach().clone().requires_grad_()
                        actual, expected = ours(x), reference(reference_x)
                        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                        seed = torch.randn_like(actual) / actual.numel() ** .5
                        gradients = torch.autograd.grad(actual, (x, *ours.parameters()), seed)
                        expected_gradients = torch.autograd.grad(expected, (reference_x, *reference.parameters()), seed)
                        for a, b in zip(gradients, expected_gradients):
                            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                        for a, b in zip(ours.buffers(), reference.buffers()):
                            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_detection_gradients_actual_updates_and_each_bn_once(self):
        images, targets = data()
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                model = build(method).train()
                losses = model.loss(images, targets)
                self.assertEqual(set(losses), {'loss_cls', 'loss_bbox'})
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                sum(losses.values()).backward()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
                watched = watched_parameters(model, method)
                self.assertEqual(len(watched), spec['watched'])
                before = {n: p.detach().clone() for n, p in watched.items()}
                for name, p in watched.items():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(float(p.grad.abs().max()), 0, name)
                optimizer.step()
                for name, p in watched.items():
                    self.assertGreater(float((p - before[name]).abs().max()), 0, name)
                for name, layer in model.named_modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_checkpoint_roundtrip_preserves_predictions_and_shared_head(self):
        images, _ = data()
        for method in SPECS:
            with self.subTest(method=method):
                model = build(method).eval()
                buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
                restored = build(method, initialize=False).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
                with torch.no_grad():
                    for aa, bb in zip(model._forward(images), restored._forward(images)):
                        for a, b in zip(aa, bb):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                head = restored.bbox_head.head_module
                self.assertIs(head.cls_convs[0][0].conv, head.cls_convs[2][0].conv)
                self.assertIsNot(head.cls_convs[0][0].bn, head.cls_convs[2][0].bn)

    def test_invalid_replacement_is_rejected_instead_of_adapting_dimensions(self):
        invalid = (dict(stride=2), dict(groups=2), dict(out_channels=96),
                   dict(bias=True), dict(dilation=2, padding=2),
                   dict(act_cfg=dict(type='ReLU')), dict(norm_cfg=None))
        for factory in (grouped_conv_from, depthwise_separable_from):
            for options in invalid:
                with self.subTest(factory=factory.__name__, options=options):
                    with self.assertRaises(ValueError):
                        factory(original_conv(**options))
            with self.assertRaises(TypeError):
                factory(torch.nn.Conv2d(48, 48, 3, padding=1))
        with self.assertRaises(ValueError):
            grouped_conv_from(original_conv(50))


if __name__ == '__main__':
    unittest.main()
