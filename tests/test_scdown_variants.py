"""Check isolated changes against SCDown3, including exact restoration parity."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from mmcv.cnn import ConvModule
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.scdown_g2.operator import grouped_conv_from
from scripts.scdown_variants_probe import SPECS, GROUP_SITES, CONTROL_SITES, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
register_all_modules(); torch.set_num_threads(2); logging.disable(logging.CRITICAL)


def build(method='scdown', initialize=True):
    cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


class SCDownVariantTests(unittest.TestCase):
    def test_scope_shapes_channels_sharing_and_counts(self):
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                model = build(method).eval()
                self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), spec['scdown'])
                groups = tuple(n.removesuffix('.conv') for n, m in model.named_modules()
                               if isinstance(m, torch.nn.Conv2d) and m.groups == 2)
                self.assertEqual(groups, spec['groups'])
                self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
                self.assertEqual(len(watched_parameters(model, method)), spec['watched'])
                for path in spec['groups']:
                    layer = model.get_submodule(path)
                    width = layer.conv.in_channels
                    self.assertIn(width, (48, 96, 192))
                    self.assertEqual(layer.conv.out_channels, width)
                    self.assertEqual(layer.conv.weight.shape, (width, width // 2, 3, 3))
                    block = model.get_submodule(path.rpartition('.')[0])
                    self.assertEqual(block.conv2.depthwise_conv.conv.groups, width)
                    self.assertEqual(block.conv2.pointwise_conv.conv.groups, 1)
                # The original first PAN transition, output projections and
                # head towers must not be included in any replacement set.
                self.assertEqual(model.neck.downsample_layers[0].conv.groups, 1)
                self.assertEqual(model.backbone.stage1[1].blocks[0].conv1.conv.groups, 1)
                for output in model.neck.out_layers:
                    self.assertEqual(output.conv.groups, 1)
                    self.assertEqual(output.conv.kernel_size, (3, 3))
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

    def test_untouched_and_control_scdowns_initialization_rng_exact(self):
        for method, spec in SPECS.items():
            for initialize in (False, True):
                with self.subTest(method=method, initialized=initialize):
                    control = build(initialize=initialize)
                    expected_rng = torch.get_rng_state().clone()
                    model = build(method, initialize)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    state = model.state_dict()
                    for name, value in control.state_dict().items():
                        if not any(name.startswith(path + '.') for path in spec['extra']):
                            torch.testing.assert_close(value, state[name], atol=0, rtol=0, msg=name)
                    if initialize:
                        self.assertEqual(model.backbone._scdown_initialization_slots, ())
                        self.assertEqual(model.neck._scdown_initialization_slots, ())
                        model.init_weights()
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        p3, p2p3 = build('scdown_p3'), build('scdown_p2p3')
        for name, value in p3.backbone.stage2[0].state_dict().items():
            torch.testing.assert_close(value, p2p3.backbone.stage2[0].state_dict()[name], atol=0, rtol=0)

    def test_restoring_only_extras_exactly_recovers_predictions_losses_and_gradients(self):
        torch.manual_seed(31)
        images = torch.randn(2, 3, 128, 128)
        targets = dict(bboxes_labels=torch.tensor(
            [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
            dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                control, model = build(), build(method)
                for path in spec['extra']:
                    parent, _, leaf = path.rpartition('.')
                    model.get_submodule(parent)._modules[leaf] = deepcopy(control.get_submodule(path))
                control.eval(); model.eval()
                with torch.no_grad():
                    for aa, bb in zip(control._forward(images), model._forward(images)):
                        for a, b in zip(aa, bb):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                control.train(); model.train()
                original, actual = control.loss(images, targets), model.loss(images, targets)
                self.assertEqual(set(actual), {'loss_cls', 'loss_bbox'})
                for key in original:
                    torch.testing.assert_close(original[key], actual[key], atol=0, rtol=0, msg=key)
                sum(original.values()).backward(); sum(actual.values()).backward()
                parameters = dict(control.named_parameters())
                for name, p in model.named_parameters():
                    self.assertEqual(p.grad is None, parameters[name].grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_grouped_forward_and_gradient_match_block_diagonal_dense_reference(self):
        for width in (48, 96, 192):
            for training in (False, True):
                with self.subTest(width=width, training=training):
                    torch.manual_seed(71)
                    dense = ConvModule(width, width, 3, padding=1,
                        norm_cfg=dict(type='BN', eps=.001, momentum=.03),
                        act_cfg=dict(type='SiLU', inplace=True)).train(training)
                    grouped = grouped_conv_from(dense).train(training)
                    half = width // 2
                    with torch.no_grad():
                        dense.conv.weight.zero_()
                        dense.conv.weight[:half, :half].copy_(grouped.conv.weight[:half])
                        dense.conv.weight[half:, half:].copy_(grouped.conv.weight[half:])
                    x = torch.randn(2, width, 9, 13, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    actual, expected = grouped(x), dense(reference_x)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                    seed = torch.randn_like(actual) / actual.numel() ** .5
                    gradients = torch.autograd.grad(actual, (x, *grouped.parameters()), seed)
                    reference = torch.autograd.grad(expected, (reference_x, *dense.parameters()), seed)
                    dense_weight_grad = reference[1]
                    aligned = (reference[0], torch.cat((dense_weight_grad[:half, :half],
                                                       dense_weight_grad[half:, half:])), *reference[2:])
                    for a, b in zip(gradients, aligned):
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                    for a, b in zip(grouped.norm.buffers(), dense.norm.buffers()):
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                    with torch.no_grad():
                        isolated = torch.randn(1, width, 7, 11)
                        isolated[:, half:] = 0
                        self.assertEqual(float(grouped.conv(isolated)[:, half:].abs().max()), 0)

    def test_new_scdown_entries_keep_odd_grid_centers_and_no_final_activation(self):
        for ci, co in ((24, 48), (48, 96)):
            layer = SCDown(ci, co).eval()
            matrix = torch.randn(co, ci)
            with torch.no_grad():
                layer.cv1.conv.weight.copy_(matrix[:, :, None, None])
                layer.cv2.conv.weight.zero_(); layer.cv2.conv.weight[:, 0, 1, 1] = -1
                for part in (layer.cv1, layer.cv2):
                    part.norm.running_mean.zero_(); part.norm.running_var.fill_(1)
                    part.norm.weight.fill_((1 + part.norm.eps) ** .5); part.norm.bias.zero_()
            for h, w in ((13, 19), (14, 20)):
                x = torch.randn(2, ci, h, w)
                expected = -torch.nn.functional.silu(torch.einsum('oc,bchw->bohw', matrix, x))[:, :, ::2, ::2]
                torch.testing.assert_close(layer(x), expected, atol=1e-5, rtol=1e-5)
                self.assertFalse(layer.cv2.with_activation)


if __name__ == '__main__':
    unittest.main()
