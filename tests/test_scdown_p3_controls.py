"""P3-only controls: exact initialization, numerical references and restoration."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.scdown_p3_dense import P3DenseSpatial
from custom_models.scdown_p3_noact import P3NoActivation
from scripts.scdown_p3_controls_probe import SPECS, P3, CONTROL_SITES, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
register_all_modules(); torch.set_num_threads(2); logging.disable(logging.CRITICAL)


def build(method='scdown_p3', initialize=True):
    cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


def data():
    torch.manual_seed(31)
    images = torch.randn(2, 3, 128, 128)
    targets = dict(bboxes_labels=torch.tensor(
        [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
        dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)
    return images, targets


def plain_torch_reference(module):
    pw, spatial = module.cv1, module.cv2
    first = torch.nn.Conv2d(pw.conv.in_channels, pw.conv.out_channels, 1, bias=False)
    first_bn = torch.nn.BatchNorm2d(pw.conv.out_channels, eps=pw.bn.eps, momentum=pw.bn.momentum)
    second = torch.nn.Conv2d(spatial.conv.in_channels, spatial.conv.out_channels, 3,
        stride=2, padding=1, groups=spatial.conv.groups, bias=False)
    second_bn = torch.nn.BatchNorm2d(spatial.conv.out_channels, eps=spatial.bn.eps, momentum=spatial.bn.momentum)
    first.load_state_dict(pw.conv.state_dict()); first_bn.load_state_dict(pw.bn.state_dict())
    second.load_state_dict(spatial.conv.state_dict()); second_bn.load_state_dict(spatial.bn.state_dict())
    parts = [first, first_bn]
    if pw.with_activation:
        parts.append(torch.nn.SiLU())
    parts.extend((second, second_bn))
    return torch.nn.Sequential(*parts)


class P3ControlTests(unittest.TestCase):
    def test_only_intended_operator_changes_with_correct_grid_head_and_counts(self):
        for method, spec in SPECS.items():
            with self.subTest(method=method):
                model, reference = build(method).eval(), build().eval()
                layer = model.get_submodule(P3)
                self.assertIsInstance(layer, P3DenseSpatial if method == 'scdown_p3_dense' else P3NoActivation)
                self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), CONTROL_SITES)
                self.assertEqual(layer.cv1.conv.weight.shape, (96, 48, 1, 1))
                self.assertEqual(layer.cv1.with_activation, spec['intermediate_silu'])
                self.assertEqual(layer.cv2.conv.groups, spec['groups'])
                self.assertEqual(layer.cv2.conv.stride, (2, 2))
                self.assertEqual(layer.cv2.conv.padding, (1, 1))
                self.assertEqual(layer.cv2.conv.dilation, (1, 1))
                self.assertFalse(layer.cv2.with_activation)
                self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in layer.modules()), 2)
                for name, module in reference.named_modules():
                    if name in ('', 'backbone', P3) or name.startswith(P3 + '.'):
                        continue
                    self.assertIs(type(module), type(model.get_submodule(name)), name)
                self.assertEqual(sum(p.numel() for p in model.parameters()), spec['parameters'])
                self.assertEqual(len(watched_parameters(model, method)), 24)
                trace, handles = [], []
                for path in (P3 + '.cv1', P3 + '.cv2'):
                    handles.append(model.get_submodule(path).register_forward_hook(
                        lambda mod, inputs, output, path=path:
                            trace.append((path, list(inputs[0].shape), list(output.shape)))))
                with torch.no_grad():
                    cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
                for handle in handles:
                    handle.remove()
                self.assertEqual(trace, [(P3 + '.cv1', [1, 48, 160, 160], [1, 96, 160, 160]),
                                         (P3 + '.cv2', [1, 96, 160, 160], [1, 96, 80, 80])])
                self.assertEqual([list(v.shape) for v in cls], [[1, 20, s, s] for s in (80, 40, 20)])
                self.assertEqual([list(v.shape) for v in reg], [[1, 4, s, s] for s in (80, 40, 20)])
                priors = model.bbox_head.prior_generator.grid_priors(
                    [v.shape[-2:] for v in cls], device='cpu', with_stride=True)
                self.assertEqual(sum(len(p) for p in priors), 8400)
                head = model.bbox_head.head_module
                for towers in (head.cls_convs, head.reg_convs):
                    for depth in range(2):
                        for scale in (1, 2):
                            self.assertIs(towers[0][depth].conv, towers[scale][depth].conv)
                            self.assertIsNot(towers[0][depth].bn, towers[scale][depth].bn)

    def test_initialization_rng_exact_including_p3_pw_bn_and_other_scdowns(self):
        for method in SPECS:
            for initialize in (False, True):
                with self.subTest(method=method, initialized=initialize):
                    reference = build(initialize=initialize)
                    expected_rng = torch.get_rng_state().clone()
                    model = build(method, initialize)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
                    exception = P3 + '.cv2.conv.weight' if method == 'scdown_p3_dense' else None
                    for name, value in reference.state_dict().items():
                        if name != exception:
                            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
                    if initialize:
                        self.assertEqual(model.backbone._scdown_initialization_slots, ())
                        self.assertEqual(model.neck._scdown_initialization_slots, ())
                        before = {n: v.clone() for n, v in model.state_dict().items()}
                        model.init_weights()
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                        for name, value in before.items():
                            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)

    def test_restore_only_changed_part_recovers_predictions_losses_gradients_exactly(self):
        images, targets = data()
        for method in SPECS:
            with self.subTest(method=method):
                reference, model = build(), build(method)
                part = 'cv2' if method == 'scdown_p3_dense' else 'cv1'
                setattr(model.get_submodule(P3), part, deepcopy(getattr(reference.get_submodule(P3), part)))
                reference.eval(); model.eval()
                with torch.no_grad():
                    for aa, bb in zip(reference._forward(images), model._forward(images)):
                        for a, b in zip(aa, bb):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                reference.train(); model.train()
                expected, actual = reference.loss(images, targets), model.loss(images, targets)
                self.assertEqual(set(actual), {'loss_cls', 'loss_bbox'})
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)
                sum(expected.values()).backward(); sum(actual.values()).backward()
                parameters = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.grad is None, parameters[name].grad is None, name)
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_dense_diagonal_kernel_contains_scdown_function_and_gradients(self):
        for training in (False, True):
            for h, w in ((13, 19), (14, 20)):
                with self.subTest(training=training, hw=(h, w)):
                    torch.manual_seed(19)
                    reference = SCDown(48, 96).train(training)
                    dense = P3DenseSpatial(reference).train(training)
                    indices = torch.arange(96)
                    with torch.no_grad():
                        dense.cv2.conv.weight.zero_()
                        dense.cv2.conv.weight[indices, indices] = reference.cv2.conv.weight[:, 0]
                    x = torch.randn(2, 48, h, w, requires_grad=True)
                    reference_x = x.detach().clone().requires_grad_()
                    actual, expected = dense(x), reference(reference_x)
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                    seed = torch.randn_like(actual) / actual.numel() ** .5
                    gradients = torch.autograd.grad(actual, (x, *dense.parameters()), seed)
                    expected_gradients = torch.autograd.grad(expected, (reference_x, *reference.parameters()), seed)
                    names = ('input', *(name for name, _ in dense.named_parameters()))
                    for name, a, b in zip(names, gradients, expected_gradients):
                        if name == 'cv2.conv.weight':
                            a = a[indices, indices].unsqueeze(1)
                        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=name)

    def test_forward_backward_and_bn_states_match_independent_torch_reference(self):
        for cls in (P3DenseSpatial, P3NoActivation):
            for training in (False, True):
                for h, w in ((13, 19), (14, 20)):
                    with self.subTest(operator=cls.__name__, training=training, hw=(h, w)):
                        torch.manual_seed(23)
                        source = SCDown(48, 96)
                        with torch.no_grad():
                            for part in (source.cv1, source.cv2):
                                part.bn.weight.uniform_(-1.2, 1.2); part.bn.bias.normal_(std=.1)
                                part.bn.running_mean.normal_(std=.2); part.bn.running_var.uniform_(.4, 1.6)
                        module = cls(source).train(training)
                        reference = plain_torch_reference(module).train(training)
                        x = torch.randn(2, 48, h, w, requires_grad=True)
                        reference_x = x.detach().clone().requires_grad_()
                        actual, expected = module(x), reference(reference_x)
                        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                        seed = torch.randn_like(actual) / actual.numel() ** .5
                        gradients = torch.autograd.grad(actual, (x, *module.parameters()), seed)
                        expected_gradients = torch.autograd.grad(expected, (reference_x, *reference.parameters()), seed)
                        for a, b in zip(gradients, expected_gradients):
                            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
                        for a, b in zip(module.buffers(), reference.buffers()):
                            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)

    def test_real_detection_gradients_updates_and_one_bn_execution(self):
        images, targets = data()
        for method in SPECS:
            with self.subTest(method=method):
                model = build(method).train()
                losses = model.loss(images, targets)
                self.assertEqual(set(losses), {'loss_cls', 'loss_bbox'})
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                sum(losses.values()).backward()
                watched = watched_parameters(model, method)
                self.assertEqual(len(watched), 24)
                initial = {n: p.detach().clone() for n, p in watched.items()}
                for name, p in watched.items():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(float(p.grad.abs().max()), 0, name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
                optimizer.step()
                for name, p in watched.items():
                    self.assertGreater(float((p - initial[name]).abs().max()), 0, name)
                for name, layer in model.named_modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_checkpoint_roundtrip_preserves_prediction_and_shared_head(self):
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

    def test_original_modules_not_modified_and_incorrect_sources_rejected(self):
        source = SCDown(48, 96)
        before = {n: v.clone() for n, v in source.state_dict().items()}
        dense, noact = P3DenseSpatial(source), P3NoActivation(source)
        self.assertTrue(source.cv1.with_activation)
        self.assertIsInstance(source.cv1.activate, torch.nn.SiLU)
        self.assertFalse(noact.cv1.with_activation)
        self.assertFalse(hasattr(noact.cv1, 'activate'))
        self.assertFalse(noact.cv2.with_activation)
        for module in (dense, noact):
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)
            self.assertNotEqual(source.cv1.conv.weight.data_ptr(), module.cv1.conv.weight.data_ptr())
        for cls in (P3DenseSpatial, P3NoActivation):
            with self.assertRaises(TypeError):
                cls(torch.nn.Conv2d(48, 96, 3, stride=2, padding=1))
        invalid = SCDown(48, 96)
        invalid.cv1.with_activation = False
        for cls in (P3DenseSpatial, P3NoActivation):
            with self.assertRaises(ValueError):
                cls(invalid)


if __name__ == '__main__':
    unittest.main()
