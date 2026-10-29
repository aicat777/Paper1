"""Verify that classifier coverage and feature-gradient coverage are distinct."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_split.supervision import MatchedFeatureGradientGate, split_roi_loss
from custom_models.roi_supervision_common.supervision import selected_roi_loss

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build(method, weight=.1):
    cfg = Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))
    cfg.model.bbox_head.semantic_loss_weight = weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def batch():
    torch.manual_seed(13)
    images = torch.randn(2, 3, 128, 128)
    rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
                         [1, 2, 24, 26, 116, 118], [0, 1, 10, 12, 30, 36],
                         [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    return images, dict(bboxes_labels=rows, img_metas=[dict(img_shape=(128, 128))] * 2)


def gradients(loss, parameters):
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


class SplitROISemanticTests(unittest.TestCase):
    def test_gate_closed_form_classifier_and_descriptor_gradients(self):
        torch.manual_seed(9)
        z = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        linear = torch.nn.Linear(4, 3).double()
        labels = torch.tensor([0, 1, 2])
        matched = torch.tensor([True, False, True])
        routed = MatchedFeatureGradientGate()(z, matched)
        torch.testing.assert_close(routed, z, atol=0, rtol=0)
        loss = .1 * F.cross_entropy(linear(routed), labels)
        gz, gw, gb = torch.autograd.grad(loss, (z, linear.weight, linear.bias))
        error = (linear(z).softmax(-1) - F.one_hot(labels, 3)) * (.1 / 3)
        torch.testing.assert_close(gw, error.T @ z)
        torch.testing.assert_close(gb, error.sum(0))
        torch.testing.assert_close(gz, (error @ linear.weight) * matched[:, None])
        self.assertTrue(gz[matched].abs().sum() > 0)
        self.assertTrue(gw.abs().sum() > 0)
        self.assertEqual(float(gz[~matched].abs().sum()), 0)

    def test_roi_forward_classifier_gradient_and_matched_feature_reference(self):
        torch.manual_seed(6)
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (80, 40, 20)]
        # Duplicate overlapping ROIs, fractional sub-stride ROI, all three levels.
        rows = torch.tensor([[1, 1, 8.25, 8.5, 9.25, 9.5], [0, 1, 16, 16, 48, 48],
                             [0, 2, 16, 16, 48, 48], [1, 0, 32, 32, 160, 160],
                             [0, 2, 48, 48, 368, 368]], dtype=torch.float32)
        matched = torch.tensor([False, True, False, True, False])
        split = split_roi_loss(module, MatchedFeatureGradientGate(), features, rows, matched)
        original = module(features, rows)
        selected = selected_roi_loss(module, features, rows, matched)
        torch.testing.assert_close(split, original, atol=0, rtol=0)
        parameters = (*features, *module.parameters())
        actual, full, only_matched = [gradients(loss, parameters) for loss in (split, original, selected)]
        for i, (a, f, m) in enumerate(zip(actual, full, only_matched)):
            expected = f if i >= len(features) else m
            if expected is None:
                self.assertTrue(a is None or not a.any())
            else:
                torch.testing.assert_close(a, expected, atol=2e-7, rtol=2e-6)
        # Removing an unmatched contribution does not zero a shared ROI's pixels.
        self.assertGreater(float(actual[0][0, :, 2:6, 2:6].abs().sum()), 0)
        self.assertGreater(float((actual[-2] - only_matched[-2]).abs().sum()), 0)

    def test_no_matched_gt_still_trains_classifier_without_feature_gradient(self):
        torch.manual_seed(4)
        module = InstanceSemanticLoss(4, 3, strides=(8,))
        feature = torch.randn(1, 4, 16, 16, requires_grad=True)
        rows = torch.tensor([[0., 1., 4., 4., 20., 20.], [0., 2., 32., 32., 52., 52.]])
        loss = split_roi_loss(module, MatchedFeatureGradientGate(), [feature], rows, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(loss, module([feature], rows), atol=0, rtol=0)
        self.assertGreater(float(loss), 0)
        optimizer = torch.optim.SGD(module.parameters(), lr=.1)
        before = module.classifier.weight.detach().clone()
        loss.backward()
        self.assertEqual(float(feature.grad.abs().sum()), 0)
        self.assertGreater(float(module.classifier.weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertGreater(float((module.classifier.weight - before).abs().sum()), 0)

    def test_all_matched_equals_original_loss_and_all_gradients(self):
        torch.manual_seed(8)
        module = InstanceSemanticLoss(4, 3, strides=(8,))
        feature = torch.randn(1, 4, 16, 16, requires_grad=True)
        rows = torch.tensor([[0., 1., 4., 4., 20., 20.], [0., 2., 32., 32., 52., 52.]])
        expected = module([feature], rows)
        actual = split_roi_loss(module, MatchedFeatureGradientGate(), [feature], rows, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        params = (feature, *module.parameters())
        for a, b in zip(gradients(actual, params), gradients(expected, params)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_empty_gt_connected_zero_and_bad_masks_rejected(self):
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(1, 4, s, s, requires_grad=True) for s in (16, 8, 4)]
        loss = split_roi_loss(module, MatchedFeatureGradientGate(), features, torch.empty(0, 6), torch.empty(0, dtype=torch.bool))
        self.assertEqual(float(loss), 0)
        for grad in torch.autograd.grad(loss, (*features, *module.parameters())):
            self.assertTrue(torch.isfinite(grad).all() and not grad.any())
        for mask in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                MatchedFeatureGradientGate()(torch.randn(2, 4), mask)

    def test_model_initialization_protocol_inference_and_rng_unchanged(self):
        original = build('instance_semantic').eval()
        original_rng = torch.get_rng_state().clone()
        model = build('instance_semantic_split').eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4880648)
        self.assertEqual(set(original.state_dict()), set(model.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        images, _ = batch()
        calls = []
        handle = model.bbox_head.gradient_gate.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            for a, b in zip(original._forward(images), model._forward(images)):
                for first, second in zip(a, b):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
        handle.remove()
        self.assertEqual(calls, [])
        configs = [Config.fromfile(str(ROOT / 'configs' / m / 'rtmdet_tiny_dior_640.py'))
                   for m in ['instance_semantic', 'instance_semantic_split']]
        for key in ('train_dataloader', 'val_dataloader', 'val_evaluator', 'train_cfg',
                    'optim_wrapper', 'custom_hooks', 'param_scheduler', 'default_hooks', 'randomness'):
            self.assertEqual(configs[0][key], configs[1][key], key)

    def test_model_single_assignment_loss_and_two_gradient_paths(self):
        images, data = batch()
        original, model, matched = [build(m).train() for m in
            ['instance_semantic', 'instance_semantic_split', 'instance_semantic_matched']]
        calls = []
        handle = model.bbox_head.assigner.register_forward_hook(lambda *args: calls.append(1))
        expected, actual, feature_reference = [m.loss(images, data) for m in (original, model, matched)]
        handle.remove()
        self.assertEqual(calls, [1])
        self.assertLess(float(actual['semantic_feature_gt']), float(actual['semantic_classifier_gt']))
        self.assertEqual(float(actual['semantic_classifier_gt']), 4)
        self.assertEqual(float(actual['semantic_feature_gt']), float(actual['semantic_matched_gt']))
        for key in ('loss_cls', 'loss_bbox', 'loss_instance_semantic'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        for loss in (expected, actual, feature_reference):
            loss['loss_instance_semantic'].backward()
        ref, mat = dict(original.named_parameters()), dict(matched.named_parameters())
        for name, parameter in model.named_parameters():
            comparison = ref[name] if name.startswith('bbox_head.instance_semantic.') else mat[name]
            if comparison.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, comparison.grad, atol=2e-7, rtol=3e-5, msg=name)
        head = model.bbox_head.head_module
        self.assertTrue(all(p.grad is None for p in (*head.rtm_cls.parameters(), *head.rtm_reg.parameters(), *head.reg_convs.parameters())))
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_tensor_and_list_gt_interfaces_with_degenerate_rows_and_empty_batch(self):
        images, data = batch()
        for empty in (False, True):
            rows = data['bboxes_labels'] if not empty else data['bboxes_labels'][:0]
            packed = dict(bboxes_labels=rows, img_metas=data['img_metas'])
            samples = []
            for image in range(2):
                row = rows[rows[:, 0] == image]
                sample = DetDataSample(metainfo=dict(img_shape=(128, 128)))
                sample.gt_instances = InstanceData(bboxes=row[:, 2:], labels=row[:, 1].long())
                samples.append(sample)
            a, b = build('instance_semantic_split').eval(), build('instance_semantic_split').eval()
            first, second = a.loss(images, packed), b.loss(images, samples)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=2e-7, rtol=2e-6, msg=key)

    def test_zero_auxiliary_weight_preserves_total_gradients_and_sgd_update(self):
        images, data = batch()
        a, b = build('instance_semantic', 0).train(), build('instance_semantic_split', 0).train()
        for model in (a, b):
            losses = model.loss(images, data)
            sum(v for k, v in losses.items() if 'loss' in k).backward()
        ref = dict(a.named_parameters())
        for name, p in b.named_parameters():
            if ref[name].grad is None:
                self.assertIsNone(p.grad)
            else:
                torch.testing.assert_close(p.grad, ref[name].grad, atol=2e-7, rtol=2e-5, msg=name)
        for model in (a, b):
            torch.optim.SGD(model.parameters(), lr=.001).step()
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=2e-7, rtol=2e-6, msg=name)


if __name__ == '__main__':
    unittest.main()
