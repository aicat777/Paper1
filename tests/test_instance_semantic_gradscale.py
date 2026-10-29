"""Independent derivatives, original-ROI boundaries and three real models."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_gradscale import (
    GradientScaledInstanceSemanticLoss, scale_feature_gradient)

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ('instance_semantic', 'scdown_p3_semantic', 'scdown_p3_g4_semantic')
METHODS = tuple(name + '_gradscale' for name in REFERENCES)
PLAIN = ('plain', 'scdown_p3', 'scdown_p3_g4')
COUNTS = (4880648, 3821096, 3105704)
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method, alpha=.25):
    cfg = config('instance_semantic' if method == 'plain' else method)
    if method == 'plain':
        cfg.model.bbox_head = deepcopy(Config.fromfile(
            str(ROOT / 'configs/rtmdet_tiny_dior_300e.py')).model.bbox_head)
    elif method in METHODS:
        cfg.model.bbox_head.semantic_feature_gradient_scale = alpha
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def data(empty=False, as_list=False):
    torch.manual_seed(17)
    images = torch.randn(2, 3, 256, 256)
    # Interleaved images, duplicate GTs, all levels, fractional sub-stride ROI,
    # and one degenerate box filtered only by the original auxiliary task.
    rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
                         [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
                         [0, 3, 8, 8, 248, 248], [1, 4, 4.25, 5.5, 5.25, 6.5],
                         [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    if empty:
        rows = rows[:0]
    if as_list:
        samples = []
        for b in range(2):
            valid = rows[(rows[:, 0] == b) & (rows[:, 4:6] > rows[:, 2:4]).all(-1)]
            sample = DetDataSample(metainfo=dict(img_shape=(256, 256)))
            sample.gt_instances = InstanceData(bboxes=valid[:, 2:], labels=valid[:, 1].long())
            samples.append(sample)
    else:
        samples = dict(bboxes_labels=rows, img_metas=[dict(img_shape=(256, 256))] * 2)
    return images, samples


def gradients(loss, params, retain_graph=False):
    return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)


class ROIGradientScaleTests(unittest.TestCase):
    def test_independent_closed_form_and_identity(self):
        torch.manual_seed(7)
        for alpha in (0., .25, 1.):
            z = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
            w = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
            b = torch.randn(3, dtype=torch.float64, requires_grad=True)
            labels = torch.tensor([0, 1, 2, 0, 1, 0])
            routed = scale_feature_gradient(z, alpha)
            torch.testing.assert_close(routed, z, atol=0, rtol=0)
            loss = .1 * F.cross_entropy(F.linear(routed, w, b), labels)
            actual = gradients(loss, (z, w, b))
            with torch.no_grad():
                error = F.linear(z, w, b).softmax(-1)
                error[torch.arange(len(labels)), labels] -= 1
                error *= .1 / len(labels)
                wanted = (alpha * (error @ w), error.T @ z, error.sum(0))
            for a, e in zip(actual, wanted):
                torch.testing.assert_close(a, e, atol=2e-16, rtol=2e-14)

    def test_original_classifier_reused_and_roi_feature_gradients_scaled(self):
        torch.manual_seed(7)
        reference = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (32, 16, 8)]
        rows = torch.tensor([[0, 1, 1.25, 2.5, 2.25, 3.5], [1, 2, 0, 0, 240, 240],
                             [1, 0, 0, 0, 140, 140], [0, 1, 1.25, 2.5, 2.25, 3.5],
                             [0, 2, 4, 4, 4, 9]], dtype=torch.float32)
        rng = torch.get_rng_state().clone()
        for alpha in (0., .25, 1.):
            module = GradientScaledInstanceSemanticLoss(reference, alpha)
            self.assertIs(module.classifier, reference.classifier)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(set(module.state_dict()), set(reference.state_dict()))
            actual, wanted = module(features, rows), reference(features, rows)
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
            params = (*features, *reference.parameters())
            ag, eg = gradients(actual, params), gradients(wanted, params)
            for i, (a, e) in enumerate(zip(ag, eg)):
                factor = alpha if i < len(features) else 1
                torch.testing.assert_close(a, factor * e, atol=2e-8, rtol=3e-6)
            self.assertTrue(all(e.abs().sum() > 0 for e in eg[:3]))
        with patch('custom_models.instance_semantic.roi.reduce_mean', side_effect=lambda x: x * 2):
            # The inherited distributed denominator is preserved, not locally
            # replaced by a per-image or matched-GT mean.
            torch.testing.assert_close(module(features, rows), reference(features, rows), atol=0, rtol=0)

    def test_invalid_scales_and_empty_gt_keep_connected_zero(self):
        reference = InstanceSemanticLoss(4, 3)
        for alpha in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                GradientScaledInstanceSemanticLoss(reference, alpha)
        for alpha in (0., .25, 1.):
            module = GradientScaledInstanceSemanticLoss(reference, alpha)
            features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (4, 2, 1)]
            loss = module(features, torch.empty(0, 6))
            self.assertEqual(float(loss), 0)
            gs = gradients(loss, (*features, *module.parameters()))
            self.assertTrue(all(g is not None and not g.any() for g in gs))

    def test_three_configs_preserve_protocol_initialization_and_sharing(self):
        keys = ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                'val_evaluator', 'test_dataloader', 'test_evaluator', 'test_cfg',
                'train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'custom_hooks',
                'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness', 'env_cfg')
        for method, ref, count in zip(METHODS, REFERENCES, COUNTS):
            a, b = config(method), config(ref)
            for key in keys:
                self.assertEqual(a[key], b[key], (method, key))
            expected = deepcopy(b.model)
            expected.bbox_head.update(type='GradientScaledSemanticRTMDetHead',
                                      semantic_feature_gradient_scale=.25)
            self.assertEqual(a.model, expected)
            self.assertIsNone(a.load_from); self.assertFalse(a.resume)
            self.assertIsNone(a.model.backbone.init_cfg)
            self.assertEqual(a.train_dataloader.batch_size, 36)
            reference = build(ref).eval()
            rng = torch.get_rng_state().clone()
            model = build(method).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)
            head = model.bbox_head.head_module
            for level in (1, 2):
                for i in range(2):
                    self.assertIs(head.cls_convs[0][i].conv, head.cls_convs[level][i].conv)
                    self.assertIs(head.reg_convs[0][i].conv, head.reg_convs[level][i].conv)
                    self.assertIsNot(head.cls_convs[0][i].bn, head.cls_convs[level][i].bn)
            images, _ = data()
            with torch.no_grad():
                for first, second in zip(model._forward(images), reference._forward(images)):
                    for x, y in zip(first, second):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_total_gradients_preserve_main_and_only_scale_auxiliary_path(self):
        images, samples = data()
        for method, ref in zip(METHODS, REFERENCES):
            reference, model = build(ref).train(), build(method).train()
            losses = reference.loss(images, samples)
            params = tuple(reference.parameters())
            main = gradients(losses['loss_cls'] + losses['loss_bbox'], params, retain_graph=True)
            auxiliary = gradients(losses['loss_instance_semantic'], params)
            calls = dict(tower=0, assigner=0, roi=0)
            def count(key):
                def hook(*args):
                    calls[key] += 1
                return hook
            handles = [model.bbox_head.head_module.register_forward_hook(count('tower')),
                       model.bbox_head.assigner.register_forward_hook(count('assigner')),
                       model.bbox_head.instance_semantic.register_forward_hook(count('roi'))]
            actual = model.loss(images, samples)
            for hook in handles:
                hook.remove()
            self.assertEqual(calls, dict(tower=1, assigner=1, roi=1))
            for key in losses:
                torch.testing.assert_close(actual[key], losses[key], atol=0, rtol=0)
            sum(actual.values()).backward()
            for (name, p), gm, ga in zip(model.named_parameters(), main, auxiliary):
                if gm is None and ga is None:
                    self.assertIsNone(p.grad)
                    continue
                factor = 1 if name.startswith('bbox_head.instance_semantic.') else .25
                expected = (0 if gm is None else gm) + (0 if ga is None else factor * ga)
                torch.testing.assert_close(p.grad, expected, atol=2e-6, rtol=8e-5, msg=name)
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_alpha_one_preserves_original_gradients_and_adamw_update(self):
        images, samples = data()
        for method, ref in zip(METHODS, REFERENCES):
            reference, model = build(ref).train(), build(method, 1).train()
            for candidate in (reference, model):
                sum(candidate.loss(images, samples).values()).backward()
            for (name, p), q in zip(reference.named_parameters(), model.parameters()):
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(q.grad, p.grad, atol=0, rtol=0, msg=name)
            for candidate in (reference, model):
                torch.optim.AdamW(candidate.parameters(), lr=.0005625).step()
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)

    def test_alpha_zero_trains_classifier_but_preserves_plain_detector_update(self):
        images, samples = data()
        for method, plain in zip(METHODS, PLAIN):
            reference, model = build(plain).train(), build(method, 0).train()
            initial = model.bbox_head.instance_semantic.classifier.weight.detach().clone()
            for candidate in (reference, model):
                sum(candidate.loss(images, samples).values()).backward()
            actual = dict(model.named_parameters())
            for name, p in reference.named_parameters():
                if p.grad is None:
                    self.assertIsNone(actual[name].grad)
                else:
                    torch.testing.assert_close(actual[name].grad, p.grad, atol=2e-7, rtol=3e-5, msg=name)
            self.assertGreater(float(model.bbox_head.instance_semantic.classifier.weight.grad.abs().sum()), 0)
            for candidate in (reference, model):
                torch.optim.AdamW(candidate.parameters(), lr=.0005625).step()
            self.assertGreater(float((model.bbox_head.instance_semantic.classifier.weight - initial).abs().max()), 0)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=2e-7, rtol=3e-5, msg=name)

    def test_tensor_list_empty_gt_and_strict_state_restore(self):
        for method, ref in zip(METHODS, REFERENCES):
            for empty, as_list in ((False, True), (True, True), (True, False)):
                images, samples = data(empty, as_list)
                reference, model = build(ref).train(), build(method).train()
                actual, expected = model.loss(images, samples), reference.loss(images, samples)
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                sum(actual.values()).backward()
                self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()))
            fresh = build(method)
            fresh.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(fresh.bbox_head.instance_semantic.feature_gradient_scale, .25)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(fresh.state_dict()[name], value, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
