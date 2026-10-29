"""Independent objectives, gradient routes and real GSCE integrations."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_gradscale import GradientScaledInstanceSemanticLoss
from custom_models.instance_semantic_private import PrivateInstanceSemanticLoss
from custom_models.instance_semantic_levels import AllLevelInstanceSemanticLoss
from custom_models.instance_semantic_contrast import (
    ContrastiveInstanceSemanticLoss, supervised_contrastive_sum)
from tests.test_instance_semantic_gradscale import data

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'scdown_p3_g4_semantic_gradscale'
METHODS = tuple('scdown_p3_g4_semantic_' + k for k in ('private', 'levels', 'contrast'))
COUNTS = (3142856, 3105704, 3127432)
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method, alpha=.25, contrast_weight=None):
    cfg = config(method)
    cfg.model.bbox_head.semantic_feature_gradient_scale = alpha
    if contrast_weight is not None:
        cfg.model.bbox_head.semantic_contrast_weight = contrast_weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def total(losses):
    return sum(v for k, v in losses.items() if 'loss' in k)


def naive_contrast_sum(features, labels, temperature=.2):
    # A per-anchor scalar definition; deliberately does not reuse masks,
    # matrix multiplication, chunk code, or the custom backward.
    result = features.sum() * 0
    for i in range(len(features)):
        candidates = [j for j in range(len(features)) if j != i]
        positives = [j for j in candidates if labels[i] == labels[j]]
        if not positives:
            continue
        logits = torch.stack([(features[i] * features[j]).sum() / temperature
                              for j in candidates])
        positive_logits = torch.stack([(features[i] * features[j]).sum() / temperature
                                       for j in positives])
        result = result + torch.logsumexp(logits, 0) - positive_logits.mean()
    return result


def roi_fixture(channels=4, device='cpu', dtype=torch.float32):
    torch.manual_seed(17)
    features = [torch.randn(2, channels, s, s, device=device, dtype=dtype,
                            requires_grad=True) for s in (32, 16, 8)]
    rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
        [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
        [0, 3, 8, 8, 248, 248], [1, 0, 4.25, 5.5, 5.25, 6.5],
        [0, 0, 0, 0, 0, 0]], device=device, dtype=torch.float32)
    return features, rows


class ContrastiveFormulaTests(unittest.TestCase):
    def test_streaming_value_and_both_sides_of_gradient_against_scalar_reference(self):
        torch.manual_seed(31)
        z = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 2, 0, 2, 3, 4, 1])
        features = F.normalize(z, dim=1)
        expected = naive_contrast_sum(features, labels)
        wanted = torch.autograd.grad(expected, z, retain_graph=True)[0]
        for chunk in (1, 2, 4, 128):
            actual = supervised_contrastive_sum(features, labels, chunk_size=chunk)
            gradient = torch.autograd.grad(actual, z, retain_graph=True)[0]
            torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
            torch.testing.assert_close(gradient, wanted, atol=3e-14, rtol=3e-13)
        # Single-class-occurrence rows have no anchor loss but DO participate
        # as negatives; omitting their candidate derivatives would be wrong.
        self.assertGreater(float(wanted[6].abs().sum()), 0)
        self.assertGreater(float(wanted[7].abs().sum()), 0)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: supervised_contrastive_sum(x, labels, chunk_size=3),
            (features.detach().requires_grad_(),), eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_empty_singleton_no_positive_and_identical_features(self):
        for labels in ([], [0], [0, 1, 2], [0, 0], [0, 0, 0, 0]):
            x = torch.ones(len(labels), 4, requires_grad=True)
            y = torch.tensor(labels, dtype=torch.long)
            value = supervised_contrastive_sum(F.normalize(x, dim=1), y, chunk_size=2)
            expected = naive_contrast_sum(F.normalize(x, dim=1), y)
            torch.testing.assert_close(value, expected)
            gradient = torch.autograd.grad(value, x)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            if len(labels) < 2 or len(set(labels)) == len(labels):
                self.assertEqual(float(value), 0.)
                self.assertFalse(gradient.any())

    def test_permutation_invariance_and_external_loss_scale(self):
        torch.manual_seed(4)
        x = F.normalize(torch.randn(8, 7), dim=1).requires_grad_()
        labels = torch.tensor([1, 0, 0, 2, 2, 3, 3, 1])
        permutation = torch.randperm(8)
        expected = naive_contrast_sum(x, labels)
        wanted = torch.autograd.grad(expected, x)[0]
        for multiplier in (1., 32., 65536.):
            actual = supervised_contrastive_sum(x[permutation], labels[permutation], chunk_size=3)
            gradient = torch.autograd.grad(actual * multiplier, x)[0]
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-6)
            torch.testing.assert_close(gradient / multiplier, wanted, atol=4e-6, rtol=5e-6)


class ROIVariantLossTests(unittest.TestCase):
    def test_all_level_objective_and_gradients_equal_independent_per_level_ce(self):
        features, rows = roi_fixture()
        base = InstanceSemanticLoss(4, 4)
        model = AllLevelInstanceSemanticLoss(base, .25)
        actual = model(features, rows)
        valid = rows[(rows[:, 4:6] > rows[:, 2:4]).all(1)]
        losses = []
        for feature, stride in zip(features, (8, 16, 32)):
            boxes = torch.cat((valid[:, :1], valid[:, 2:] / stride + .5), 1)
            z = roi_align(feature, boxes, 3, sampling_ratio=2, aligned=True).mean((-2, -1))
            losses.append(.1 * F.cross_entropy(base.classifier(z), valid[:, 1].long()))
        expected = sum(losses) / 3
        actual_g = torch.autograd.grad(actual, (*features, *base.parameters()), retain_graph=True)
        expected_g = torch.autograd.grad(expected, (*features, *base.parameters()))
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
        for i, (a, b) in enumerate(zip(actual_g, expected_g)):
            torch.testing.assert_close(a, b * (.25 if i < 3 else 1), atol=2e-8, rtol=3e-6)

    def test_one_small_gt_reaches_every_level_without_tripling_weight(self):
        torch.manual_seed(3)
        base = InstanceSemanticLoss(4, 3)
        model = AllLevelInstanceSemanticLoss(base)
        # Identical descriptors at all levels reduce exactly to ONE GT CE.
        vector = torch.tensor([.2, -.1, .7, .3])
        features = [vector[None, :, None, None].expand(1, 4, s, s).clone().requires_grad_()
                    for s in (32, 16, 8)]
        rows = torch.tensor([[0, 1, 32, 32, 48, 48]], dtype=torch.float32)
        value = model(features, rows)
        wanted = .1 * F.cross_entropy(base.classifier(vector[None]), torch.tensor([1]))
        torch.testing.assert_close(value, wanted, atol=2e-8, rtol=1e-6)
        gradients = torch.autograd.grad(value, features)
        self.assertTrue(all(g.abs().sum() > 0 for g in gradients))

    def test_private_and_contrast_scale_only_shared_features(self):
        for cls in (PrivateInstanceSemanticLoss, ContrastiveInstanceSemanticLoss):
            features, rows = roi_fixture()
            reference = InstanceSemanticLoss(4, 4)
            state = torch.get_rng_state().clone()
            full, scaled = cls(deepcopy(reference), 1.), cls(deepcopy(reference), .25)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(set(full.state_dict()), set(scaled.state_dict()))
            for name, p in full.state_dict().items():
                torch.testing.assert_close(p, scaled.state_dict()[name], atol=0, rtol=0)
            a, b = full(features, rows), scaled(features, rows)
            if isinstance(a, dict):
                a, b = total(a), total(b)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            ag = torch.autograd.grad(a, (*features, *full.parameters()))
            bg = torch.autograd.grad(b, (*features, *scaled.parameters()))
            for i, (g, h) in enumerate(zip(ag, bg)):
                torch.testing.assert_close(h, g * (.25 if i < 3 else 1), atol=2e-8, rtol=3e-5)

    def test_contrast_keeps_original_ce_and_normalizes_by_all_gt(self):
        features, rows = roi_fixture()
        original = InstanceSemanticLoss(4, 4)
        model = ContrastiveInstanceSemanticLoss(original, pair_chunk_size=2)
        actual = model(features, rows)
        reference = GradientScaledInstanceSemanticLoss(original)
        torch.testing.assert_close(actual['loss_instance_semantic'], reference(features, rows),
                                   atol=0, rtol=0)
        valid = rows[(rows[:, 4:6] > rows[:, 2:4]).all(1)]
        rois = torch.cat((valid[:, :1], valid[:, 2:]), 1)
        vectors = model.pool_instances(features, rois)
        z = F.normalize(model.projector(vectors), dim=1)
        expected = .1 * naive_contrast_sum(z, valid[:, 1].long()) / len(valid)
        torch.testing.assert_close(actual['loss_instance_contrast'], expected, atol=1e-7, rtol=2e-6)
        self.assertEqual(float(actual['semantic_all_gt']), 6.)
        self.assertEqual(float(actual['semantic_contrast_anchors']), 3.)

    def test_distributed_normalizer_and_empty_gt_are_finite_connected(self):
        for cls, module_path in ((PrivateInstanceSemanticLoss, 'custom_models.instance_semantic.roi'),
                                (AllLevelInstanceSemanticLoss, 'custom_models.instance_semantic_levels.roi'),
                                (ContrastiveInstanceSemanticLoss, 'custom_models.instance_semantic_contrast.roi')):
            features, rows = roi_fixture()
            model = cls(InstanceSemanticLoss(4, 4))
            regular = model(features, rows)
            with patch(module_path + '.reduce_mean', side_effect=lambda x: x * 2):
                distributed = model(features, rows)
            if isinstance(regular, dict):
                regular, distributed = total(regular), total(distributed)
            torch.testing.assert_close(distributed, regular / 2, atol=1e-8, rtol=3e-6)
            empty = model(features, rows[:0])
            if isinstance(empty, dict):
                empty = total(empty)
            self.assertEqual(float(empty), 0.)
            gs = torch.autograd.grad(empty, (*features, *model.parameters()))
            self.assertTrue(all(g is not None and not g.any() for g in gs))


class GSCEVariantIntegrationTests(unittest.TestCase):
    def test_protocol_initialization_inference_and_head_sharing(self):
        ref_cfg = config(REFERENCE)
        reference = build(REFERENCE).eval()
        reference_rng = torch.get_rng_state().clone()
        images, _ = data()
        for method, count in zip(METHODS, COUNTS):
            cfg = config(method)
            for key in ('train_dataloader', 'val_dataloader', 'test_dataloader', 'train_cfg',
                        'val_cfg', 'test_cfg', 'val_evaluator', 'test_evaluator', 'custom_hooks',
                        'default_hooks', 'param_scheduler', 'optim_wrapper', 'randomness', 'env_cfg'):
                self.assertEqual(cfg[key], ref_cfg[key], (method, key))
            self.assertIsNone(cfg.load_from); self.assertFalse(cfg.resume)
            model = build(method).eval()
            self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            head = model.bbox_head.head_module
            for level in (1, 2):
                for i in range(len(head.cls_convs[0])):
                    self.assertIs(head.cls_convs[0][i].conv, head.cls_convs[level][i].conv)
                    self.assertIs(head.reg_convs[0][i].conv, head.reg_convs[level][i].conv)
                    self.assertIsNot(head.cls_convs[0][i].bn, head.cls_convs[level][i].bn)
            with torch.no_grad():
                for a, b in zip(model._forward(images), reference._forward(images)):
                    for x, y in zip(a, b):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_main_losses_and_derivatives_unchanged_and_one_forward_per_step(self):
        images, samples = data()
        reference = build(REFERENCE).train()
        ref_losses = reference.loss(images, samples)
        names, parameters = zip(*[(n, p) for n, p in reference.named_parameters()
                                  if not n.startswith('bbox_head.instance_semantic.')])
        gradients = torch.autograd.grad(ref_losses['loss_cls'] + ref_losses['loss_bbox'],
                                        parameters, allow_unused=True)
        for method in METHODS:
            model = build(method).train()
            calls = dict(backbone=0, neck=0, tower=0, assignment=0, roi=0)
            observed = {}
            def hook(key):
                def capture(module, inputs, outputs):
                    calls[key] += 1
                    if key in ('neck', 'tower'):
                        observed[key] = outputs
                return capture
            handles = [module.register_forward_hook(hook(key)) for key, module in
                       [('backbone', model.backbone), ('neck', model.neck),
                        ('tower', model.bbox_head.head_module),
                        ('assignment', model.bbox_head.assigner), ('roi', model.bbox_head.instance_semantic)]]
            handles.append(model.bbox_head.instance_semantic.register_forward_pre_hook(
                lambda m, args: observed.update(roi_inputs=args[0])))
            losses = model.loss(images, samples)
            for handle in handles: handle.remove()
            self.assertEqual(set(calls.values()), {1})
            expected_inputs = observed['neck'] if method.endswith('private') else observed['tower'][2]
            self.assertTrue(all(a is b for a, b in zip(observed['roi_inputs'], expected_inputs)))
            for key in ('loss_cls', 'loss_bbox'):
                torch.testing.assert_close(losses[key], ref_losses[key], atol=0, rtol=0)
            params = dict(model.named_parameters())
            actual = torch.autograd.grad(losses['loss_cls'] + losses['loss_bbox'],
                                         [params[n] for n in names], allow_unused=True, retain_graph=True)
            for name, a, b in zip(names, actual, gradients):
                if b is None:
                    self.assertIsNone(a)
                else:
                    torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_actual_auxiliary_gradient_boundaries(self):
        images, samples = data()
        for method in METHODS:
            model = build(method).train()
            losses = model.loss(images, samples)
            auxiliary = sum(v for k, v in losses.items() if k.startswith('loss_instance'))
            auxiliary.backward()
            for prefix in ('backbone.', 'neck.', 'bbox_head.instance_semantic.'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if n.startswith(prefix)))
            for n, p in model.bbox_head.head_module.named_parameters():
                expects_cls = n.startswith('cls_convs.') and not method.endswith('private')
                if expects_cls:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(p.grad.abs().sum() > 0, n)
                else:
                    self.assertTrue(p.grad is None or not p.grad.any(), n)

    def test_zero_contrast_weight_recovers_original_ce_and_full_detector_update(self):
        images, samples = data()
        reference = build(REFERENCE).train()
        model = build(METHODS[2], contrast_weight=0).train()
        a, b = reference.loss(images, samples), model.loss(images, samples)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        self.assertEqual(float(b['loss_instance_contrast']), 0.)
        for candidate, loss in ((reference, a), (model, b)):
            total(loss).backward()
            torch.optim.AdamW(candidate.parameters(), lr=.0005625).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)

    def test_list_tensor_gt_equivalence_empty_gt_and_state_roundtrip(self):
        for method in METHODS:
            a, b = build(method).train(), build(method).train()
            x, tensor_gt = data()
            _, list_gt = data(as_list=True)
            first, second = a.loss(x, tensor_gt), b.loss(x, list_gt)
            for key in first:
                # The list interface groups ROIs by image, so auxiliary sum
                # reductions can differ by FP32 rounding after a permutation.
                tolerance = dict(atol=2e-7, rtol=1e-6) if key.startswith('loss_instance') \
                    else dict(atol=0, rtol=0)
                torch.testing.assert_close(first[key], second[key], msg=(method, key), **tolerance)
            buffer = io.BytesIO()
            torch.save(a.state_dict(), buffer); buffer.seek(0)
            b.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
            _, empty = data(empty=True)
            losses = b.loss(x, empty)
            self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
            self.assertEqual(float(losses['loss_instance_semantic']), 0.)
            total(losses).backward()


if __name__ == '__main__':
    unittest.main()
