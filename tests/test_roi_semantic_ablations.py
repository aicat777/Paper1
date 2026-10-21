"""Mechanism controls: coordinates, exact matching, selection, gradients, state."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import random
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmengine.config import Config
from mmengine.model import ExponentialMovingAverage
from mmengine.structures import InstanceData
from mmyolo.models.task_modules.assigners.batch_dsl_assigner import BatchDynamicSoftLabelAssigner
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_center.sampling import CenterInstanceSemanticLoss
from custom_models.instance_semantic_matched.selection import MatchedGTSelection
from custom_models.instance_semantic_random.selection import RandomGTSelection
from custom_models.roi_supervision_common.assigner import ROISemanticIndexedAssigner
from custom_models.roi_supervision_common.supervision import selected_roi_loss, valid_rows_and_matches

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('instance_semantic_center', 'instance_semantic_matched', 'instance_semantic_random')
register_all_modules(); torch.set_num_threads(2); logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method, weight=.1):
    cfg = config(method)
    cfg.model.bbox_head.semantic_loss_weight = weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def data():
    torch.manual_seed(13)
    images = torch.randn(2, 3, 128, 128)
    # Same-class, same-box GTs retain different identities; rows interleave images.
    rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
                         [1, 2, 24, 26, 116, 118], [0, 1, 10, 12, 30, 36],
                         [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    return images, dict(bboxes_labels=rows, img_metas=[dict(img_shape=(128, 128))] * 2)


class ROISemanticAblationTests(unittest.TestCase):
    def test_center_ramps_fractional_substride_batch_and_levels(self):
        module = CenterInstanceSemanticLoss(1, 3)
        features = []
        for stride in (8, 16, 32):
            y, x = torch.meshgrid(torch.arange(640 // stride), torch.arange(640 // stride), indexing='ij')
            ramp = (x + 2 * y).float() * stride
            features.append(torch.stack((ramp, ramp + 1000))[:, None])
        rois = torch.tensor([[1, 5.25, 9.75, 6.25, 10.75], [0, 8, 16, 136, 144],
                             [1, 32, 32, 288, 288], [0, 48, 32, 56, 36]])
        expected = (rois[:, 1] + rois[:, 3]) / 2 + rois[:, 2] + rois[:, 4] + 1000 * rois[:, 0]
        torch.testing.assert_close(module.pool_instances(features, rois)[:, 0], expected)

    def test_center_against_independent_roi_align_one_sample_forward_and_gradient(self):
        for height, width in ((9, 13), (1, 1)):
            torch.manual_seed(9)
            features = [torch.randn(2, 3, height, width, requires_grad=True)]
            reference = [features[0].detach().clone().requires_grad_()]
            rois = torch.tensor([[1, .1, .2, .2, .3],
                                 [0, 0, 0, width * 8., height * 8.],
                                 [1, width * 8 - .2, height * 8 - .2, width * 8, height * 8]])
            module = CenterInstanceSemanticLoss(3, 2, strides=(8,))
            actual = module.pool_instances(features, rois)
            scaled = rois.clone(); scaled[:, 1:] = scaled[:, 1:] / 8 + .5
            expected = roi_align(reference[0], scaled, 1, sampling_ratio=1, aligned=True)[:, :, 0, 0]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            seed = torch.randn_like(actual)
            a = torch.autograd.grad(actual, features, seed)[0]
            b = torch.autograd.grad(expected, reference, seed)[0]
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_center_changes_only_sampling_on_spatially_nonuniform_features(self):
        feature = torch.zeros(1, 1, 80, 80)
        feature[0, 0, 40, 40] = 9
        rois = torch.tensor([[0., 256., 256., 384., 384.]])
        center = CenterInstanceSemanticLoss(1, 2, strides=(8,)).pool_instances([feature], rois)
        region = InstanceSemanticLoss(1, 2, strides=(8,)).pool_instances([feature], rois)
        self.assertEqual(center.item(), 9)
        self.assertLess(region.item(), center.item())

    def test_index_assigner_preserves_official_outputs_and_actual_indices(self):
        torch.manual_seed(17)
        xy = torch.rand(2, 29, 2) * 40
        boxes = torch.cat((xy, xy + torch.rand_like(xy) * 25 + 2), -1)
        scores = torch.randn(2, 29, 3)
        priors = torch.cat((torch.rand(29, 2) * 64, torch.full((29, 2), 8.)), 1)
        gt = torch.tensor([[[2, 2, 32, 34], [2, 2, 32, 34], [20, 20, 60, 60]],
                           [[4, 4, 38, 44], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float32)
        labels = torch.tensor([[[1], [1], [2]], [[0], [0], [0]]])
        flags = (gt.sum(-1, keepdim=True) > 0).float()
        original, indexed = BatchDynamicSoftLabelAssigner(3), ROISemanticIndexedAssigner(3)
        args = boxes, scores, priors, labels, gt, flags
        expected, actual = original(*args), indexed(*args)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        b, p = (actual['assigned_gt_indices'] >= 0).nonzero(as_tuple=True)
        g = actual['assigned_gt_indices'][b, p]
        torch.testing.assert_close(actual['assigned_bboxes'][b, p], gt[b, g], atol=0, rtol=0)
        self.assertIs(indexed.dynamic_k_matching.__func__, original.dynamic_k_matching.__func__)

    def test_mapping_interleaved_rows_same_boxes_and_degenerate_slots(self):
        rows = torch.tensor([[1, 2, 8, 8, 16, 16], [0, 1, 2, 2, 9, 9],
                             [0, 0, 4, 4, 4, 8], [1, 2, 8, 8, 16, 16],
                             [0, 1, 2, 2, 9, 9]], dtype=torch.float32)
        padded = gt_instances_preprocess(rows, 2)
        # Only image0 slot2 and image1 slot0 match, not their duplicate boxes.
        assigned = torch.tensor([[2, 2, -1], [0, -1, -1]])
        valid, matched = valid_rows_and_matches(rows, padded, assigned)
        torch.testing.assert_close(valid, rows[[0, 1, 3, 4]], atol=0, rtol=0)
        self.assertEqual(matched.tolist(), [True, False, False, True])
        gt = [InstanceData(bboxes=rows[rows[:, 0] == b, 2:], labels=rows[rows[:, 0] == b, 1].long()) for b in range(2)]
        valid_list, matched_list = valid_rows_and_matches(gt, padded, assigned)
        self.assertEqual(matched_list.tolist(), [False, True, True, False])
        self.assertEqual(len(valid_list), 4)

    def test_subset_denominator_is_all_gt_and_all_selected_equals_original(self):
        torch.manual_seed(6)
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (32, 16, 8)]
        rows = torch.tensor([[0, 1, 8, 8, 24, 24], [1, 2, 32, 32, 200, 200],
                             [1, 0, 48, 48, 60, 60]], dtype=torch.float32)
        complete = selected_roi_loss(module, features, rows, torch.ones(3, dtype=torch.bool))
        expected = module(features, rows)
        torch.testing.assert_close(complete, expected, atol=0, rtol=0)
        keep = torch.tensor([False, True, False])
        partial = selected_roi_loss(module, features, rows, keep)
        torch.testing.assert_close(partial, module(features, rows[keep]) / 3, atol=0, rtol=0)
        parameters = tuple(features) + tuple(module.parameters())
        first = torch.autograd.grad(complete, parameters, retain_graph=True, allow_unused=True)
        second = torch.autograd.grad(expected, parameters, retain_graph=True, allow_unused=True)
        for a, b in zip(first, second):
            if a is not None:
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_empty_selection_and_empty_gt_remain_finite_and_connected(self):
        module = InstanceSemanticLoss(4, 3)
        for rows in (torch.empty(0, 6), torch.tensor([[0., 1., 4., 4., 8., 8.]])):
            features = [torch.randn(1, 4, s, s, requires_grad=True) for s in (16, 8, 4)]
            loss = selected_roi_loss(module, features, rows, torch.zeros(len(rows), dtype=torch.bool))
            grads = torch.autograd.grad(loss, (*features, *module.parameters()))
            self.assertEqual(float(loss), 0)
            self.assertTrue(all(torch.isfinite(g).all() and not g.any() for g in grads))

    def test_random_count_uniform_coverage_and_no_global_rng_consumption(self):
        selector = RandomGTSelection(42)
        torch.manual_seed(15); np.random.seed(15); random.seed(15)
        torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        matched = torch.tensor([True, False, True, False, False, False])
        counts = torch.zeros(6)
        for _ in range(100):
            selected = selector(matched)
            self.assertEqual(int(selected.sum()), 2)
            counts += selected
        self.assertTrue((counts > 15).all() and (counts < 55).all(), counts)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(MatchedGTSelection()(matched), matched))

    def test_random_save_restore_empty_full_and_ema_buffer(self):
        selector = RandomGTSelection(42)
        state = selector.rng_state.clone()
        for flags in (torch.empty(0, dtype=torch.bool), torch.zeros(5, dtype=torch.bool), torch.ones(5, dtype=torch.bool)):
            torch.testing.assert_close(selector(flags), flags, atol=0, rtol=0)
            self.assertTrue(torch.equal(selector.rng_state, state))
        flags = torch.tensor([True, False, False, True, False])
        selector(flags)
        saved = deepcopy(selector.state_dict())
        expected = selector(flags)
        selector.load_state_dict(saved)
        torch.testing.assert_close(selector(flags), expected, atol=0, rtol=0)
        ema = ExponentialMovingAverage(selector, update_buffers=True)
        selector(flags); ema.update_parameters(selector)
        torch.testing.assert_close(ema.module.rng_state, selector.rng_state, atol=0, rtol=0)

    def test_models_initialization_inference_and_training_protocol_unchanged(self):
        original = build('instance_semantic').eval()
        images, batch = data()
        with torch.no_grad():
            reference = original._forward(images)
        for method in METHODS:
            model = build(method).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), 4880648)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=method + ':' + name)
            with torch.no_grad():
                for actual, expected in zip(model._forward(images), reference):
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            a, b = config('instance_semantic'), config(method)
            for key in ('train_dataloader', 'val_dataloader', 'val_evaluator', 'optim_wrapper', 'custom_hooks',
                        'param_scheduler', 'default_hooks', 'randomness', 'train_cfg'):
                self.assertEqual(a[key], b[key], (method, key))

    def test_main_losses_exact_and_aux_gradient_scope_single_head_pass(self):
        images, batch = data()
        original = build('instance_semantic').train()
        ref = original.loss(images, batch)
        for method in METHODS:
            model = build(method).train()
            calls = []
            handle = model.bbox_head.assigner.register_forward_hook(lambda *args: calls.append(1))
            losses = model.loss(images, batch)
            handle.remove()
            self.assertEqual(calls, [1])
            for key in ('loss_cls', 'loss_bbox'):
                torch.testing.assert_close(losses[key], ref[key], atol=0, rtol=0, msg=method + key)
            losses['loss_instance_semantic'].backward()
            head = model.bbox_head.head_module
            self.assertGreater(model.bbox_head.instance_semantic.classifier.weight.grad.abs().sum().item(), 0)
            self.assertGreater(head.cls_convs[0][0].conv.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.backbone.stem[0].conv.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in head.rtm_cls.parameters()))
            self.assertTrue(all(p.grad is None for p in head.rtm_reg.parameters()))
            self.assertTrue(all(p.grad is None for p in head.reg_convs.parameters()))
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(int(layer.num_batches_tracked), 1, method + name)

    def test_all_matched_subset_models_equal_original_roi_loss_and_gradients(self):
        images, batch = data()
        # Exactly one GT/image, both with a positive in this fixed initialization.
        batch['bboxes_labels'] = batch['bboxes_labels'][[0, 1]]
        reference = build('instance_semantic').eval()
        first = reference.loss(images, batch)
        first['loss_instance_semantic'].backward()
        for method in METHODS[1:]:
            model = build(method).eval()
            second = model.loss(images, batch)
            self.assertEqual(float(second['semantic_matched_gt']), float(second['semantic_all_gt']))
            torch.testing.assert_close(second['loss_instance_semantic'], first['loss_instance_semantic'], atol=0, rtol=0)
            second['loss_instance_semantic'].backward()
            ref = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.grad is None, ref[name].grad is None)
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, ref[name].grad, atol=0, rtol=0, msg=name)

    def test_zero_weight_gradients_match_original_and_sgd_step(self):
        images, batch = data()
        original = build('instance_semantic', 0).train()
        losses = original.loss(images, batch)
        sum(v for k, v in losses.items() if k.startswith('loss')).backward()
        for method in METHODS:
            model = build(method, 0).train()
            values = model.loss(images, batch)
            sum(v for k, v in values.items() if k.startswith('loss')).backward()
            ref = dict(original.named_parameters())
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.grad is None, ref[name].grad is None, method + name)
                if parameter.grad is not None:
                    # Center sampling has a different zero-valued autograd
                    # branch, which can reorder shared-convolution reductions.
                    # FP32 maximum difference is ~2.6e-9 in this fixture.
                    torch.testing.assert_close(parameter.grad, ref[name].grad,
                                               atol=1e-8, rtol=1e-5, msg=method + name)
            expected = deepcopy(original)
            # deepcopy of a Module does not retain .grad; copy gradients explicitly.
            for p, old in zip(expected.parameters(), original.parameters()):
                p.grad = None if old.grad is None else old.grad.clone()
            # SGD directly checks the effect of these near-identical
            # gradients. AdamW integration is checked by actual GPU updates;
            # its first g/(abs(g)+eps) step is sensitive to tiny differences
            # close to zero and is not a bitwise pooling-parity criterion.
            torch.optim.SGD(expected.parameters(), lr=.0005625).step()
            torch.optim.SGD(model.parameters(), lr=.0005625).step()
            expected_parameters = dict(expected.named_parameters())
            for name, p in model.named_parameters():
                torch.testing.assert_close(p, expected_parameters[name], atol=1e-8, rtol=1e-6)

    def test_model_serialization_random_continuation_and_inference_no_rng_changes(self):
        model = build('instance_semantic_random').eval()
        flags = torch.tensor([True, False, False, True, False])
        model.bbox_head.selection(flags)
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = build('instance_semantic_random').eval()
        restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        torch.testing.assert_close(model.bbox_head.selection(flags), restored.bbox_head.selection(flags), atol=0, rtol=0)
        state = model.bbox_head.selection.rng_state.clone()
        with torch.no_grad():
            model._forward(data()[0])
        torch.testing.assert_close(state, model.bbox_head.selection.rng_state, atol=0, rtol=0)

    def test_empty_gt_all_models_and_list_interface(self):
        images, batch = data()
        rows = batch['bboxes_labels']
        gt = [InstanceData(bboxes=rows[rows[:, 0] == b, 2:], labels=rows[rows[:, 0] == b, 1].long()) for b in range(2)]
        for method in METHODS:
            model = build(method).eval()
            with torch.no_grad():
                cls, reg = model.bbox_head(model.extract_feat(images))
                first = model.bbox_head.loss_by_feat(cls, reg, rows, batch['img_metas'])
                second = model.bbox_head.loss_by_feat(cls, reg, gt, batch['img_metas'])
                for key in first:
                    torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
                empty = model.loss(images, dict(bboxes_labels=torch.empty(0, 6), img_metas=batch['img_metas']))
            self.assertEqual(float(empty['loss_instance_semantic']), 0)
            self.assertTrue(all(torch.isfinite(v) for v in empty.values()))


if __name__ == '__main__':
    unittest.main()
