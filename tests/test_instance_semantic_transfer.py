"""Selection, exact GT identity, per-instance normalization, and gradient routing."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_transfer import (
    RegionToPointTransfer, original_roi_readout, valid_rows_and_lookup)

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build(method='instance_semantic_transfer', weight=.1, min_iou=.5):
    cfg = Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))
    if method == 'instance_semantic_transfer':
        cfg.model.bbox_head.update(transfer_loss_weight=weight, transfer_min_iou=min_iou)
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def model_batch():
    torch.manual_seed(13)
    images = torch.randn(2, 3, 128, 128)
    rows = torch.tensor([[1, 1, 8.25, 8.5, 9.25, 9.5], [0, 1, 10, 12, 30, 36],
                         [1, 2, 24, 26, 116, 118], [0, 1, 10, 12, 30, 36],
                         [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    return images, dict(bboxes_labels=rows, img_metas=[dict(img_shape=(128, 128))] * 2)


def fixture(device='cpu', dtype=torch.float32):
    torch.manual_seed(13)
    # Raw slots: image0 [row1, degenerate, row3, row5], image1 [row0,row4].
    raw = torch.tensor([[1, 2, 0, 0, 16, 16], [0, 0, 8, 8, 24, 24],
                        [0, 1, 8, 8, 8, 24], [0, 0, 8, 8, 24, 24],
                        [1, 1, 0, 0, 32, 32], [0, 2, 8, 8, 24, 24]],
                       dtype=torch.float32, device=device)
    padded = gt_instances_preprocess(raw, 2)
    rows, lookup = valid_rows_and_lookup(raw, padded)
    features = [torch.randn(2, 3, h, w, device=device, dtype=dtype, requires_grad=True)
                for h, w in [(2, 4), (1, 2), (1, 1)]]
    scores = [torch.zeros(2, 3, *x.shape[-2:], device=device, dtype=dtype) for x in features]
    teacher = F.one_hot(rows[:, 1].long(), 3).float() * 4
    teacher[3] = torch.tensor([4., 0., 0.], device=device)  # valid row3 is label1, wrong teacher
    teacher.requires_grad_()
    slots = torch.full((2, 11), -1, dtype=torch.long, device=device)
    quality = torch.zeros(2, 11, device=device)
    assignments = [(0, 0, 0, .7), (0, 1, 0, .8), (0, 3, 0, .5), (0, 8, 0, .6),
                   (0, 10, 2, .8), (1, 0, 0, .6), (1, 10, 0, .9),
                   (1, 1, 1, .9), (0, 2, 3, .49), (0, 4, 0, .9), (0, 5, 1, .9)]
    for batch, point, slot, iou in assignments:
        slots[batch, point], quality[batch, point] = slot, iou
        level, local = (0, point) if point < 8 else ((1, point - 8) if point < 10 else (2, point - 10))
        height, width = scores[level].shape[-2:]
        label = int(padded[batch, slot, 0])
        scores[level][batch, (label + 1) % 3, local // width, local % width] = 4
    scores[0][0, :, 1, 0] = torch.tensor([5., 0., 0.], device=device, dtype=dtype)  # correct detector
    scores = [x.requires_grad_() for x in scores]
    classifier = torch.nn.Linear(3, 3, device=device)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[.7, -.2, .1], [.3, .6, -.1], [-.2, .1, .8]], device=device))
        classifier.bias.copy_(torch.tensor([.1, -.2, .05], device=device))
    return features, scores, dict(assigned_gt_indices=slots, assign_metrics=quality), rows, lookup, teacher, classifier


def reference_points(features, scores, assigned, rows, lookup, teacher, classifier, temperature=1.):
    grouped, chosen = {}, []
    offset = 0
    for level, (feature, score) in enumerate(zip(features, scores)):
        h, w = feature.shape[-2:]
        for b in range(feature.shape[0]):
            for y in range(h):
                for x in range(w):
                    point = offset + y * w + x
                    slot = int(assigned['assigned_gt_indices'][b, point])
                    if slot < 0:
                        continue
                    row = int(lookup[b, slot])
                    if row < 0 or float(assigned['assign_metrics'][b, point]) < .5:
                        continue
                    label = int(rows[row, 1])
                    if int(teacher[row].argmax()) != label or int(score[b, :, y, x].argmax()) == label:
                        continue
                    z = feature[b, :, y, x].float()
                    p = (F.linear(z, classifier.weight.detach(), classifier.bias.detach()) / temperature).softmax(-1)
                    q = (teacher[row].detach() / temperature).softmax(-1)
                    value = (q * (q.log() - p.log())).sum() * temperature ** 2
                    grouped.setdefault(row, []).append(value)
                    chosen.append((level, b, y, x, row, p.detach(), q))
        offset += h * w
    loss = .1 * torch.stack([torch.stack(v).mean() for v in grouped.values()]).mean()
    return loss, chosen, grouped


class TransferROITests(unittest.TestCase):
    def test_gt_slot_identity_duplicates_degenerate_interleaved_and_list(self):
        _, _, _, rows, lookup, _, _ = fixture()
        self.assertEqual(lookup.tolist(), [[1, -1, 2, 4], [0, 3, -1, -1]])
        self.assertTrue(torch.equal(rows[1], rows[2]))
        images, batch = model_batch()
        raw = batch['bboxes_labels']
        padded = gt_instances_preprocess(raw, 2)
        actual, mapping = valid_rows_and_lookup(raw, padded)
        self.assertEqual(mapping.tolist(), [[1, 3, -1], [0, 2, -1]])
        samples = [InstanceData(bboxes=raw[raw[:, 0] == i, 2:], labels=raw[raw[:, 0] == i, 1].long())
                   for i in range(2)]
        list_rows, list_lookup = valid_rows_and_lookup(samples, padded)
        for b in range(2):
            for slot in range(padded.shape[1]):
                a, c = int(mapping[b, slot]), int(list_lookup[b, slot])
                self.assertEqual(a < 0, c < 0)
                if a >= 0:
                    torch.testing.assert_close(actual[a], list_rows[c], atol=0, rtol=0)

    def test_selection_closed_form_kl_per_gt_normalization_and_feature_gradient(self):
        for temperature in (1., 2.):
            values = fixture()
            features, scores, assigned, rows, lookup, teacher, classifier = values
            module = RegionToPointTransfer(temperature=temperature)
            actual, stats = module(*values)
            expected, chosen, grouped = reference_points(*values, temperature=temperature)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
            self.assertEqual(len(chosen), 7)
            self.assertEqual(sorted(len(v) for v in grouped.values()), [1, 2, 4])
            self.assertEqual(float(stats['transfer_selected_gt']), 3)
            self.assertEqual(float(stats['transfer_selected_points']), 7)
            self.assertTrue(all(not x.requires_grad for x in stats.values()))
            parameters = (*features, *scores, teacher, *classifier.parameters())
            grads = torch.autograd.grad(actual, parameters, allow_unused=True)
            closed = [torch.zeros_like(x, dtype=torch.float32) for x in features]
            for level, b, y, x, row, p, q in chosen:
                closed[level][b, :, y, x] += (.1 * temperature / len(grouped) / len(grouped[row])) * ((p - q) @ classifier.weight.detach())
            for actual_grad, wanted in zip(grads[:3], closed):
                torch.testing.assert_close(actual_grad, wanted, atol=2e-7, rtol=2e-6)
            self.assertTrue(all(g is None for g in grads[3:]))

    def test_original_roi_ce_and_every_gradient_match_with_all_valid_gt(self):
        torch.manual_seed(6)
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (80, 40, 20)]
        rows = torch.tensor([[1, 1, 8.25, 8.5, 9.25, 9.5], [0, 1, 16, 16, 48, 48],
                             [0, 2, 16, 16, 48, 48], [1, 0, 32, 32, 160, 160],
                             [0, 2, 48, 48, 368, 368], [0, 1, 4, 4, 4, 9]], dtype=torch.float32)
        original = module(features, rows)
        valid = rows[(rows[:, 4:6] > rows[:, 2:4]).all(-1)]
        actual, logits = original_roi_readout(module, features, valid)
        self.assertEqual(len(logits), 5)
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
        params = (*features, *module.parameters())
        ga = torch.autograd.grad(actual, params, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(original, params, allow_unused=True)
        for a, b in zip(ga, gb):
            if b is None:
                self.assertIsNone(a)
            else:
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_no_selection_empty_gt_and_zero_weight_connected_zero(self):
        for case in ('no_match', 'wrong_teacher', 'empty', 'zero_weight'):
            values = list(fixture())
            if case == 'no_match':
                values[2]['assigned_gt_indices'].fill_(-1)
            elif case == 'wrong_teacher':
                with torch.no_grad():
                    values[5].zero_()
                    for row, label in enumerate(values[3][:, 1].long()):
                        values[5][row, (int(label) + 1) % 3] = 4
            elif case == 'empty':
                values[2]['assigned_gt_indices'].fill_(-1)
                values[3], values[4], values[5] = values[3][:0], values[4][:, :0], values[5][:0]
            loss, stats = RegionToPointTransfer(loss_weight=0 if case == 'zero_weight' else .1)(*values)
            self.assertEqual(float(loss), 0)
            self.assertEqual(float(stats['transfer_selected_points']), 0)
            for grad in torch.autograd.grad(loss, values[0]):
                self.assertTrue(torch.isfinite(grad).all() and not grad.any())

    def test_capacity_only_stress_activates_full_grid_without_mutating_actual_assignment(self):
        from scripts.instance_semantic_transfer_probe import stress_selection
        values = fixture()
        original_slots = values[2]['assigned_gt_indices'].clone()
        original_quality = values[2]['assign_metrics'].clone()
        stressed = stress_selection(None, values)
        loss, stats = RegionToPointTransfer()(*stressed)
        self.assertEqual(float(stats['transfer_selected_points']), 22)
        self.assertEqual(float(stats['transfer_selected_gt']), 5)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(values[2]['assigned_gt_indices'], original_slots, atol=0, rtol=0)
        torch.testing.assert_close(values[2]['assign_metrics'], original_quality, atol=0, rtol=0)

    def test_initialization_rng_protocol_shared_convs_and_inference_exact(self):
        original = build('instance_semantic').eval()
        rng = torch.get_rng_state().clone()
        model = build().eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4880648)
        self.assertEqual(set(model.state_dict()), set(original.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
        head = model.bbox_head.head_module
        for tower in (head.cls_convs, head.reg_convs):
            for depth in range(2):
                self.assertIs(tower[0][depth].conv.weight, tower[2][depth].conv.weight)
                self.assertIsNot(tower[0][depth].bn, tower[2][depth].bn)
        calls = []
        handle = model.bbox_head.local_transfer.register_forward_hook(lambda *a: calls.append(1))
        images, _ = model_batch()
        with torch.no_grad():
            for a, b in zip(original._forward(images), model._forward(images)):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
        handle.remove()
        self.assertEqual(calls, [])
        cfg, ref = [Config.fromfile(str(ROOT / 'configs' / m / 'rtmdet_tiny_dior_640.py'))
                    for m in ('instance_semantic_transfer', 'instance_semantic')]
        for key in ('train_dataloader', 'val_dataloader', 'val_evaluator', 'train_cfg',
                    'optim_wrapper', 'custom_hooks', 'param_scheduler', 'default_hooks', 'randomness'):
            self.assertEqual(cfg[key], ref[key], key)

    def test_main_and_roi_losses_unchanged_one_head_assignment_and_bn_update(self):
        images, data = model_batch()
        original, model = build('instance_semantic').train(), build().train()
        calls = dict(head=0, assignment=0, roi=0)

        def counter(key):
            def hook(*args):
                calls[key] += 1
            return hook

        handles = [model.bbox_head.head_module.register_forward_hook(counter('head')),
                   model.bbox_head.assigner.register_forward_hook(counter('assignment')),
                   model.bbox_head.instance_semantic.classifier.register_forward_hook(counter('roi'))]
        expected, actual = original.loss(images, data), model.loss(images, data)
        for h in handles:
            h.remove()
        self.assertEqual(calls, dict(head=1, assignment=1, roi=1))
        for key in ('loss_cls', 'loss_bbox', 'loss_instance_semantic'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertEqual(float(actual['semantic_all_gt']), 4)
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_active_transfer_reaches_shared_features_excludes_predictors_and_roi_classifier(self):
        # Test-only zero threshold activates the route on an untrained model;
        # formal .5 threshold is covered by the independent selection fixture.
        model = build(min_iou=0).train()
        with torch.no_grad():
            model.bbox_head.instance_semantic.classifier.bias[1] = 5
            for layer in model.bbox_head.head_module.rtm_cls:
                layer.bias.zero_()
                layer.bias[0] = 5
        images, data = model_batch()
        data['bboxes_labels'][:, 1] = 1
        losses = model.loss(images, data)
        self.assertGreater(float(losses['transfer_selected_points']), 0)
        self.assertGreater(float(losses['loss_instance_transfer']), 0)
        losses['loss_instance_transfer'].backward()
        for prefix in ('backbone.', 'neck.', 'bbox_head.head_module.cls_convs.'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in model.named_parameters() if n.startswith(prefix)), prefix)
        for prefix in ('bbox_head.instance_semantic.', 'bbox_head.head_module.rtm_cls.',
                       'bbox_head.head_module.rtm_reg.', 'bbox_head.head_module.reg_convs.'):
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n.startswith(prefix)), prefix)

    def test_zero_transfer_weight_total_gradients_and_sgd_update_match_original(self):
        a, b = build('instance_semantic').train(), build(weight=0).train()
        images, data = model_batch()
        for model in (a, b):
            losses = model.loss(images, data)
            sum(v for k, v in losses.items() if 'loss' in k).backward()
        reference = dict(a.named_parameters())
        for name, p in b.named_parameters():
            if reference[name].grad is None:
                self.assertIsNone(p.grad, name)
            else:
                torch.testing.assert_close(p.grad, reference[name].grad, atol=2e-7, rtol=2e-5, msg=name)
        for model in (a, b):
            torch.optim.SGD(model.parameters(), lr=.001).step()
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], atol=2e-7, rtol=2e-6, msg=name)

    def test_tensor_list_and_empty_gt_interfaces(self):
        images, data = model_batch()
        for empty in (False, True):
            rows = data['bboxes_labels'][:0] if empty else data['bboxes_labels']
            samples = []
            for image in range(2):
                raw = rows[rows[:, 0] == image]
                sample = DetDataSample(metainfo=dict(img_shape=(128, 128)))
                sample.gt_instances = InstanceData(bboxes=raw[:, 2:], labels=raw[:, 1].long())
                samples.append(sample)
            a, b = build().eval(), build().eval()
            first = a.loss(images, dict(bboxes_labels=rows, img_metas=data['img_metas']))
            second = b.loss(images, samples)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=2e-7, rtol=2e-6, msg=key)


if __name__ == '__main__':
    unittest.main()
