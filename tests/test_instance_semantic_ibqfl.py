"""Independent QFL formula, GT identity, gradient scales and GSCE integration."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from mmdet.models.losses import QualityFocalLoss
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_ibqfl import InstanceBalancedQFLLoss

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('instance_semantic_ibqfl', 'scdown_p3_g4_semantic_ibqfl')
REFERENCES = ('instance_semantic', 'scdown_p3_g4_semantic')
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method=METHODS[0], weight=.1):
    cfg = config(method)
    if method in METHODS:
        cfg.model.bbox_head.instance_qfl_weight = weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def fixture(device='cpu', dtype=torch.float32):
    torch.manual_seed(13)
    scores = [torch.randn(2, 3, h, w, device=device, dtype=dtype, requires_grad=True)
              for h, w in [(2, 4), (1, 2), (1, 1)]]
    valid = torch.tensor([[True, False, True, True], [True, True, False, False]], device=device)
    slots = torch.full((2, 11), -1, device=device, dtype=torch.long)
    qualities = torch.zeros(2, 11, device=device)
    labels = torch.full((2, 11), 3, device=device, dtype=torch.long)
    gt_labels = [[0, 1, 0, 2], [2, 1, 0, 0]]
    # Two image0 GTs have the same class (and may have identical boxes).
    # The image1 slot0 is distinct from image0 slot0. One GT is unmatched,
    # one assigned slot is degenerate, and one positive has true IoU zero.
    rows = [(0, 0, 0, 0.), (0, 1, 0, .3), (0, 8, 0, .7), (0, 10, 0, .9),
            (0, 2, 2, .4), (0, 9, 2, .8), (0, 3, 1, .6),
            (1, 0, 0, .2), (1, 8, 0, .8), (1, 10, 1, .6)]
    for b, p, gt, q in rows:
        slots[b, p], qualities[b, p], labels[b, p] = gt, q, gt_labels[b][gt]
    return scores, dict(assigned_gt_indices=slots, assigned_labels=labels,
                        assign_metrics=qualities), valid, QualityFocalLoss(beta=2.)


def scalar_qfl(z, label, quality, beta=2.):
    # Direct independent definition, without calling the project's QFL code.
    target = torch.zeros_like(z)
    target[label] = quality
    return (F.binary_cross_entropy_with_logits(z, target, reduction='none')
            * (target - z.sigmoid()).abs().pow(beta)).sum()


def reference(scores, assigned, valid, weight=.1, beta=2.):
    grouped = {}
    offset = 0
    for score in scores:
        h, w = score.shape[-2:]
        for b in range(score.shape[0]):
            for y in range(h):
                for x in range(w):
                    p = offset + y * w + x
                    slot = int(assigned['assigned_gt_indices'][b, p])
                    if slot < 0 or not bool(valid[b, slot]):
                        continue
                    value = scalar_qfl(score[b, :, y, x],
                                       int(assigned['assigned_labels'][b, p]),
                                       assigned['assign_metrics'][b, p], beta)
                    grouped.setdefault((b, slot), []).append(value)
        offset += h * w
    return weight * sum(torch.stack(xs).mean() for xs in grouped.values()) / valid.sum(), grouped


def data(empty=False):
    torch.manual_seed(13)
    images = torch.randn(2, 3, 128, 128)
    rows = torch.tensor([[1, 1, 8.25, 8.5, 9.25, 9.5], [0, 1, 10, 12, 30, 36],
                         [1, 2, 24, 26, 116, 118], [0, 1, 10, 12, 30, 36],
                         [0, 3, 12, 10, 12, 16]], dtype=torch.float32)
    return images, dict(bboxes_labels=rows[:0] if empty else rows,
                        img_metas=[dict(img_shape=(128, 128))] * 2)


class InstanceBalancedQFLTests(unittest.TestCase):
    def test_independent_formula_and_gradients_cross_levels_images_and_duplicate_gt(self):
        scores, assigned, valid, qfl = fixture()
        saved = {k: v.clone() for k, v in assigned.items()}
        actual, stats = InstanceBalancedQFLLoss()(scores, assigned, valid, qfl)
        wanted, groups = reference(scores, assigned, valid)
        self.assertEqual(sorted(len(x) for x in groups.values()), [1, 2, 2, 4])
        torch.testing.assert_close(actual, wanted, atol=2e-7, rtol=2e-6)
        a = torch.autograd.grad(actual, scores, retain_graph=True)
        b = torch.autograd.grad(wanted, scores)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-6)
        self.assertEqual(float(stats['ibqfl_all_gt']), 5)
        self.assertEqual(float(stats['ibqfl_matched_gt']), 4)
        self.assertEqual(float(stats['ibqfl_positive_points']), 9)
        self.assertTrue(all(not x.requires_grad for x in stats.values()))
        for k, value in saved.items():
            torch.testing.assert_close(value, assigned[k], atol=0, rtol=0)

    def test_added_gradient_is_positive_multiple_of_original_qfl_at_each_point(self):
        scores, assigned, valid, qfl = fixture()
        flat = torch.cat([x.permute(0, 2, 3, 1).reshape(2, -1, 3) for x in scores], 1)
        avg = float(assigned['assign_metrics'].sum().clamp_min(1))
        main = qfl(flat.reshape(-1, 3), (assigned['assigned_labels'].flatten(),
                    assigned['assign_metrics'].flatten()), avg_factor=avg)
        extra, _ = InstanceBalancedQFLLoss()(scores, assigned, valid, qfl)
        gm = torch.autograd.grad(main, scores, retain_graph=True)
        ge = torch.autograd.grad(extra, scores)
        all_slots = assigned['assigned_gt_indices']
        offset = 0
        for score, a, b in zip(scores, gm, ge):
            h, w = score.shape[-2:]
            scale = torch.zeros(2, h * w)
            for image in range(2):
                for point in range(h * w):
                    slot = int(all_slots[image, offset + point])
                    if slot >= 0 and bool(valid[image, slot]):
                        count = int((all_slots[image] == slot).sum())
                        scale[image, point] = .1 * avg / (int(valid.sum()) * count)
            torch.testing.assert_close(b, a * scale.reshape(2, 1, h, w), atol=2e-7, rtol=2e-6)
            offset += h * w

    def test_all_gt_denominator_includes_unmatched_gt_and_empty_ddp_rank(self):
        values = fixture()
        full, _ = InstanceBalancedQFLLoss()(*values)
        valid = values[2].clone()
        valid[0, 3] = False  # Only the unmatched GT changes.
        reduced, _ = InstanceBalancedQFLLoss()(values[0], values[1], valid, values[3])
        torch.testing.assert_close(full * 5, reduced * 4)
        with patch('custom_models.instance_semantic_ibqfl.loss.reduce_mean', return_value=torch.tensor(.5)):
            partial_rank, _ = InstanceBalancedQFLLoss()(*values)
        torch.testing.assert_close(partial_rank, full * 10)
        values[1]['assigned_gt_indices'].fill_(-1)
        values[2].zero_()
        with patch('custom_models.instance_semantic_ibqfl.loss.reduce_mean', return_value=torch.tensor(.5)) as sync:
            empty, _ = InstanceBalancedQFLLoss()(*values)
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(float(empty), 0)
        self.assertTrue(all(not x.any() for x in torch.autograd.grad(empty, values[0])))

    def test_uses_actual_quality_loss_settings_detaches_quality_and_zero_weight(self):
        scores, assigned, valid, _ = fixture()
        assigned['assign_metrics'].requires_grad_()
        qfl = QualityFocalLoss(beta=1.5, loss_weight=2.)
        actual, _ = InstanceBalancedQFLLoss()(scores, assigned, valid, qfl)
        wanted, _ = reference(scores, assigned, valid, beta=1.5)
        torch.testing.assert_close(actual, wanted * 2, atol=2e-7, rtol=2e-6)
        self.assertIsNone(torch.autograd.grad(actual, assigned['assign_metrics'], allow_unused=True)[0])
        with patch.object(qfl, 'forward', side_effect=AssertionError('Zero weight should skip extra QFL')):
            zero, _ = InstanceBalancedQFLLoss(0)(scores, assigned, valid, qfl)
        self.assertEqual(float(zero), 0)
        self.assertTrue(all(not x.any() for x in torch.autograd.grad(zero, scores)))

    def test_no_match_no_gt_and_invalid_arguments(self):
        for empty in (False, True):
            scores, assigned, valid, qfl = fixture()
            assigned['assigned_gt_indices'].fill_(-1)
            if empty:
                valid = valid[:, :0]
            loss, stats = InstanceBalancedQFLLoss()(scores, assigned, valid, qfl)
            self.assertEqual(float(loss), 0)
            self.assertEqual(float(stats['ibqfl_positive_points']), 0)
            self.assertTrue(all(torch.isfinite(x).all() and not x.any()
                                for x in torch.autograd.grad(loss, scores)))
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                InstanceBalancedQFLLoss(weight)
        values = fixture()
        with self.assertRaises(ValueError):
            InstanceBalancedQFLLoss()(values[0][:-1], values[1], values[2], values[3])

    def test_baseline_and_gsce_initialization_protocol_inference_and_sharing(self):
        for method, ref, total in zip(METHODS, REFERENCES, (4880648, 3105704)):
            a = build(ref).eval()
            rng = torch.get_rng_state().clone()
            b = build(method).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            for n, v in a.state_dict().items():
                torch.testing.assert_close(v, b.state_dict()[n], atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in b.parameters()), total)
            self.assertEqual(len(list(b.bbox_head.instance_qfl.parameters())), 0)
            cfg, before = config(method), config(ref)
            for key in ('train_dataloader', 'val_dataloader', 'val_evaluator', 'train_cfg',
                        'optim_wrapper', 'custom_hooks', 'param_scheduler', 'default_hooks', 'randomness'):
                self.assertEqual(cfg[key], before[key], key)
            self.assertIsNone(cfg.load_from)
            self.assertFalse(cfg.resume)
            self.assertIsNone(cfg.model.backbone.init_cfg)
            expected = deepcopy(before.model)
            expected.bbox_head.update(type='InstanceBalancedSemanticRTMDetHead', instance_qfl_weight=.1)
            expected.train_cfg.assigner.type = 'ROISemanticIndexedAssigner'
            self.assertEqual(cfg.model, expected)
            head = b.bbox_head.head_module
            for tower in (head.cls_convs, head.reg_convs):
                for layer in range(2):
                    self.assertIs(tower[0][layer].conv, tower[2][layer].conv)
                    self.assertIsNot(tower[0][layer].bn, tower[2][layer].bn)
            images, _ = data()
            calls = []
            hook = b.bbox_head.instance_qfl.register_forward_hook(lambda *args: calls.append(1))
            with torch.no_grad():
                for first, second in zip(a._forward(images), b._forward(images)):
                    for x, y in zip(first, second):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)
            hook.remove()
            self.assertEqual(calls, [])

    def test_main_and_roi_unchanged_one_head_assignment_and_bn_update(self):
        for method, ref in zip(METHODS, REFERENCES):
            a, b = build(ref).train(), build(method).train()
            images, samples = data()
            calls = dict(head=0, assignment=0, roi=0, extra=0)

            def counter(key):
                def hook(*args):
                    calls[key] += 1
                return hook

            hooks = [b.bbox_head.head_module.register_forward_hook(counter('head')),
                     b.bbox_head.assigner.register_forward_hook(counter('assignment')),
                     b.bbox_head.instance_semantic.register_forward_hook(counter('roi')),
                     b.bbox_head.instance_qfl.register_forward_hook(counter('extra'))]
            original, new = a.loss(images, samples), b.loss(images, samples)
            for hook in hooks:
                hook.remove()
            for key in ('loss_cls', 'loss_bbox', 'loss_instance_semantic'):
                torch.testing.assert_close(original[key], new[key], atol=0, rtol=0)
            self.assertEqual(calls, dict(head=1, assignment=1, roi=1, extra=1))
            self.assertEqual(float(new['ibqfl_all_gt']), 4)
            self.assertGreater(float(new['loss_instance_qfl']), 0)
            for n, layer in b.named_modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(int(layer.num_batches_tracked), 1, n)

    def test_zero_weight_gradients_and_adamw_update_match_each_reference(self):
        for method, ref in zip(METHODS, REFERENCES):
            a, b = build(ref).train(), build(method, weight=0).train()
            images, samples = data()
            for model in (a, b):
                losses = model.loss(images, samples)
                sum(v for k, v in losses.items() if 'loss' in k).backward()
            before = dict(a.named_parameters())
            for name, p in b.named_parameters():
                if before[name].grad is None:
                    self.assertIsNone(p.grad, name)
                else:
                    torch.testing.assert_close(p.grad, before[name].grad, atol=2e-7, rtol=2e-5, msg=name)
            for model in (a, b):
                torch.optim.AdamW(model.parameters(), lr=.0005625, weight_decay=.05).step()
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], atol=3e-7, rtol=3e-6, msg=name)

    def test_extra_gradient_reaches_features_predictors_and_gsce_not_aux_or_regression(self):
        model = build(METHODS[1]).train()
        images, samples = data()
        losses = model.loss(images, samples)
        losses['loss_instance_qfl'].backward()
        for prefix in ('backbone.', 'neck.', 'bbox_head.head_module.cls_convs.',
                       'bbox_head.head_module.rtm_cls.'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in model.named_parameters() if n.startswith(prefix)), prefix)
        for prefix in ('bbox_head.instance_semantic.', 'bbox_head.head_module.reg_convs.',
                       'bbox_head.head_module.rtm_reg.'):
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                                if n.startswith(prefix)), prefix)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d) and m.groups == 4 and m.kernel_size == (3, 3):
                self.assertIsNotNone(m.weight.grad, n)
                self.assertTrue(m.weight.grad.isfinite().all() and m.weight.grad.abs().sum() > 0, n)

    def test_tensor_list_and_empty_gt_loss_interfaces_and_state_roundtrip(self):
        images, batch = data()
        for empty in (False, True):
            rows = batch['bboxes_labels'][:0] if empty else batch['bboxes_labels']
            samples = []
            for image in range(2):
                raw = rows[rows[:, 0] == image]
                sample = DetDataSample(metainfo=dict(img_shape=(128, 128)))
                sample.gt_instances = InstanceData(bboxes=raw[:, 2:], labels=raw[:, 1].long())
                samples.append(sample)
            a, b = build().eval(), build().eval()
            first = a.loss(images, dict(bboxes_labels=rows, img_metas=batch['img_metas']))
            second = b.loss(images, samples)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=2e-7, rtol=2e-6, msg=key)
            b.load_state_dict(a.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
