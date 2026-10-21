"""Check exact assignment, instance identities, real gradients and parity."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmyolo.models.task_modules.assigners.batch_dsl_assigner import BatchDynamicSoftLabelAssigner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_rank import (
    IndexedDynamicSoftLabelAssigner, InstanceCategoryRankLoss)

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build_model(rank=True, weight=.1):
    method = 'instance_semantic_rank' if rank else 'instance_semantic'
    cfg = Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))
    if rank:
        cfg.model.bbox_head.rank_loss_weight = weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def model_data():
    torch.manual_seed(19)
    images = torch.randn(2, 3, 128, 128)
    # Same class within/across images must retain separate GT identities.
    data = dict(bboxes_labels=torch.tensor(
        [[0, 1, 18, 20, 34, 38], [0, 1, 44, 48, 88, 92],
         [1, 1, 18, 20, 34, 38], [1, 2, 28, 28, 116, 116]], dtype=torch.float32),
        img_metas=[dict(img_shape=(128, 128))] * 2)
    return images, data


class InstanceRankTests(unittest.TestCase):
    def test_assigner_exact_official_outputs_and_gt_identity(self):
        for batch_iou in (True, False):
            reference = BatchDynamicSoftLabelAssigner(4, batch_iou=batch_iou)
            module = IndexedDynamicSoftLabelAssigner(4, batch_iou=batch_iou)
            for seed in range(6):
                torch.manual_seed(seed)
                priors = torch.cat((torch.rand(37, 2) * 128, torch.full((37, 2), 8.)), dim=1)
                xy = torch.rand(3, 37, 2) * 90
                pred = torch.cat((xy, xy + torch.rand_like(xy) * 60 + 1), dim=-1)
                scores = torch.randn(3, 37, 4)
                xy = torch.rand(3, 5, 2) * 90
                boxes = torch.cat((xy, xy + torch.rand_like(xy) * 40 + 2), dim=-1)
                boxes[1, 3:] = 0
                boxes[2] = 0  # A completely empty image inside a nonempty batch.
                boxes[0, 1] = boxes[0, 0]  # Identical boxes, different GT IDs.
                labels = torch.randint(0, 4, (3, 5, 1)).float()
                labels[0, :2] = 1
                flags = (boxes.sum(-1, keepdim=True) > 0).float()
                inputs = (pred, scores, priors, labels, boxes, flags)
                matching = []
                official_match = reference.dynamic_k_matching

                def trace_matching(*args):
                    result = official_match(*args)
                    matching.append(result)
                    return result

                with patch.object(reference, 'dynamic_k_matching', side_effect=trace_matching):
                    expected = reference(*inputs)
                actual = module(*inputs)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                indices = actual['assigned_gt_indices']
                expected_indices = torch.full_like(indices, -1)
                expected_indices[matching[0][2]] = matching[0][1]
                torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
                pos = indices >= 0
                self.assertTrue(torch.equal(pos, actual['assigned_labels'] < 4))
                b, p = pos.nonzero(as_tuple=True)
                torch.testing.assert_close(actual['assigned_bboxes'][b, p], boxes[b, indices[b, p]], atol=0, rtol=0)
                torch.testing.assert_close(actual['assigned_labels'][b, p], labels[b, indices[b, p], 0].long(), atol=0, rtol=0)
                self.assertTrue((indices[2] == -1).all())

    def test_assigner_empty_outside_and_zero_iou_preserves_official_matching(self):
        original, module = BatchDynamicSoftLabelAssigner(3), IndexedDynamicSoftLabelAssigner(3)
        for n, g, outside in ((0, 2, False), (9, 0, False), (9, 2, False), (9, 2, True)):
            boxes = torch.tensor([[[16., 16., 32., 32.], [40., 40., 56., 56.]]])[:, :g]
            priors = torch.tensor([[24., 24., 8., 8.]]).expand(n, 4).clone()
            if outside:
                priors[:, :2] = 1000
            inputs = (torch.zeros(1, n, 4), torch.zeros(1, n, 3), priors,
                      torch.zeros(1, g, 1), boxes, torch.ones(1, g, 1))
            expected, actual = original(*inputs), module(*inputs)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            self.assertTrue(torch.equal(actual['assigned_gt_indices'] >= 0, actual['assigned_labels'] < 3))

    def test_grouping_matches_loop_values_gradients_and_instance_balance(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 7, 4, requires_grad=True)
        reference = logits.detach().clone().requires_grad_()
        indices = torch.tensor([[0, 0, 0, 1, -1, -1, -1], [0, 1, 1, 1, 1, -1, -1]])
        # All equal categories; separate instances must still give four terms.
        labels = torch.tensor([[[2], [2], [1]], [[2], [2], [0]]])
        module = InstanceCategoryRankLoss(.1)
        pooled, targets, keys, counts = module.pool_logits(logits, indices, labels)
        self.assertEqual(keys.tolist(), [0, 1, 3, 4])
        self.assertEqual(counts.tolist(), [3, 1, 1, 4])
        terms = [F.cross_entropy(reference[b, indices[b] == g].mean(0)[None], labels[b, g])
                 for b in range(2) for g in range(3) if (indices[b] == g).any()]
        expected = torch.stack(terms).mean() * .1
        actual = module(logits, indices, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward(); expected.backward()
        torch.testing.assert_close(logits.grad, reference.grad)
        self.assertTrue((logits.grad[indices < 0] == 0).all())
        # Repeating an instance's entire prior set does not increase its weight.
        duplicate_logits = torch.cat((logits.detach(), logits.detach()[:, :3]), dim=1)
        duplicate_ids = torch.cat((indices, torch.tensor([[0, 0, 0], [-1, -1, -1]])), dim=1)
        torch.testing.assert_close(module(duplicate_logits, duplicate_ids, labels), actual)
        torch.testing.assert_close(pooled, torch.stack([reference[b, indices[b] == g].mean(0)
                                  for b in range(2) for g in range(2)]))
        self.assertEqual(targets.tolist(), [2] * 4)

    def test_relative_logit_ranking_no_absolute_score_target(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 6, 4, requires_grad=True)
        ids = torch.tensor([[0, 0, 1, -1, -1, -1], [0, 1, 1, -1, -1, -1]])
        labels = torch.tensor([[[1], [2]], [[0], [1]]])
        loss = InstanceCategoryRankLoss(.1)
        original = loss(logits, ids, labels)
        shifted = loss(logits + torch.randn(2, 6, 1) * 3, ids, labels)
        torch.testing.assert_close(original, shifted)
        original.backward()
        torch.testing.assert_close(logits.grad.sum(-1), torch.zeros(2, 6), atol=3e-9, rtol=0)
        with self.assertRaises(ValueError):
            InstanceCategoryRankLoss(float('nan'))

    def test_empty_and_ddp_normalization_including_empty_rank(self):
        module = InstanceCategoryRankLoss(.1)
        logits = torch.randn(1, 4, 3, requires_grad=True)
        for labels in (torch.empty(1, 0, 1), torch.tensor([[[1.]]])):
            with patch('custom_models.instance_semantic_rank.loss.reduce_mean', return_value=torch.tensor(.5)) as mean:
                loss = module(logits, torch.full((1, 4), -1), labels)
                self.assertEqual(mean.call_count, 1)  # Empty ranks still reduce.
                self.assertEqual(loss.item(), 0)
                loss.backward()
                self.assertTrue((logits.grad == 0).all())
        ids, labels = torch.tensor([[0, -1, -1, -1]]), torch.tensor([[[1]]])
        local = module(logits, ids, labels)
        with patch('custom_models.instance_semantic_rank.loss.reduce_mean', return_value=torch.tensor(.5)):
            torch.testing.assert_close(module(logits, ids, labels), local * 2)

    def test_zero_weight_model_predictions_losses_gradients_and_update_equal(self):
        original, model = build_model(False), build_model(True, 0.)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4880648)
        self.assertEqual(set(original.state_dict()), set(model.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
        images, data = model_data()
        original.eval(); model.eval()
        with torch.no_grad():
            for first, second in zip(original._forward(images), model._forward(images)):
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        original.train(); model.train()
        first, second = original.loss(images, data), model.loss(images, data)
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0, msg=key)
        self.assertEqual(second['loss_instance_rank'].item(), 0)
        sum(first.values()).backward(); sum(second.values()).backward()
        refs = dict(original.named_parameters())
        for name, p in model.named_parameters():
            self.assertEqual(p.grad is None, refs[name].grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, refs[name].grad, atol=0, rtol=0, msg=name)
        for item in (original, model):
            torch.optim.AdamW(item.parameters(), lr=.0005625).step()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)

    def test_active_rank_preserves_main_losses_and_reaches_real_predictors_once(self):
        original, model = build_model(False).train(), build_model(True).train()
        images, data = model_data()
        calls, roi_rows = [], []
        handles = [model.bbox_head.assigner.register_forward_hook(lambda *args: calls.append(1)),
                   model.bbox_head.instance_semantic.register_forward_pre_hook(
                       lambda mod, args: roi_rows.append(args[1].clone()))]
        first, second = original.loss(images, data), model.loss(images, data)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [1])
        torch.testing.assert_close(roi_rows[0], data['bboxes_labels'], atol=0, rtol=0)
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0, msg=key)
        self.assertGreater(second['loss_instance_rank'].item(), 0)
        second['loss_instance_rank'].backward()
        head = model.bbox_head.head_module
        self.assertTrue(any(p.weight.grad is not None and p.weight.grad.abs().sum() > 0 for p in head.rtm_cls))
        self.assertGreater(head.cls_convs[0][0].conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.stem[0].conv.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.rtm_reg.parameters()))
        self.assertTrue(all(p.grad is None for p in head.reg_convs.parameters()))
        self.assertTrue(all(p.grad is None for p in model.bbox_head.instance_semantic.parameters()))
        # Every BN runs once: neither a second image pass nor a second head pass.
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(int(module.num_batches_tracked), 1, name)
        model.eval()
        with patch.object(model.bbox_head.instance_rank, 'forward', side_effect=AssertionError('inference')):
            with torch.no_grad():
                model._forward(images)

    def test_all_gt_roi_is_retained_when_assignment_has_no_positive(self):
        model = build_model(True).eval()
        images, data = model_data()
        real_assigner = model.bbox_head.assigner

        with patch.object(real_assigner, 'forward', wraps=real_assigner.forward) as forward:
            # Modify only the returned assignment via a hook, avoiding recursion.
            def erase(mod, args, result):
                result['assigned_gt_indices'].fill_(-1)
                result['assigned_labels'].fill_(20)
                result['assign_metrics'].zero_()
            handle = real_assigner.register_forward_hook(erase)
            losses = model.loss(images, data)
            handle.remove()
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(losses['loss_instance_rank'].item(), 0)
        self.assertGreater(losses['loss_instance_semantic'].item(), 0)
        losses['loss_instance_semantic'].backward()
        self.assertGreater(model.bbox_head.instance_semantic.classifier.weight.grad.abs().sum().item(), 0)

    def test_list_gt_interface_empty_gt_and_predict_only_construction(self):
        model = build_model(True).eval()
        images, data = model_data()
        rows = data['bboxes_labels']
        gt = [InstanceData(bboxes=rows[rows[:, 0] == b, 2:], labels=rows[rows[:, 0] == b, 1].long())
              for b in range(2)]
        with torch.no_grad():
            feats = model.extract_feat(images)
            cls, reg = model.bbox_head(feats)
            first = model.bbox_head.loss_by_feat(cls, reg, rows, data['img_metas'])
            second = model.bbox_head.loss_by_feat(cls, reg, gt, data['img_metas'])
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            empty = dict(bboxes_labels=torch.empty(0, 6), img_metas=data['img_metas'])
            losses = model.loss(images, empty)
        self.assertEqual(losses['loss_instance_rank'].item(), 0)
        self.assertEqual(losses['loss_instance_semantic'].item(), 0)
        self.assertTrue(all(torch.isfinite(x) for x in losses.values()))
        cfg = Config.fromfile(str(ROOT / 'configs/instance_semantic_rank/rtmdet_tiny_dior_640.py'))
        cfg.model.train_cfg = None
        predict_only = MODELS.build(deepcopy(cfg.model)).eval()
        with torch.no_grad():
            predict_only._forward(images)


if __name__ == '__main__':
    unittest.main()
