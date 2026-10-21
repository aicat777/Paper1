"""Verify DSL parity, local rescue conflicts, GT identity and loss weighting."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmdet.structures.bbox import distance2bbox
from mmyolo.models.task_modules.assigners.batch_dsl_assigner import BatchDynamicSoftLabelAssigner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from custom_models.tiny_assignment.assigner import TinyRescueDSLAssigner, local_bipartite_matching

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def grid_priors():
    y, x = torch.meshgrid(torch.arange(4), torch.arange(5), indexing='ij')
    return torch.cat((torch.stack((x, y), -1).reshape(-1, 2) * 8,
                      torch.full((20, 2), 8)), 1).float()


class TinyAssignmentTests(unittest.TestCase):
    def test_disabled_is_official_dsl_including_background_and_padding(self):
        base = BatchDynamicSoftLabelAssigner(3)
        custom = TinyRescueDSLAssigner(3, enable_rescue=False, regression_floor=0)
        torch.manual_seed(7)
        priors = grid_priors()
        pred = distance2bbox(priors[:, :2], torch.rand(2, len(priors), 4) * 12)
        scores = torch.randn(2, len(priors), 3)
        boxes = torch.tensor([[[1, 1, 3, 3], [8, 4, 30, 22], [0, 0, 0, 0]],
                              [[4, 4, 20, 20], [17, 9, 19, 11], [24, 4, 31, 20]]]).float()
        labels = torch.tensor([[[0], [1], [0]], [[2], [0], [1]]]).float()
        pad = (boxes.sum(-1, keepdim=True) > 0).float()
        for gt, lab, flags in ((boxes, labels, pad), (boxes[:, :0], labels[:, :0], pad[:, :0]),
                                (boxes * 0, labels * 0, pad * 0)):
            expected = base(pred, scores, priors, lab, gt, flags)
            actual = custom(pred, scores, priors, lab, gt, flags)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0, msg=key)
            torch.testing.assert_close(actual['regression_weights'], actual['assign_metrics'], atol=0, rtol=0)

    def test_augmenting_path_and_infeasible_candidates(self):
        result = local_bipartite_matching({0: [5, 6], 1: [5], 2: [5]})
        self.assertEqual(result, {0: 6, 1: 5})
        self.assertEqual(local_bipartite_matching({0: [], 1: [7]}), {1: 7})
        # Dense chain exercises iterative reallocation rather than greedy loss
        # of a feasible match or recursive stack overflow.
        candidates = {g: [g, g + 1] for g in range(1100)}
        candidates[1100] = [0]
        result = local_bipartite_matching(candidates)
        self.assertEqual(len(result), 1101)
        self.assertEqual(len(set(result.values())), 1101)

    def test_duplicate_boxes_have_distinct_gt_identity_and_only_local_matches(self):
        priors = grid_priors()
        pred = torch.tensor([0., 0., .5, .5]).expand(1, len(priors), 4).clone()
        boxes = torch.tensor([[[1., 1., 3., 3.], [1., 1., 3., 3.], [0., 0., 0., 0.]]])
        labels = torch.tensor([[[1.], [2.], [0.]]])
        assigner = TinyRescueDSLAssigner(3)
        result = assigner(pred, torch.zeros(1, len(priors), 3), priors, labels, boxes,
                           torch.tensor([[[1.], [1.], [0.]]]))
        positive = result['assigned_gt_indices'][0] >= 0
        self.assertEqual(sorted(result['assigned_gt_indices'][0, positive].tolist()), [0, 1])
        self.assertEqual(sorted(result['assigned_labels'][0, positive].tolist()), [1, 2])
        self.assertTrue(result['fallback_mask'][0, positive].all())
        self.assertFalse(result['unmatched_tiny'].any())
        delta = (priors[positive, :2] - 2).abs() / 8
        self.assertTrue((delta <= 1.5).all())
        self.assertEqual(result['assign_metrics'].sum().item(), 0)
        torch.testing.assert_close(result['regression_weights'][0, positive], torch.full((2,), .05))

    def test_rescue_preserves_existing_valid_baseline_positives(self):
        priors = grid_priors()
        boxes = torch.tensor([[[0., 0., 20., 20.], [1., 1., 3., 3.]]])
        labels = torch.tensor([[[0.], [1.]]])
        pred = boxes[:, :1].expand(1, len(priors), 4).clone()
        scores = torch.zeros(1, len(priors), 3)
        pad = torch.ones(1, 2, 1)
        base = TinyRescueDSLAssigner(3, enable_rescue=False)(pred, scores, priors, labels, boxes, pad)
        actual = TinyRescueDSLAssigner(3)(pred, scores, priors, labels, boxes, pad)
        held = base['assigned_gt_indices'] == 0
        self.assertTrue(held.any())
        torch.testing.assert_close(actual['assigned_gt_indices'][held], base['assigned_gt_indices'][held])
        torch.testing.assert_close(actual['assign_metrics'][held], base['assign_metrics'][held])

    def test_local_capacity_exhaustion_does_not_create_far_or_duplicate_matches(self):
        priors = grid_priors()
        boxes = torch.tensor([[[1., 1., 3., 3.]]]).repeat(1, 10, 1)
        result = TinyRescueDSLAssigner(3)(
            torch.tensor([[[0., 0., .5, .5]]]).expand(1, len(priors), 4),
            torch.zeros(1, len(priors), 3), priors, torch.zeros(1, 10, 1),
            boxes, torch.ones(1, 10, 1))
        positive = result['assigned_gt_indices'][0] >= 0
        # Exactly four stride-8 points lie within 1.5 strides of (2, 2).
        self.assertEqual(int(positive.sum()), 4)
        self.assertEqual(int(result['unmatched_tiny'].sum()), 6)
        self.assertEqual(len(result['assigned_gt_indices'][0, positive].unique()), 4)
        self.assertTrue(((priors[positive, :2] - 2).abs() <= 12).all())

    def test_regression_floor_provides_gradient_without_changing_qfl(self):
        cfg = Config.fromfile(str(ROOT / 'configs/tiny_assignment/rtmdet_tiny_dior_640.py'))
        head_cfg = deepcopy(cfg.model.bbox_head)
        head_cfg.train_cfg = cfg.model.train_cfg
        head = MODELS.build(head_cfg)
        logits = torch.randn(32, 20, requires_grad=True)
        boxes = torch.tensor([[4., 4., 6., 6.]]).repeat(32, 1).requires_grad_()
        result = dict(assigned_labels=torch.zeros(1, 32, dtype=torch.long),
                      assigned_labels_weights=torch.ones(1, 32),
                      assign_metrics=torch.zeros(1, 32), regression_weights=torch.full((1, 32), .05),
                      assigned_bboxes=torch.tensor([[[1., 1., 2., 2.]]]).repeat(1, 32, 1))
        loss = head.losses_from_assignment(logits, boxes, result, torch.full((32,), 8.))
        expected_cls = head.loss_cls(logits, (result['assigned_labels'].flatten(), torch.zeros(32)),
                                     torch.ones(32), avg_factor=1)
        expected_bbox = 2 * torch.nn.functional.smooth_l1_loss(
            boxes / 8, result['assigned_bboxes'].reshape(-1, 4) / 8)
        torch.testing.assert_close(loss['loss_cls'], expected_cls)
        torch.testing.assert_close(loss['loss_bbox'], expected_bbox)
        loss['loss_bbox'].backward()
        self.assertTrue(torch.isfinite(boxes.grad).all())
        self.assertGreater(boxes.grad.abs().sum().item(), 0)
        # A point outside a GT needs signed distances; it remains representable.
        torch.testing.assert_close(distance2bbox(torch.tensor([[8., 8.]]),
                                                torch.tensor([[-5., -6., 7., 8.]])),
                                   torch.tensor([[13., 14., 15., 16.]]))

    def test_reversed_predictions_get_ordered_edge_gradients_not_invalid_giou(self):
        cfg = Config.fromfile(str(ROOT / 'configs/tiny_assignment/rtmdet_tiny_dior_640.py'))
        head_cfg = deepcopy(cfg.model.bbox_head); head_cfg.train_cfg = cfg.model.train_cfg
        head = MODELS.build(head_cfg)
        boxes = torch.tensor([[10., 10., 0., 0.]], requires_grad=True)
        target = torch.tensor([[[1., 1., 2., 2.]]])
        # Concrete counterexample: simply floor-weighting GIoU rewards an
        # increasingly reversed box, rather than repairing it.
        self.assertLess(head.loss_bbox(boxes, target[0]).item(), 0)
        result = dict(assigned_labels=torch.zeros(1, 1, dtype=torch.long),
                      assigned_labels_weights=torch.ones(1, 1), assign_metrics=torch.zeros(1, 1),
                      regression_weights=torch.full((1, 1), .05), assigned_bboxes=target)
        loss = head.losses_from_assignment(torch.zeros(1, 20), boxes, result, torch.tensor([8.]))['loss_bbox']
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue((boxes.grad[0, :2] > 0).all())  # Descent lowers x1/y1.
        self.assertTrue((boxes.grad[0, 2:] < 0).all())  # Descent raises x2/y2.

    def test_disabled_head_loss_matches_baseline(self):
        base_cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        cfg = Config.fromfile(str(ROOT / 'configs/tiny_assignment/rtmdet_tiny_dior_640.py'))
        cfg.model.train_cfg.assigner.update(enable_rescue=False, regression_floor=0)
        models = []
        for item in (base_cfg, cfg):
            torch.manual_seed(42)
            model = MODELS.build(deepcopy(item.model))
            logging.disable(logging.CRITICAL); model.init_weights(); model.eval()
            models.append(model)
        data = dict(bboxes_labels=torch.tensor([[0, 1, 18, 20, 34, 38], [1, 2, 40, 50, 80, 90]], dtype=torch.float32),
                    img_metas=[dict(img_shape=(128, 128))] * 2)
        images = torch.randn(2, 3, 128, 128)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], atol=0, rtol=0)
        expected, actual = models[0].loss(images, data), models[1].loss(images, data)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
