"""Check ROI coordinate conventions, instance order, and unchanged detector."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from mmdet.structures.bbox import HorizontalBoxes
from custom_models.instance_semantic.roi import InstanceSemanticLoss

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


class InstanceSemanticTests(unittest.TestCase):
    def test_roi_coordinate_ramps_substride_boxes_batch_and_levels(self):
        module = InstanceSemanticLoss(1, 20)
        features = []
        for stride in (8, 16, 32):
            y, x = torch.meshgrid(torch.arange(640 // stride), torch.arange(640 // stride), indexing='ij')
            ramp = (x + 2 * y).float() * stride
            features.append(torch.stack((ramp, ramp + 1000))[:, None])
        rois = torch.tensor([[1, 5.25, 9.75, 6.25, 10.75],
                             [0, 8, 16, 136, 144], [1, 32, 32, 288, 288],
                             [0, 48, 32, 56, 36]], dtype=torch.float32)
        expected = (rois[:, 1] + rois[:, 3]) / 2 + rois[:, 2] + rois[:, 4] + 1000 * rois[:, 0]
        torch.testing.assert_close(module.pool_instances(features, rois)[:, 0], expected, atol=1e-4, rtol=1e-6)

    def test_empty_and_instance_balanced_loss(self):
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (80, 40, 20)]
        empty = module(features, torch.empty(0, 6))
        empty.backward()
        self.assertEqual(empty.item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
        rows = torch.tensor([[0, 1, 5, 5, 7, 7], [1, 2, 32, 32, 288, 288]], dtype=torch.float32)
        original = module(features, rows)
        duplicated = module(features, rows.repeat(3, 1))
        torch.testing.assert_close(original, duplicated)

    def test_empty_augmented_boxes_do_not_change_valid_instance_loss(self):
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (80, 40, 20)]
        # A positive, clipped border sliver can collapse at FP32 precision when
        # flipped to the opposite border; this is not an invalid raw annotation.
        boxes = HorizontalBoxes(torch.tensor([[0., 5., 1e-5, 7.]]))
        self.assertGreater(boxes.widths.item(), 0)
        boxes.flip_((640, 640), direction='horizontal')
        self.assertEqual(boxes.widths.item(), 0)
        empty = torch.cat((torch.tensor([[0., 1.]]), boxes.tensor), dim=1)
        # A small but representable positive box must still contribute. Do not
        # apply an arbitrary minimum-size or epsilon filter to valid targets.
        valid = torch.tensor([[0., 1., 100., 100., 100.001, 100.002],
                              [1., 2., 32., 32., 288., 288.]])
        degenerate = torch.tensor([[1., 0., 40., 40., 41., 40.],
                                   [1., 0., 50., 50., 49., 52.]])
        mixed = torch.cat((empty, valid, degenerate))
        before = mixed.clone()
        expected = module(features, valid)
        actual = module(features, mixed)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(mixed, before, atol=0, rtol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(actual))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in module.parameters()))

    def test_all_degenerate_instances_have_zero_connected_loss(self):
        module = InstanceSemanticLoss(4, 3)
        features = [torch.randn(2, 4, s, s, requires_grad=True) for s in (80, 40, 20)]
        rows = torch.tensor([[0., 1., 2., 3., 2., 5.],
                             [1., 2., 4., 5., 6., 5.]])
        loss = module(features, rows)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all()
                            for x in features))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in module.parameters()))

    def test_baseline_predictions_losses_and_aux_gradient_scope(self):
        configs = [Config.fromfile(str(ROOT / path)) for path in
                   ('configs/rtmdet_tiny_dior_300e.py', 'configs/instance_semantic/rtmdet_tiny_dior_640.py')]
        models = []
        for cfg in configs:
            torch.manual_seed(42)
            model = MODELS.build(deepcopy(cfg.model))
            logging.disable(logging.CRITICAL); model.init_weights(); model.eval()
            models.append(model)
        base, model = models
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
        images = torch.randn(2, 3, 128, 128)
        data = dict(bboxes_labels=torch.tensor([[0, 1, 18, 20, 34, 38], [1, 2, 40, 50, 80, 90]], dtype=torch.float32),
                    img_metas=[dict(img_shape=(128, 128))] * 2)
        with torch.no_grad():
            for actual, expected in zip(model._forward(images), base._forward(images)):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        main, extra = base.loss(images, data), model.loss(images, data)
        for key in main:
            torch.testing.assert_close(main[key], extra[key], atol=0, rtol=0)
        extra['loss_instance_semantic'].backward()
        self.assertGreater(model.bbox_head.head_module.cls_convs[0][0].conv.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.bbox_head.head_module.reg_convs[0][0].conv.weight.grad)
        self.assertIsNone(model.bbox_head.head_module.rtm_cls[0].weight.grad)
        self.assertGreater(model.backbone.stem[0].conv.weight.grad.abs().sum().item(), 0)
        calls = []
        handle = model.bbox_head.instance_semantic.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            model._forward(images)
        handle.remove()
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
