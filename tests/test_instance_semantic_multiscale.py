"""Geometry, isolation, exact initialization and two-backward update semantics."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from mmdet.structures.bbox import HorizontalBoxes
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_multiscale.data import (
    SnapshotObjectCrop, object_crop_collate, square_object_crop)
from custom_models.instance_semantic_multiscale.normalization import crop_normalization
from custom_models.instance_semantic_multiscale.roi import CropSemanticLoss

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config():
    return Config.fromfile(str(ROOT / 'configs/instance_semantic_multiscale/rtmdet_tiny_dior_640.py'))


def build():
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(config().model))
    model.init_weights()
    return model


def example_data(size=128):
    torch.manual_seed(7)
    return dict(inputs=torch.randint(0, 256, (2, 3, size, size), dtype=torch.uint8),
        data_samples=dict(bboxes_labels=torch.tensor([[0, 1, 18, 20, 34, 38], [1, 2, 40, 50, 80, 90.]]),
            object_crops=dict(images=[torch.randint(0, 256, (3, 24, 24), dtype=torch.uint8),
                                      torch.randint(0, 256, (3, 40, 40), dtype=torch.uint8)],
                boxes=torch.tensor([[1., 2, 23, 20], [0, 0, 40, 40]]), labels=torch.tensor([1, 2]),
                batch_indices=torch.tensor([0, 1]), metadata=[])))


def optimizer(model):
    options = deepcopy(config().optim_wrapper)
    options.type = 'ObjectCropOptimWrapper'
    options.pop('loss_scale', None)
    return build_optim_wrapper(model, options)


def buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


class MultiScaleInstanceTests(unittest.TestCase):
    def test_fractional_border_and_thin_boxes_are_enclosed_without_distortion(self):
        image = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
        for box in ([1.1, 2.2, 6.1, 3.2], [0, 0, 12, .001], [11, 6, 13, 9], [3, 2, 3.001, 2.002]):
            crop, relative, origin = square_object_crop(image, box)
            left, top = origin
            side = crop.shape[0]
            self.assertEqual(crop.shape[1], side)
            self.assertTrue((relative[:2] >= 0).all() and (relative[2:] <= side).all())
            np.testing.assert_allclose(relative[2:] - relative[:2], np.array(box, np.float32)[2:] - np.array(box, np.float32)[:2], atol=1e-6)
            for y in range(side):
                for x in range(side):
                    expected = image[y + top, x + left] if 0 <= y + top < 9 and 0 <= x + left < 13 else np.full(3, 114)
                    np.testing.assert_array_equal(crop[y, x], expected)

    def test_snapshot_is_clean_and_does_not_consume_main_augmentation_rng(self):
        transform = SnapshotObjectCrop()
        image = np.full((32, 48, 3), 29, dtype=np.uint8)
        result = dict(img=image.copy(), img_id=12, gt_bboxes=HorizontalBoxes(torch.tensor([[2., 3, 20, 25]])),
                      gt_bboxes_labels=np.array([4]), gt_ignore_flags=np.array([False]))
        np.random.seed(18)
        before = np.random.get_state()
        transform(result)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        result['img'][:] = 200
        self.assertTrue((result['object_crop']['image'] == 29).all())
        self.assertEqual(result['object_crop']['label'], 4)
        self.assertFalse(np.shares_memory(result['img'], result['object_crop']['image']))

    def test_strong_and_weak_detection_pipelines_match_previous_exactly(self):
        previous = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
        new = config()
        for key in ('train_pipeline', 'train_pipeline_stage2'):
            base, modified = Compose(previous[key]), Compose(new[key])
            # Real loading transforms are replaced by already-loaded, identical
            # inputs. Cached Mosaic/MixUp still execute over twelve samples.
            base.transforms = base.transforms[2:]
            modified.transforms = modified.transforms[2:]
            first, second = [], []
            for pipeline, outputs in ((base, first), (modified, second)):
                np.random.seed(42)
                for index in range(12):
                    image = np.full((800, 800, 3), 20 + index, dtype=np.uint8)
                    result = dict(img=image, img_id=index, img_path='test.jpg', ori_shape=(800, 800),
                        img_shape=(800, 800), gt_bboxes=HorizontalBoxes(torch.tensor([[20., 30, 400, 600]])),
                        gt_bboxes_labels=np.array([index % 20]), gt_ignore_flags=np.array([False]))
                    outputs.append(pipeline(result))
            for old, changed in zip(first, second):
                # RandomCrop legitimately rejects a sample if every GT is lost.
                self.assertEqual(old is None, changed is None)
                if old is None:
                    continue
                torch.testing.assert_close(old['inputs'], changed['inputs'], atol=0, rtol=0)
                torch.testing.assert_close(old['data_samples'].gt_instances.bboxes.tensor,
                                           changed['data_samples'].gt_instances.bboxes.tensor, atol=0, rtol=0)
                torch.testing.assert_close(old['data_samples'].gt_instances.labels,
                                           changed['data_samples'].gt_instances.labels, atol=0, rtol=0)
                crop = changed['data_samples'].object_crop
                self.assertEqual(crop['source_img_id'], changed['data_samples'].img_id)
                center = ((crop['box'][:2] + crop['box'][2:]) / 2).long()
                self.assertEqual(int(crop['image'][0, center[1], center[0]]), 20 + crop['source_img_id'])
            packed = [sample for sample in second if sample is not None]
            self.assertGreaterEqual(len(packed), 2)
            collated = object_crop_collate(packed[:2])
            self.assertEqual(len(collated['data_samples']['object_crops']['images']), 2)

    def test_resize_normalization_roi_coordinates_and_real_box_level(self):
        model = build()
        preprocessor = model.data_preprocessor
        image = torch.stack([torch.full((512, 512), value, dtype=torch.uint8) for value in (10, 20, 30)])
        crops = dict(images=[image], boxes=torch.tensor([[0., 250, 512, 251]]), labels=torch.tensor([2]))
        prepared, rows = preprocessor.prepare_crops(crops, 256)
        # RTMDet's inherited means/stds and input order are BGR, unchanged.
        self.assertFalse(preprocessor._channel_conversion)
        expected = (torch.tensor([10., 20, 30]).reshape(3, 1, 1) - preprocessor.mean) / preprocessor.std
        torch.testing.assert_close(prepared[0], expected.expand(3, 256, 256))
        preprocessor._channel_conversion = True
        converted, _ = preprocessor.prepare_crops(crops, 256)
        rgb = (torch.tensor([30., 20, 10]).reshape(3, 1, 1) - preprocessor.mean) / preprocessor.std
        torch.testing.assert_close(converted[0], rgb.expand(3, 256, 256))
        torch.testing.assert_close(rows, torch.tensor([[0., 2, 0, 125, 256, 125.5]]))
        # A thin 256-wide GT still selects P3 because selection uses sqrt(w*h),
        # not the 256x256 padded square canvas.
        self.assertEqual(int(model.bbox_head.instance_semantic.levels(rows[:, 2:])[0]), 0)

    def test_roi_ramps_substride_fractional_coordinates_and_all_levels(self):
        loss = CropSemanticLoss(1, 20)
        maps = {}
        for level, stride in enumerate((8, 16, 32)):
            y, x = torch.meshgrid(torch.arange(640 // stride), torch.arange(640 // stride), indexing='ij')
            ramp = (x + 2 * y).float() * stride
            maps[level] = torch.stack((ramp, ramp + 1000))[:, None]
        rois = torch.tensor([[1, 5.25, 9.75, 6.25, 10.75], [0, 8, 16, 136, 144], [1, 32, 32, 288, 288.]])
        levels = loss.levels(rois[:, 1:])
        self.assertEqual(levels.tolist(), [0, 1, 2])
        expected = (rois[:, 1] + rois[:, 3]) / 2 + rois[:, 2] + rois[:, 4] + 1000 * rois[:, 0]
        torch.testing.assert_close(loss.pool_instances(maps, rois, levels)[:, 0], expected, atol=1e-4, rtol=1e-6)

    def test_crop_bn_output_gradient_buffers_and_exception_restore(self):
        module = nn.Sequential(nn.BatchNorm2d(4)).train()
        reference = deepcopy(module)
        value = torch.randn(2, 4, 3, 5, requires_grad=True)
        ref_value = value.detach().clone().requires_grad_()
        before = buffers(module)
        with crop_normalization(module):
            actual = module(value)
        expected = reference(ref_value)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(value.grad, ref_value.grad)
        for name, tensor in buffers(module).items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
        self.assertNotIn('forward', module[0].__dict__)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with crop_normalization(module):
                raise RuntimeError('injected')
        self.assertNotIn('forward', module[0].__dict__)
        with crop_normalization(module):
            singleton = module(torch.randn(1, 4, 1, 1, requires_grad=True))
        self.assertTrue(torch.isfinite(singleton).all())
        singleton.sum().backward()

    def test_previous_initial_state_and_inference_are_unchanged(self):
        old_cfg = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
        torch.manual_seed(42)
        previous = MODELS.build(deepcopy(old_cfg.model))
        previous.init_weights()
        previous.eval()
        model = build()
        model.eval()
        for name, value in previous.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
        with torch.no_grad():
            images = torch.randn(2, 3, 128, 128)
            for left, right in zip(previous._forward(images), model._forward(images)):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4880648)

    def test_auxiliary_reaches_shared_features_not_regression_or_prediction(self):
        model = build()
        model.train()
        data = model.data_preprocessor(example_data(), training=True)
        before = buffers(model)
        cls_calls, reg_calls = [], []
        # ModuleList is iterated rather than called, so observe its first layer.
        hooks = [tower[0].register_forward_hook(lambda *args: cls_calls.append(1))
                 for tower in model.bbox_head.head_module.cls_convs]
        hooks += [tower[0].register_forward_hook(lambda *args: reg_calls.append(1))
                  for tower in model.bbox_head.head_module.reg_convs]
        loss = model.auxiliary_loss(data['object_crops'], 128)
        loss.backward()
        for handle in hooks:
            handle.remove()
        self.assertTrue(cls_calls and not reg_calls)
        self.assertGreater(float(model.backbone.stem[0].conv.weight.grad.abs().max()), 0)
        self.assertGreater(float(model.bbox_head.instance_semantic.classifier.weight.grad.abs().max()), 0)
        self.assertIsNone(model.bbox_head.head_module.reg_convs[0][0].conv.weight.grad)
        self.assertIsNone(model.bbox_head.head_module.rtm_cls[0].weight.grad)
        for name, tensor in buffers(model).items():
            torch.testing.assert_close(tensor, before[name], atol=0, rtol=0, msg=name)

    def test_split_backwards_match_joint_adamw_update_and_one_iteration(self):
        separate, joint = build(), build()
        separate.train()
        joint.train()
        first, second = optimizer(separate), optimizer(joint)
        data = example_data()
        # Same sampled scale and RNG advance in both schedules.
        expected_scale = joint.sample_crop_scale()
        self.assertEqual(expected_scale, 32)
        prepared = joint.data_preprocessor(deepcopy(data), training=True)
        auxiliary = joint.auxiliary_loss(prepared.pop('object_crops'), expected_scale)
        main_losses = joint._run_forward(prepared, mode='loss')
        main, _ = joint.parse_losses(main_losses)
        second.update_params(main + auxiliary)
        separate.train_step(deepcopy(data), first)
        self.assertEqual(first._inner_count, 1)
        self.assertEqual(second._inner_count, 1)
        for name, value in separate.state_dict().items():
            torch.testing.assert_close(value, joint.state_dict()[name], atol=2e-7, rtol=2e-5, msg=name)
        for parameter in separate.parameters():
            self.assertEqual(int(first.optimizer.state[parameter]['step']), 1)
        counts = [int(value) for name, value in separate.named_buffers() if name.endswith('num_batches_tracked')]
        self.assertTrue(counts and all(count == 1 for count in counts))

    def test_scale_rng_is_independent_and_resumes_from_state_dict(self):
        model = build()
        state = torch.get_rng_state().clone()
        for _ in range(7):
            model.sample_crop_scale()
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        expected = [model.sample_crop_scale() for _ in range(12)]
        buffer.seek(0)
        restored = build()
        restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        self.assertEqual([restored.sample_crop_scale() for _ in range(12)], expected)

    def test_empty_and_singleton_crop_batches_remain_valid(self):
        model = build()
        model.train()
        data = model.data_preprocessor(example_data(), training=True)
        crops = data['object_crops']
        crops['images'] = crops['images'][:1]
        crops['boxes'], crops['labels'] = crops['boxes'][:1], crops['labels'][:1]
        singleton = model.auxiliary_loss(crops, 32)
        self.assertTrue(torch.isfinite(singleton))
        singleton.backward()
        crops['images'] = []
        crops['boxes'], crops['labels'] = crops['boxes'][:0], crops['labels'][:0]
        empty = model.auxiliary_loss(crops, 800)
        self.assertEqual(float(empty), 0.)
        empty.backward()


if __name__ == '__main__':
    unittest.main()
