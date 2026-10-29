"""Independent selective derivatives, real GSCE loss and optimizer parity."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_neckhead import route_parameter_gradients

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown_p3_g4_semantic_neckhead'
REFERENCE = 'scdown_p3_g4_semantic_gradscale'
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method=METHOD):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method=METHOD, alpha=.25):
    cfg = config(method)
    cfg.model.bbox_head.semantic_feature_gradient_scale = alpha
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def data(empty=False, as_list=False):
    torch.manual_seed(17)
    images = torch.randn(2, 3, 256, 256)
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


def grads(loss, model, retain_graph=False):
    return torch.autograd.grad(loss, tuple(model.parameters()),
                               allow_unused=True, retain_graph=retain_graph)


def assert_gradient(test, actual, expected, name='', exact=False):
    if expected is None:
        test.assertTrue(actual is None or not actual.any(), name)
    else:
        test.assertIsNotNone(actual, name)
        torch.testing.assert_close(actual, expected, atol=0 if exact else 3e-6,
                                   rtol=0 if exact else 8e-5, msg=name)


class ROINeckHeadTests(unittest.TestCase):
    def test_closed_form_with_shared_paths_unused_parameters_and_scale(self):
        for multiplier in (1., 32., 65536.):
            a = torch.tensor(2., dtype=torch.float64, requires_grad=True)
            b = torch.tensor(3., dtype=torch.float64, requires_grad=True)
            c = torch.tensor(4., dtype=torch.float64, requires_grad=True)
            unused = torch.tensor(1., dtype=torch.float64, requires_grad=True)
            z = a * b
            main, auxiliary = (z * c).square(), z.square()
            routed = route_parameter_gradients(auxiliary, (b, unused))
            torch.testing.assert_close(routed, auxiliary, atol=0, rtol=0)
            actual = torch.autograd.grad(multiplier * (main + routed), (a, b, c, unused),
                                         allow_unused=True)
            # Main derivatives plus auxiliary only on b. Independent scalar
            # derivatives avoid constructing a second version of the router.
            expected = (multiplier * 2*a*b*b*c*c,
                        multiplier * (2*a*a*b*c*c + 2*a*a*b),
                        multiplier * 2*a*a*b*b*c, None)
            for x, y in zip(actual, expected):
                assert_gradient(self, x, y, exact=True)
            self.assertTrue(all(p.grad is None for p in (a, b, c, unused)))
        with self.assertRaises(ValueError):
            route_parameter_gradients(auxiliary, (b, b))

    def test_config_initialization_parameters_sharing_and_inference(self):
        cfg, ref = config(), config(REFERENCE)
        expected = deepcopy(ref.model); expected.type = 'NeckHeadROISemanticDetector'
        self.assertEqual(cfg.model, expected)
        for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                    'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator',
                    'train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'custom_hooks',
                    'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness', 'env_cfg'):
            self.assertEqual(cfg[key], ref[key], key)
        self.assertIsNone(cfg.load_from); self.assertFalse(cfg.resume)
        self.assertIsNone(cfg.model.backbone.init_cfg)
        reference = build(REFERENCE).eval(); rng = torch.get_rng_state().clone()
        model = build().eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3105704)
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
        selected = {id(p) for p in model.roi_gradient_parameters()}
        self.assertEqual(len(selected), len(model.roi_gradient_parameters()))
        self.assertFalse(selected & {id(p) for p in model.backbone.parameters()})
        self.assertTrue({id(p) for p in model.neck.parameters()} <= selected)
        head = model.bbox_head.head_module
        self.assertFalse(selected & {id(p) for p in head.reg_convs.parameters()})
        self.assertFalse(selected & {id(p) for p in head.rtm_cls.parameters()})
        for level in (1, 2):
            for i in range(2):
                self.assertIs(head.cls_convs[0][i].conv, head.cls_convs[level][i].conv)
                self.assertIs(head.reg_convs[0][i].conv, head.reg_convs[level][i].conv)
                self.assertIsNot(head.cls_convs[0][i].bn, head.cls_convs[level][i].bn)
        images, _ = data()
        with torch.no_grad():
            for a, b in zip(model._forward(images), reference._forward(images)):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_roi_only_has_zero_backbone_and_unchanged_neck_head_gradients(self):
        images, samples = data()
        model, reference = build().train(), build(REFERENCE).train()
        actual = model.loss(images, samples)
        expected = reference.loss(images, samples)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        ag = grads(actual['loss_instance_semantic'], model)
        eg = grads(expected['loss_instance_semantic'], reference)
        active_groups = dict(neck=0., cls=0., classifier=0.)
        for (name, _), a, e in zip(model.named_parameters(), ag, eg):
            if name.startswith('backbone.'):
                assert_gradient(self, a, None, name)
                continue
            assert_gradient(self, a, e, name)
            for group, prefix in [('neck', 'neck.'), ('cls', 'bbox_head.head_module.cls_convs.'),
                                   ('classifier', 'bbox_head.instance_semantic.')]:
                if name.startswith(prefix) and a is not None:
                    active_groups[group] += float(a.abs().sum())
        self.assertTrue(all(v > 0 for v in active_groups.values()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_main_gradients_unchanged_and_backbone_still_trains(self):
        images, samples = data()
        model, reference = build().train(), build(REFERENCE).train()
        actual, expected = model.loss(images, samples), reference.loss(images, samples)
        ag = grads(actual['loss_cls'] + actual['loss_bbox'], model)
        eg = grads(expected['loss_cls'] + expected['loss_bbox'], reference)
        bone_norm = 0.
        for (name, _), a, e in zip(model.named_parameters(), ag, eg):
            assert_gradient(self, a, e, name, exact=True)
            if name.startswith('backbone.') and a is not None:
                bone_norm += float(a.abs().sum())
        self.assertGreater(bone_norm, 0.)

    def test_total_gradient_single_forward_bn_and_adamw_reference_step(self):
        images, samples = data()
        model, reference = build().train(), build(REFERENCE).train()
        expected = reference.loss(images, samples)
        gm = grads(expected['loss_cls'] + expected['loss_bbox'], reference, retain_graph=True)
        ga = grads(expected['loss_instance_semantic'], reference)
        for (name, p), main, aux in zip(reference.named_parameters(), gm, ga):
            p.grad = main if name.startswith('backbone.') or aux is None else (
                aux.clone() if main is None else main + aux)
        calls = dict(backbone=0, neck=0, tower=0, assignment=0, roi=0)
        def hook(key):
            def count(*args):
                calls[key] += 1
            return count
        handles = [module.register_forward_hook(hook(key)) for key, module in [
            ('backbone', model.backbone), ('neck', model.neck),
            ('tower', model.bbox_head.head_module), ('assignment', model.bbox_head.assigner),
            ('roi', model.bbox_head.instance_semantic)]]
        actual = model.loss(images, samples)
        sum(actual.values()).backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(set(calls.values()), {1})
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(int(module.num_batches_tracked), 1, name)
        for (name, a), b in zip(model.named_parameters(), reference.parameters()):
            assert_gradient(self, a.grad, b.grad, name)
        for candidate in (model, reference):
            torch.optim.AdamW(candidate.parameters(), lr=.0005625).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=2e-7, rtol=3e-5, msg=name)

    def test_empty_gt_tensor_list_and_alpha_boundaries(self):
        for alpha in (0., .25, 1.):
            for empty, as_list in ((False, True), (True, False), (True, True)):
                images, samples = data(empty, as_list)
                model, reference = build(alpha=alpha).train(), build(REFERENCE, alpha).train()
                actual, expected = model.loss(images, samples), reference.loss(images, samples)
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                sum(actual.values()).backward()
                self.assertTrue(all(p.grad is None or p.grad.isfinite().all() for p in model.parameters()))
                if not empty:
                    model.zero_grad(set_to_none=True)
                    roi = model.loss(images, samples)['loss_instance_semantic']
                    roi.backward()
                    self.assertTrue(all(p.grad is None or not p.grad.any() for p in model.backbone.parameters()))
        model = build(); model.neck.requires_grad_(False)
        images, samples = data()
        sum(model.loss(images, samples).values()).backward()
        self.assertTrue(all(p.grad is None for p in model.neck.parameters()))

    def test_serialization_and_optimizer_accumulation(self):
        images, samples = data()
        model = build().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        for _ in range(2):
            (sum(model.loss(images, samples).values()) / 2).backward()
        optimizer.step()
        saved = deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
        restored = build()
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.0005625)
        restored_optimizer.load_state_dict(saved['optimizer'])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(restored.bbox_head.instance_semantic.feature_gradient_scale, .25)
        self.assertTrue(all(int(state['step']) == 1 for state in restored_optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
