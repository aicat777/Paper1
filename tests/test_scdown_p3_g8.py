"""The combination must preserve both established treatments independently."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from scripts.scdown_variants_probe import CONTROL_SITES, GROUP_SITES

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown_p3_g8'
P3 = 'backbone.stage2.0'
SC_SITES = (P3, *CONTROL_SITES)
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def build(method=METHOD, initialize=True):
    cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    if initialize:
        model.init_weights()
    return model


def watched_parameters(model):
    return {name: p for name, p in model.named_parameters()
            if any(name.startswith(site + '.') for site in (*SC_SITES, *GROUP_SITES))}


class SCDownP3G8Tests(unittest.TestCase):
    def test_scope_channels_untouched_paths_and_shared_head(self):
        model = build().eval()
        self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), SC_SITES)
        self.assertEqual(tuple(n.removesuffix('.conv') for n, m in model.named_modules()
                               if isinstance(m, torch.nn.Conv2d) and m.groups == 8), GROUP_SITES)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2984532)
        self.assertEqual(len(watched_parameters(model)), 45)
        for site in GROUP_SITES:
            conv = model.get_submodule(site).conv
            width = conv.in_channels
            self.assertIn(width, (48, 96, 192))
            self.assertEqual(conv.weight.shape, (width, width // 8, 3, 3))
            block = model.get_submodule(site.rpartition('.')[0])
            self.assertEqual(block.conv2.depthwise_conv.conv.groups, width)
            self.assertEqual(block.conv2.pointwise_conv.conv.groups, 1)
        self.assertEqual(model.backbone.stage1[0].conv.stride, (2, 2))
        self.assertEqual(model.backbone.stage1[1].blocks[0].conv1.conv.groups, 1)
        self.assertEqual(model.neck.downsample_layers[0].conv.groups, 1)
        for layer in model.neck.out_layers:
            self.assertEqual(layer.conv.kernel_size, (3, 3))
            self.assertEqual(layer.conv.groups, 1)
        head = model.bbox_head.head_module
        for towers in (head.cls_convs, head.reg_convs):
            for depth in range(2):
                for scale in (1, 2):
                    self.assertIs(towers[0][depth].conv, towers[scale][depth].conv)
                    self.assertIsNot(towers[0][depth].bn, towers[scale][depth].bn)
        with torch.no_grad():
            cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
        self.assertEqual([list(v.shape) for v in cls], [[1, 20, s, s] for s in (80, 40, 20)])
        self.assertEqual([list(v.shape) for v in reg], [[1, 4, s, s] for s in (80, 40, 20)])
        priors = model.bbox_head.prior_generator.grid_priors([v.shape[-2:] for v in cls])
        self.assertEqual(sum(len(p) for p in priors), 8400)

    def test_protocol_matches_both_completed_controls(self):
        def normalized(method):
            cfg = Config.fromfile(str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py')).to_dict()
            for key in ('custom_imports', 'work_dir'):
                cfg.pop(key)
            for component in ('backbone', 'neck'):
                cfg['model'][component].pop('type')
            return cfg
        actual = normalized(METHOD)
        self.assertEqual(actual, normalized('scdown_p3'))
        self.assertEqual(actual, normalized('scdown_g8'))

    def test_initialization_rng_and_existing_treatments_are_exact(self):
        for initialize in (False, True):
            for reference, exceptions in (('scdown_p3', GROUP_SITES), ('scdown_g8', (P3,))):
                with self.subTest(initialize=initialize, reference=reference):
                    control = build(reference, initialize)
                    expected_rng = torch.get_rng_state().clone()
                    model = build(initialize=initialize)
                    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
                    actual = model.state_dict()
                    for name, value in control.state_dict().items():
                        if not any(name.startswith(site + '.') for site in exceptions):
                            torch.testing.assert_close(value, actual[name], atol=0, rtol=0, msg=name)
                    if initialize:
                        self.assertEqual(model.backbone._scdown_initialization_slots, ())
                        self.assertEqual(model.neck._scdown_initialization_slots, ())
                        model.init_weights()
                        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)

    def test_each_treatment_can_be_restored_to_its_control_exactly(self):
        torch.manual_seed(31)
        images = torch.randn(2, 3, 128, 128)
        targets = dict(bboxes_labels=torch.tensor(
            [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
            dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)
        for reference, restored in (('scdown_p3', GROUP_SITES), ('scdown_g8', (P3,))):
            with self.subTest(reference=reference):
                control, model = build(reference), build()
                for site in restored:
                    parent, _, name = site.rpartition('.')
                    model.get_submodule(parent)._modules[name] = deepcopy(control.get_submodule(site))
                control.eval()
                model.eval()
                with torch.no_grad():
                    for expected, actual in zip(control._forward(images), model._forward(images)):
                        for a, b in zip(expected, actual):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                control.train()
                model.train()
                expected, actual = control.loss(images, targets), model.loss(images, targets)
                self.assertEqual(set(actual), {'loss_cls', 'loss_bbox'})
                for key in expected:
                    torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
                sum(expected.values()).backward()
                sum(actual.values()).backward()
                parameters = dict(control.named_parameters())
                for name, p in model.named_parameters():
                    self.assertEqual(p.grad is None, parameters[name].grad is None, name)
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad, parameters[name].grad, atol=0, rtol=0, msg=name)

    def test_detection_updates_bn_once_and_serialization(self):
        model = build().train()
        watched = watched_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        included = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(included), len({id(p) for p in included}))
        self.assertTrue({id(p) for p in watched.values()} <= {id(p) for p in included})
        initial = {name: p.detach().clone() for name, p in watched.items()}
        counters = {name: value.clone() for name, value in model.named_buffers()
                    if name.endswith('num_batches_tracked') and
                    any(name.startswith(site + '.') for site in (*SC_SITES, *GROUP_SITES))}
        gradients = set()
        images = torch.randn(2, 3, 128, 128)
        targets = dict(bboxes_labels=torch.tensor(
            [[0, 1, 16, 20, 28, 32], [0, 2, 20, 24, 108, 112], [1, 3, 40, 36, 80, 76]],
            dtype=torch.float32), img_metas=[dict(img_shape=(128, 128))] * 2)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(images, targets)
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            sum(losses.values()).backward()
            for name, p in watched.items():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                if p.grad.abs().max() > 0:
                    gradients.add(name)
            optimizer.step()
            for name, value in model.named_buffers():
                if name in counters:
                    self.assertEqual(int(value), int(counters[name]) + step + 1, name)
        self.assertEqual(gradients, set(watched))
        for name, p in watched.items():
            self.assertGreater(float((p.detach() - initial[name]).abs().max()), 0, name)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu')
        restored = build()
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.0005625)
        restored_optimizer.load_state_dict(saved['optimizer'])
        for name, p in restored.named_parameters():
            torch.testing.assert_close(p, dict(model.named_parameters())[name], atol=0, rtol=0)
            self.assertEqual(int(restored_optimizer.state[p]['step']), 2)
        model.eval()
        restored.eval()
        with torch.no_grad():
            for expected, actual in zip(model._forward(images), restored._forward(images)):
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        head = restored.bbox_head.head_module
        self.assertIs(head.cls_convs[0][1].conv, head.cls_convs[2][1].conv)
        self.assertIsNot(head.cls_convs[0][1].bn, head.cls_convs[2][1].bn)


if __name__ == '__main__':
    unittest.main()
