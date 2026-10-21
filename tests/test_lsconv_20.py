"""20-site scope, faithful initialization, sharing and actual detection updates."""
from copy import deepcopy
import hashlib
import io
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from mmengine.config import Config
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmyolo.models.dense_heads.rtmdet_head import RTMDetSepBNHeadModule
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

import custom_models.lsconv_20
from custom_models.lsconv_20.operator import ConvBN, LSConv

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def config():
    return Config.fromfile(str(ROOT / 'configs/lsconv_20/rtmdet_tiny_dior_640.py'))


def initialize(model):
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        model.init_weights()
    finally:
        logging.disable(previous)


def build(initialize=True):
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(config().model))
    if initialize:
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            model.init_weights()
        finally:
            logging.disable(previous)
    return model


def watched(model):
    ids = {id(p) for module in model.modules() if isinstance(module, (LSConv, ConvBN))
           for p in module.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def check_sharing(head, shared=True):
    for towers in (head.cls_convs, head.reg_convs):
        for depth in range(head.stacked_convs):
            operators = [tower[depth][0] for tower in towers]
            for part in ('cv1', 'cv2', 'cv3'):
                layers = [getattr(op.lkp, part) for op in operators]
                assert len({id(layer.c) for layer in layers}) == (1 if shared else 3)
                assert len({id(layer.bn) for layer in layers}) == 3
            assert len({id(op.lkp.cv4) for op in operators}) == (1 if shared else 3)
            assert len({id(op.lkp.norm) for op in operators}) == 3
            assert len({id(op.bn) for op in operators}) == 3
            projections = [tower[depth][1] for tower in towers]
            assert len({id(layer.c) for layer in projections}) == (1 if shared else 3)
            assert len({id(layer.bn) for layer in projections}) == 3
    assert not ({id(p) for p in head.cls_convs.parameters()}
                & {id(p) for p in head.reg_convs.parameters()})


class LSConv20Tests(unittest.TestCase):
    def test_verified_core_is_unchanged_and_independent(self):
        for name in ('operator.py', 'aggregation.py'):
            actual = ROOT / 'custom_models/lsconv_20' / name
            previous = ROOT / 'custom_models/lsconv' / name
            self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(),
                             hashlib.sha256(previous.read_bytes()).hexdigest())
        for path in (ROOT / 'custom_models/lsconv_20').glob('*.py'):
            self.assertNotIn('custom_models.lsconv_full', path.read_text())
            self.assertNotIn('custom_models.lsconv_csp', path.read_text())

    def test_exact_scope_outputs_and_unchanged_csp_conv2(self):
        model = build().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3855645)
        self.assertEqual(len(watched(model)), 288)
        self.assertEqual(len(model.backbone.lsconv_replacements), 4)
        self.assertEqual(len(model.neck.lsconv_replacements), 4)
        self.assertEqual(len(model.bbox_head.head_module.lsconv_replacements), 12)
        cores = [(name, module) for name, module in model.named_modules() if isinstance(module, LSConv)]
        self.assertEqual(len(cores), 20)
        for prefix, count in [('backbone.', 4), ('neck.', 4), ('bbox_head.', 12)]:
            self.assertEqual(sum(name.startswith(prefix) for name, _ in cores), count)
        for name, module in cores:
            self.assertNotIn('out_layers.', name)
        blocks = [m for m in model.modules() if isinstance(m, CSPNeXtBlock)]
        self.assertEqual(len(blocks), 8)
        for block in blocks:
            self.assertIsInstance(block.conv1[0], LSConv)
            self.assertIsInstance(block.conv1[1], nn.SiLU)
            self.assertEqual(block.conv2.depthwise_conv.conv.kernel_size, (5, 5))
        self.assertEqual([(layer.conv.in_channels, layer.conv.out_channels, layer.conv.kernel_size)
                          for layer in model.neck.out_layers],
                         [(96, 96, (3, 3)), (192, 96, (3, 3)), (384, 96, (3, 3))])
        calls = []
        handles = [m.register_forward_hook(lambda m, args, y, name=n: calls.append((name, tuple(y.shape))))
                   for n, m in cores]
        with torch.no_grad():
            cls, reg = model._forward(torch.randn(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(calls), 20)
        self.assertEqual([tuple(x.shape) for x in cls], [(1, 20, s, s) for s in (80, 40, 20)])
        self.assertEqual([tuple(x.shape) for x in reg], [(1, 4, s, s) for s in (80, 40, 20)])
        for side in (80, 40, 20):
            self.assertEqual(sum(name.startswith('bbox_head.') and shape == (1, 96, side, side)
                                 for name, shape in calls), 4)
        self.assertEqual(sum(s * s for s in (80, 40, 20)), 8400)

    def test_twenty_sites_equal_legacy_twenty_three_with_original_outputs(self):
        # A numerical structural control: equalize states to remove the
        # deliberately different initialization, then compare complete outputs.
        import custom_models.lsconv_full
        previous_cfg = Config.fromfile(str(ROOT / 'configs/lsconv_full/rtmdet_tiny_dior_640.py'))
        baseline_cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        previous = MODELS.build(deepcopy(previous_cfg.model))
        baseline = MODELS.build(deepcopy(baseline_cfg.model))
        previous.neck.out_layers = deepcopy(baseline.neck.out_layers)
        model = build().eval()
        previous.load_state_dict(model.state_dict(), strict=True)
        previous.eval()
        with torch.no_grad():
            image = torch.randn(2, 3, 128, 160)
            for actual, expected in zip(model._forward(image), previous._forward(image)):
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_root_initialization_preserves_all_ls_states_and_original_prediction_policy(self):
        model = build(initialize=False)
        before = {name: {key: value.clone() for key, value in layer.state_dict().items()}
                  for name, layer in model.named_modules() if isinstance(layer, LSConv)}
        initialize(model)
        for name, expected in before.items():
            for key, value in model.get_submodule(name).state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0, msg=name + '.' + key)
        head = model.bbox_head.head_module
        for tower in (head.cls_convs, head.reg_convs):
            for block in tower[0]:
                self.assertAlmostEqual(float(block[1].c.weight.std()), .01, delta=.001)
                self.assertGreater(float(block[0].lkp.cv1.c.weight.std()), .04)
        for layer in head.rtm_cls:
            torch.testing.assert_close(layer.bias.sigmoid(), torch.full_like(layer.bias, .01))
        for layer in head.rtm_reg:
            self.assertEqual(int(torch.count_nonzero(layer.bias)), 0)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        model.neck.init_weights(); head.init_weights()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0, msg=name)

    def test_initialization_exception_restores_modules_order_and_aliases(self):
        for name, parent_class in [('neck', CSPNeXtPAFPN),
                                   ('bbox_head.head_module', RTMDetSepBNHeadModule)]:
            model = build(initialize=False)
            child = model.get_submodule(name)
            order = [(path, id(layer)) for path, layer in child.named_modules()]
            def fail():
                self.assertFalse(any(isinstance(layer, LSConv) for layer in child.modules()))
                raise RuntimeError('intentional init failure')
            with patch.object(parent_class, 'init_weights', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'intentional init failure'):
                    child.init_weights()
            self.assertEqual(order, [(path, id(layer)) for path, layer in child.named_modules()])
            check_sharing(model.bbox_head.head_module)

    def test_scale_sharing_task_boundaries_and_unshared_option(self):
        model = build()
        head = model.bbox_head.head_module
        check_sharing(head)
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in head.modules()), 60)
        self.assertEqual(sum(isinstance(m, nn.GroupNorm) for m in head.modules()), 12)
        settings = deepcopy(config().model.bbox_head.head_module)
        settings.share_conv = False
        check_sharing(MODELS.build(settings), shared=False)

    def test_shared_gradients_equal_sum_over_independent_scales(self):
        head = build().bbox_head.head_module.eval()
        separate = deepcopy(head)
        separate.cls_convs = nn.ModuleList([deepcopy(tower) for tower in head.cls_convs])
        separate.reg_convs = nn.ModuleList([deepcopy(tower) for tower in head.reg_convs])
        features = tuple(torch.randn(2, 96, h, w) for h, w in [(8, 10), (4, 5), (2, 3)])
        for current in (head, separate):
            sum(y.square().sum() for group in current(features) for y in group).backward()
        for group in ('cls_convs', 'reg_convs'):
            actual_towers, expected_towers = getattr(head, group), getattr(separate, group)
            for depth in range(2):
                first = actual_towers[0][depth]
                others = [tower[depth] for tower in expected_towers]
                for part in ('cv1', 'cv2', 'cv3'):
                    torch.testing.assert_close(getattr(first[0].lkp, part).c.weight.grad,
                        sum(getattr(block[0].lkp, part).c.weight.grad for block in others),
                        atol=1e-5, rtol=3e-4)
                for parameter in ('weight', 'bias'):
                    torch.testing.assert_close(getattr(first[0].lkp.cv4, parameter).grad,
                        sum(getattr(block[0].lkp.cv4, parameter).grad for block in others),
                        atol=1e-5, rtol=3e-4)
                torch.testing.assert_close(first[1].c.weight.grad,
                    sum(block[1].c.weight.grad for block in others), atol=1e-5, rtol=3e-4)

    def test_detection_updates_and_serialization_keep_sharing(self):
        model = build().train()
        parameters = watched(model)
        initial = {name: p.detach().clone() for name, p in parameters.items()}
        gradients = {}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        all_parameters = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(all_parameters), len({id(p) for p in all_parameters}))
        self.assertTrue({id(p) for p in parameters.values()} <= {id(p) for p in all_parameters})
        # Include tiny-to-large GT centered on multiple scales' prior locations.
        # Prediction biases and model initialization are left unchanged.
        rows = []
        for image in range(2):
            for center in (32, 64, 128, 192, 224):
                for label, side in enumerate((1, 2, 4, 8, 16, 64, 192)):
                    left, right = max(0, center - side / 2), min(256, center + side / 2)
                    rows.append([image, label, left, left, right, right])
        samples = dict(bboxes_labels=torch.tensor(rows, dtype=torch.float32),
                       img_metas=[dict(img_shape=(256, 256)) for _ in range(2)])
        for step in range(3):
            optimizer.zero_grad()
            losses = model.loss(torch.randn(2, 3, 256, 256), samples)
            loss = sum(losses.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, p in parameters.items():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                gradients[name] = max(gradients.get(name, 0), float(p.grad.abs().max()))
            optimizer.step()
        for name, p in parameters.items():
            self.assertGreater(gradients[name], 0, name)
            self.assertGreater(float((p.detach() - initial[name]).abs().max()), 0, name)
        memory = io.BytesIO()
        torch.save(model.state_dict(), memory); memory.seek(0)
        restored = build()
        restored.load_state_dict(torch.load(memory, map_location='cpu'), strict=True)
        check_sharing(restored.bbox_head.head_module)
        model.eval(); restored.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 128, 160)
            for actual, expected in zip(model._forward(image), restored._forward(image)):
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_protocol_matches_completed_real_batch36_experiment(self):
        cfg = config()
        baseline = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
        for key in ('train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'train_cfg',
                    'train_dataloader', 'val_dataloader', 'val_cfg', 'val_evaluator',
                    'test_dataloader', 'test_cfg', 'test_evaluator', 'optim_wrapper',
                    'param_scheduler', 'custom_hooks', 'default_hooks', 'randomness'):
            self.assertEqual(cfg[key], baseline[key], key)
        settings, expected = deepcopy(cfg.model), deepcopy(baseline.model)
        for section in ('backbone', 'neck'):
            settings[section].pop('type'); expected[section].pop('type')
        expected.backbone.pop('lsconv_stages')
        settings.bbox_head.head_module.pop('type'); expected.bbox_head.head_module.pop('type')
        self.assertEqual(settings, expected)
        self.assertEqual(cfg.train_dataloader.batch_size, 36)
        self.assertEqual(cfg.optim_wrapper.get('accumulative_counts', 1), 1)
        self.assertIsNone(cfg.load_from)
        self.assertIsNone(cfg.model.backbone.init_cfg)
        self.assertFalse(cfg.resume)


if __name__ == '__main__':
    unittest.main()
