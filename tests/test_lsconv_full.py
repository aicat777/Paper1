"""Expanded LSConv: integration, scale sharing, gradients, and scratch protocol."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
import torch
from torch import nn

from custom_models.lsconv_full.operator import ConvBN, LSConv

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def check_sharing(head):
    for towers in (head.cls_convs, head.reg_convs):
        for depth in range(head.stacked_convs):
            for name in ('cv1', 'cv2', 'cv3'):
                layers = [getattr(towers[level][depth][0].lkp, name) for level in range(3)]
                assert len({id(layer.c.weight) for layer in layers}) == 1
                assert len({id(layer.bn) for layer in layers}) == 3
            operators = [towers[level][depth][0] for level in range(3)]
            assert len({id(op.lkp.cv4.weight) for op in operators}) == 1
            assert len({id(op.lkp.norm) for op in operators}) == 3
            assert len({id(op.bn) for op in operators}) == 3
            projections = [towers[level][depth][1] for level in range(3)]
            assert len({id(layer.c.weight) for layer in projections}) == 1
            assert len({id(layer.bn) for layer in projections}) == 3
    cls_ids = {id(p) for p in head.cls_convs.parameters()}
    reg_ids = {id(p) for p in head.reg_convs.parameters()}
    assert not cls_ids & reg_ids


def watched_parameters(model):
    identities = {id(p) for layer in model.modules()
                  if isinstance(layer, (LSConv, ConvBN)) for p in layer.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in identities}


class FullLSConvIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(ROOT / 'configs/lsconv_full/rtmdet_tiny_dior_640.py'))

    def build_model(self):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(self.cfg.model))
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            model.init_weights()
        finally:
            logging.disable(previous)
        return model

    def test_scope_channels_predictors_and_runtime_calls(self):
        model = self.build_model().eval()
        self.assertEqual(len(model.backbone.lsconv_replacements), 4)
        self.assertEqual(len(model.neck.lsconv_replacements), 7)
        self.assertEqual(len(model.bbox_head.head_module.lsconv_replacements), 12)
        self.assertEqual(sum(isinstance(layer, LSConv) for layer in model.modules()), 23)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3385305)
        self.assertEqual(len(watched_parameters(model)), 342)
        calls = []
        handles = [layer.register_forward_hook(lambda m, a, out: calls.append(tuple(out.shape)))
                   for layer in model.modules() if isinstance(layer, LSConv)]
        with torch.no_grad():
            cls, reg = model._forward(torch.randn(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(calls), 23)
        self.assertEqual([tuple(t.shape) for t in cls], [(1, 20, s, s) for s in (80, 40, 20)])
        self.assertEqual([tuple(t.shape) for t in reg], [(1, 4, s, s) for s in (80, 40, 20)])
        for tower in (model.bbox_head.head_module.rtm_cls, model.bbox_head.head_module.rtm_reg):
            self.assertTrue(all(isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1)
                                for layer in tower))
        self.assertTrue(all(torch.allclose(layer.bias, torch.full_like(layer.bias, -4.59511985))
                            for layer in model.bbox_head.head_module.rtm_cls))

    def test_shared_convolutions_independent_norms_and_roundtrip(self):
        model = self.build_model()
        check_sharing(model.bbox_head.head_module)
        clone = deepcopy(model)
        check_sharing(clone.bbox_head.head_module)
        memory = io.BytesIO()
        torch.save(model.state_dict(), memory)
        memory.seek(0)
        restored = self.build_model()
        restored.load_state_dict(torch.load(memory, map_location='cpu'), strict=True)
        check_sharing(restored.bbox_head.head_module)
        model.eval(); restored.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 128, 160)
            for actual, expected in zip(model._forward(image), restored._forward(image)):
                for first, second in zip(actual, expected):
                    torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_shared_gradient_is_sum_of_scale_gradients(self):
        head = self.build_model().bbox_head.head_module.eval()
        independent = deepcopy(head)
        # Separate the scale towers while preserving all parameter values.
        independent.cls_convs = nn.ModuleList([deepcopy(tower) for tower in head.cls_convs])
        independent.reg_convs = nn.ModuleList([deepcopy(tower) for tower in head.reg_convs])
        feats = tuple(torch.randn(2, 96, side, side) for side in (8, 4, 2))
        for current in (head, independent):
            outputs = current(feats)
            sum(t.square().sum() for group in outputs for t in group).backward()
        for shared_towers, separate_towers in ((head.cls_convs, independent.cls_convs),
                                               (head.reg_convs, independent.reg_convs)):
            for depth in range(2):
                first = shared_towers[0][depth]
                references = [tower[depth] for tower in separate_towers]
                for name in ('cv1', 'cv2', 'cv3'):
                    actual = getattr(first[0].lkp, name).c.weight.grad
                    expected = sum(getattr(layer[0].lkp, name).c.weight.grad for layer in references)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
                torch.testing.assert_close(first[1].c.weight.grad,
                                           sum(layer[1].c.weight.grad for layer in references),
                                           atol=1e-6, rtol=1e-4)

    def test_all_new_parameters_receive_detection_gradients_and_update(self):
        model = self.build_model().train()
        watched = watched_parameters(model)
        initial = {name: p.detach().clone() for name, p in watched.items()}
        with torch.no_grad():
            for layer in model.bbox_head.head_module.rtm_reg:
                layer.bias.fill_(2.)
        # Sparse tiny GT can leave a scale-specific regression BN/GN inactive.
        # Cover all three scales when asserting every new parameter is active.
        boxes = [[image, label, *coordinates] for image in range(2)
                 for label, coordinates in enumerate(((32, 32, 48, 48),
                                                       (64, 64, 144, 144),
                                                       (8, 8, 248, 248)))]
        samples = dict(bboxes_labels=torch.tensor(boxes, dtype=torch.float32),
                       img_metas=[dict(img_shape=(256, 256)) for _ in range(2)])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        loss = sum(model.loss(torch.randn(2, 3, 256, 256), samples).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, p in watched.items():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().max()), 0, name)
        optimizer.step()
        for name, p in watched.items():
            self.assertGreater(float((p.detach() - initial[name]).abs().max()), 0, name)

    def test_scratch_protocol_and_effective_batch(self):
        baseline = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        for key in ('train_pipeline', 'train_pipeline_stage2', 'train_cfg',
                    'val_dataloader', 'val_cfg', 'val_evaluator', 'default_hooks', 'randomness'):
            self.assertEqual(self.cfg[key], baseline[key], key)
        self.assertEqual(self.cfg.train_dataloader.dataset, baseline.train_dataloader.dataset)
        self.assertEqual(self.cfg.model.train_cfg, baseline.model.train_cfg)
        self.assertEqual(self.cfg.model.bbox_head.prior_generator, baseline.model.bbox_head.prior_generator)
        self.assertEqual(self.cfg.model.bbox_head.loss_cls, baseline.model.bbox_head.loss_cls)
        self.assertEqual(self.cfg.model.bbox_head.loss_bbox, baseline.model.bbox_head.loss_bbox)
        self.assertEqual(self.cfg.train_dataloader.batch_size, 12)
        self.assertEqual(self.cfg.optim_wrapper.accumulative_counts, 3)
        self.assertAlmostEqual(self.cfg.optim_wrapper.optimizer.lr, .0005625)
        self.assertEqual(self.cfg.param_scheduler[0].end, 3000)
        self.assertEqual(self.cfg.custom_hooks[0].gamma, 6000)
        self.assertEqual(self.cfg.custom_hooks[0].interval, 3)
        self.assertEqual(self.cfg.custom_hooks[1], baseline.custom_hooks[1])
        self.assertIsNone(self.cfg.model.backbone.init_cfg)
        self.assertIsNone(self.cfg.load_from)
        self.assertFalse(self.cfg.resume)


if __name__ == '__main__':
    unittest.main()
