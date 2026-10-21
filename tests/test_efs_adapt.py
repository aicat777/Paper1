"""Check preserved lateral content, head ownership and real detection gradients."""
from copy import deepcopy
import io
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.models.layers.ema import ExpMomentumEMA
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


class EFSAdaptIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(ROOT / 'configs/efs_adapt/rtmdet_tiny_dior_640.py'))
        cls.efs = Config.fromfile(str(ROOT / 'configs/efs/rtmdet_tiny_dior_640.py'))
        cls.baseline = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))

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

    def build_head(self):
        return MODELS.build(deepcopy(self.cfg.model.bbox_head.head_module))

    def assert_sharing(self, head):
        for towers in (head.cls_convs, head.reg_convs):
            for idx in range(head.stacked_convs):
                self.assertIsNot(towers[0][idx].conv, towers[1][idx].conv)
                self.assertIsNot(towers[0][idx].conv.weight, towers[1][idx].conv.weight)
                self.assertNotEqual(towers[0][idx].conv.weight.data_ptr(),
                                    towers[1][idx].conv.weight.data_ptr())
                for level in (2, 3):
                    self.assertIs(towers[1][idx].conv, towers[level][idx].conv)
                self.assertEqual(len({id(towers[n][idx].norm) for n in range(4)}), 4)
        self.assertEqual(len({id(layer.weight) for layer in head.rtm_cls}), 4)
        self.assertEqual(len({id(layer.weight) for layer in head.rtm_reg}), 4)

    def test_protocol_and_exact_scope(self):
        for key in ('img_scale', 'train_cfg', 'train_pipeline', 'train_pipeline_stage2',
                    'train_dataloader', 'val_dataloader', 'val_cfg', 'val_evaluator',
                    'test_dataloader', 'test_evaluator', 'default_hooks', 'randomness',
                    'optim_wrapper', 'param_scheduler', 'custom_hooks', 'load_from', 'resume'):
            self.assertEqual(self.cfg[key], self.efs[key], key)
        self.assertEqual(self.cfg.model.backbone, self.efs.model.backbone)
        self.assertEqual(self.cfg.model.train_cfg, self.efs.model.train_cfg)
        self.assertIsNone(self.cfg.model.backbone.init_cfg)
        self.assertIsNone(self.cfg.load_from)
        self.assertFalse(self.cfg.resume)
        self.assertEqual(self.cfg.custom_hooks[0].gamma, 8000)
        neck = deepcopy(self.cfg.model.neck)
        neck.type = self.efs.model.neck.type
        self.assertEqual(neck, self.efs.model.neck)
        head = deepcopy(self.cfg.model.bbox_head)
        head.head_module.type = self.efs.model.bbox_head.head_module.type
        self.assertEqual(head, self.efs.model.bbox_head)

    def test_f3_matches_original_fpn_and_is_preserved_in_pan(self):
        baseline = MODELS.build(deepcopy(self.baseline.model.neck))
        neck = MODELS.build(deepcopy(self.cfg.model.neck))
        baseline.eval()
        neck.eval()
        neck.reduce_layers[3].load_state_dict(baseline.reduce_layers[2].state_dict())
        neck.top_down_layers[0].load_state_dict(baseline.top_down_layers[0].state_dict())
        neck.top_down_layers[1].load_state_dict(baseline.top_down_layers[1].state_dict())
        seen = {}
        handles = [
            baseline.top_down_layers[1].register_forward_hook(
                lambda m, a, out: seen.update(baseline_f3=out.detach())),
            neck.top_down_layers[1].register_forward_hook(
                lambda m, a, out: seen.update(f3=out.detach())),
            neck.upsample_layers[2].register_forward_pre_hook(
                lambda m, a: seen.update(branch_input=a[0].detach())),
            neck.upsample_layers[2].register_forward_hook(
                lambda m, a, out: seen.update(branch_output=out.detach())),
            neck.bottom_up_layers[0].register_forward_pre_hook(
                lambda m, a: seen.update(pan_input=a[0].detach()))]
        features = [torch.randn(2, channels, size, size)
                    for channels, size in ((48, 160), (96, 80), (192, 40), (384, 20))]
        with torch.no_grad():
            baseline(features[1:])
            outputs = neck(features)
        for handle in handles:
            handle.remove()
        self.assertEqual(tuple(seen['f3'].shape), (2, 96, 80, 80))
        torch.testing.assert_close(seen['f3'], seen['baseline_f3'], atol=0, rtol=0)
        torch.testing.assert_close(seen['branch_input'], seen['f3'], atol=0, rtol=0)
        self.assertEqual(tuple(seen['branch_output'].shape), (2, 48, 160, 160))
        self.assertEqual(tuple(seen['pan_input'].shape), (2, 144, 80, 80))
        torch.testing.assert_close(seen['pan_input'][:, 48:], seen['f3'], atol=0, rtol=0)
        self.assertEqual([tuple(value.shape) for value in outputs],
                         [(2, 96, s, s) for s in (160, 80, 40, 20)])

    def test_head_weight_ownership_and_gradients(self):
        head = self.build_head().train()
        self.assert_sharing(head)
        features = [torch.randn(2, 96, size, size) for size in (32, 16, 8, 4)]
        cls, reg = head(features)
        (cls[0].square().mean() + reg[0].square().mean()).backward()
        for towers in (head.cls_convs, head.reg_convs):
            for idx in range(head.stacked_convs):
                self.assertGreater(towers[0][idx].conv.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(towers[1][idx].conv.weight.grad)
        head.zero_grad(set_to_none=True)
        cls, reg = head(features)
        sum(value.square().mean() for value in (*cls[1:], *reg[1:])).backward()
        for towers in (head.cls_convs, head.reg_convs):
            for idx in range(head.stacked_convs):
                self.assertIsNone(towers[0][idx].conv.weight.grad)
                self.assertGreater(towers[1][idx].conv.weight.grad.abs().sum().item(), 0)

    def test_full_shapes_priors_and_parameter_budget(self):
        model = self.build_model().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5342340)
        with torch.no_grad():
            cls, reg = model.bbox_head.head_module(model.extract_feat(torch.randn(1, 3, 640, 640)))
        self.assertEqual([tuple(value.shape) for value in cls],
                         [(1, 20, s, s) for s in (160, 80, 40, 20)])
        self.assertEqual([tuple(value.shape) for value in reg],
                         [(1, 4, s, s) for s in (160, 80, 40, 20)])
        priors = model.bbox_head.prior_generator.grid_priors(
            [value.shape[-2:] for value in cls], with_stride=True)
        self.assertEqual(sum(len(value) for value in priors), 34000)
        for prior, stride in zip(priors, (4, 8, 16, 32)):
            torch.testing.assert_close(prior[:, 2:], torch.full_like(prior[:, 2:], stride))

    def test_tiny_box_updates_both_adaptations(self):
        model = self.build_model().train()
        with torch.no_grad():
            for layer in model.bbox_head.head_module.rtm_reg:
                layer.bias.fill_(.4)
        data = dict(
            bboxes_labels=torch.tensor([[n, 0, 2, 2, 6, 6] for n in range(2)],
                                       dtype=torch.float32),
            img_metas=[dict(img_shape=(128, 128)) for _ in range(2)])
        losses = model.loss(torch.randn(2, 3, 128, 128), data)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(losses['loss_bbox'].item(), 0)
        loss.backward()
        head = model.bbox_head.head_module
        watched = dict(projection=model.neck.upsample_layers[2][0].conv.weight,
                       pan=model.neck.bottom_up_layers[0].main_conv.conv.weight,
                       p2_cls=head.cls_convs[0][0].conv.weight,
                       p2_reg=head.reg_convs[0][0].conv.weight)
        for name, parameter in watched.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        initial = {name: value.detach().clone() for name, value in watched.items()}
        torch.optim.AdamW(model.parameters(), lr=.0005625).step()
        for name, parameter in watched.items():
            self.assertGreater((parameter - initial[name]).abs().max().item(), 0, name)

    def test_checkpoint_and_ema_preserve_sharing(self):
        head = self.build_head().eval()
        ema = ExpMomentumEMA(head, momentum=.0002, gamma=8000,
                             interval=4, update_buffers=True)
        ema.update_parameters(head)
        with torch.no_grad():
            head.cls_convs[0][0].conv.weight.add_(.1)
        for _ in range(4):
            ema.update_parameters(head)
        saved = io.BytesIO()
        torch.save(ema.module.state_dict(), saved)
        saved.seek(0)
        restored = self.build_head().eval()
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        self.assert_sharing(ema.module)
        self.assert_sharing(restored)
        features = [torch.randn(2, 96, size, size) for size in (32, 16, 8, 4)]
        with torch.no_grad():
            for actual, expected in zip(restored(features), ema.module.eval()(features)):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
