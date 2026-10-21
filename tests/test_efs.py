"""Check EFS scale alignment and that P2 participates in detection training."""
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()


class EFSIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = Config.fromfile(str(ROOT / 'configs/efs/rtmdet_tiny_dior_640.py'))

    def build_model(self):
        torch.manual_seed(42)
        model = MODELS.build(self.cfg.model)
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            model.init_weights()
        finally:
            logging.disable(previous)
        return model

    def test_scratch_data_and_evaluation_protocol(self):
        base = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        for key in ('img_scale', 'train_cfg', 'train_pipeline',
                    'train_pipeline_stage2', 'val_dataloader', 'val_cfg',
                    'val_evaluator', 'test_dataloader', 'test_evaluator',
                    'default_hooks', 'randomness'):
            self.assertEqual(self.cfg[key], base[key], key)
        self.assertEqual(self.cfg.train_dataloader.dataset, base.train_dataloader.dataset)
        self.assertIsNone(self.cfg.model.backbone.init_cfg)
        self.assertIsNone(self.cfg.load_from)
        self.assertFalse(self.cfg.resume)
        self.assertEqual(self.cfg.model.neck.type, 'CSPNeXtPAFPN')
        self.assertEqual(self.cfg.model.bbox_head.type, 'RTMDetHead')
        self.assertEqual(self.cfg.custom_hooks[1], base.custom_hooks[1])
        accumulation = self.cfg.optim_wrapper.accumulative_counts
        self.assertEqual(self.cfg.train_dataloader.batch_size * accumulation, 36)
        self.assertAlmostEqual(self.cfg.optim_wrapper.optimizer.lr, 0.0005625)
        self.assertEqual(self.cfg.param_scheduler[0].end, 1000 * accumulation)
        self.assertEqual(self.cfg.custom_hooks[0].interval, accumulation)
        # Use the actual MMYOLO EMA defaults and avg_func, not a duplicate of
        # the configured number: MMDetection's EMA has different defaults.
        baseline_ema = MODELS.build(dict(
            type=base.custom_hooks[0].ema_type, model=torch.nn.Linear(1, 1),
            momentum=base.custom_hooks[0].momentum))
        efs_ema = MODELS.build(dict(
            type=self.cfg.custom_hooks[0].ema_type, model=torch.nn.Linear(1, 1),
            momentum=self.cfg.custom_hooks[0].momentum,
            gamma=self.cfg.custom_hooks[0].gamma, interval=accumulation))
        self.assertEqual(efs_ema.gamma, baseline_ema.gamma * accumulation)
        for optimizer_steps in (1, 100, 1000, 4890, 50000):
            base_weight = torch.zeros(1, dtype=torch.float64)
            efs_weight = torch.zeros_like(base_weight)
            source = torch.ones_like(base_weight)
            baseline_ema.avg_func(base_weight, source, optimizer_steps)
            efs_ema.avg_func(efs_weight, source, optimizer_steps * accumulation)
            torch.testing.assert_close(efs_weight, base_weight, rtol=4e-4, atol=1e-7)

    def test_four_scales_align_with_priors_at_640(self):
        model = self.build_model()
        model.eval()
        with torch.no_grad():
            backbone = model.backbone(torch.zeros(1, 3, 640, 640))
            neck = model.neck(backbone)
            cls, reg = model.bbox_head(neck)
        sizes = (160, 80, 40, 20)
        self.assertEqual([tuple(t.shape) for t in backbone], [
            (1, channels, size, size) for channels, size
            in zip((48, 96, 192, 384), sizes)])
        self.assertEqual([tuple(t.shape) for t in neck], [
            (1, 96, size, size) for size in sizes])
        self.assertEqual([tuple(t.shape) for t in cls], [
            (1, 20, size, size) for size in sizes])
        self.assertEqual([tuple(t.shape) for t in reg], [
            (1, 4, size, size) for size in sizes])
        priors = model.bbox_head.prior_generator.grid_priors(
            [(size, size) for size in sizes], device='cpu', with_stride=True)
        self.assertEqual(sum(len(p) for p in priors), 34000)
        for p, stride in zip(priors, (4, 8, 16, 32)):
            torch.testing.assert_close(p[:, 2:], torch.full_like(p[:, 2:], stride))

    def test_tiny_box_trains_p2_regression_and_feature_path(self):
        model = self.build_model()
        model.train()
        # The box contains a stride-4 point (4,4), with no coarser point inside.
        # Positive initial distances avoid relying on random invalid boxes.
        with torch.no_grad():
            for layer in model.bbox_head.head_module.rtm_reg:
                layer.bias.fill_(0.4)
        data = dict(
            bboxes_labels=torch.tensor([[0, 0, 2, 2, 6, 6], [1, 0, 2, 2, 6, 6]],
                                       dtype=torch.float32),
            img_metas=[dict(img_shape=(128, 128)) for _ in range(2)])
        losses = model.loss(torch.randn(2, 3, 128, 128), data)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(losses['loss_bbox'].item(), 0)
        loss.backward()
        modules = dict(
            p2_output=model.neck.out_layers[0],
            p2_top_down=model.neck.top_down_layers[-1],
            p2_cls=model.bbox_head.head_module.rtm_cls[0],
            p2_reg=model.bbox_head.head_module.rtm_reg[0])
        for name, module in modules.items():
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        before = model.bbox_head.head_module.rtm_reg[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005625)
        optimizer.step()
        self.assertGreater((before - model.bbox_head.head_module.rtm_reg[0].weight)
                           .abs().max().item(), 0)


if __name__ == '__main__':
    unittest.main()
