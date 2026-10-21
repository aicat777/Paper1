"""LSConv fidelity, cyclic kernel sharing, gradients, and detector integration."""
import copy
import logging
import os
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.lsconv.aggregation import small_kernel_aggregation
from custom_models.lsconv.operator import LSConv

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def coordinate_reference(x, weights):
    # Explicit shifts, independent of unfold and all Triton kernels.
    kernel = int(weights.shape[2] ** .5)
    _, channels, height, width = x.shape
    padded = F.pad(x, [kernel // 2] * 4)
    groups = torch.arange(channels, device=x.device) % weights.shape[1]
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    result = torch.zeros_like(x, dtype=dtype)
    for row in range(kernel):
        for col in range(kernel):
            values = padded[:, :, row:row + height, col:col + width].to(dtype)
            local = weights[:, groups, row * kernel + col].to(x.dtype).to(dtype)
            result = result + values * local
    return result.to(x.dtype)


class LSConvNumerics(unittest.TestCase):
    def test_signed_cyclic_groups_and_zero_padding(self):
        x = torch.arange(16 * 3 * 5).reshape(1, 16, 3, 5).double()
        weights = torch.zeros(1, 2, 9, 3, 5, dtype=torch.float64)
        weights[:, 0, 4] = 2
        weights[:, 1, 4] = -3
        scale = torch.tensor([2, -3] * 8).reshape(1, 16, 1, 1)
        torch.testing.assert_close(small_kernel_aggregation(x, weights), x * scale)
        weights.zero_()
        weights[:, :, 0] = -1
        expected = torch.zeros_like(x)
        expected[:, :, 1:, 1:] = -x[:, :, :-1, :-1]
        torch.testing.assert_close(small_kernel_aggregation(x, weights), expected)

    def test_reference_forward_backward_and_gradcheck(self):
        torch.manual_seed(9)
        x = torch.randn(1, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(1, 2, 9, 3, 5, dtype=torch.float64, requires_grad=True)
        result = small_kernel_aggregation(x, weights)
        expected = coordinate_reference(x, weights)
        torch.testing.assert_close(result, expected)
        grad = torch.randn_like(x)
        actual_grads = torch.autograd.grad(result, (x, weights), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (x, weights), grad)
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference)
        self.assertTrue(torch.autograd.gradcheck(small_kernel_aggregation,
                                               (x, weights), fast_mode=True))

    def test_complete_operator_preserves_signed_weights_and_residual(self):
        module = LSConv(48)
        module.eval()
        x = torch.randn(2, 48, 7, 9)
        with torch.no_grad():
            weights = module.lkp(x)
            self.assertEqual(weights.shape, (2, 6, 9, 7, 9))
            self.assertLess(weights.min().item(), 0)
            self.assertGreater(weights.max().item(), 0)
            expected = x + module.bn(coordinate_reference(x, weights))
            torch.testing.assert_close(module(x), expected)
        # No softmax or tiny learned residual scale is added to the source operator.
        self.assertEqual(sum(p.numel() for p in module.parameters()), 4602)

    @unittest.skipUnless(os.environ.get('PAPER1_RUN_CUDA_TESTS') == '1',
                         'Set PAPER1_RUN_CUDA_TESTS=1 for Triton comparisons')
    def test_triton_forward_and_both_gradients(self):
        torch.cuda.set_per_process_memory_fraction(.06)
        for dtype, weight_dtype in ((torch.float32, torch.float32),
                                    (torch.float16, torch.float16),
                                    (torch.float16, torch.float32)):
            for channels, groups, shape in ((16, 2, (7, 9)), (48, 6, (8, 11))):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype, channels=channels):
                    torch.manual_seed(17)
                    x = torch.randn(2, channels, *shape, device='cuda', dtype=dtype,
                                    requires_grad=True)
                    # Noncontiguous features expose accidental stride assumptions.
                    x = x.transpose(-1, -2).detach().requires_grad_()
                    weights = torch.randn(2, groups, 9, *x.shape[-2:], device='cuda',
                                          dtype=weight_dtype, requires_grad=True)
                    result = small_kernel_aggregation(x, weights)
                    expected = coordinate_reference(x, weights)
                    tolerance = 1e-5 if dtype == torch.float32 else 0.015
                    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
                    grad = torch.randn_like(x)
                    actual = torch.autograd.grad(result, (x, weights), grad, retain_graph=True)
                    reference = torch.autograd.grad(expected, (x, weights), grad)
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


class LSConvIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))

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

    def test_replacement_locations_shapes_and_parameter_count(self):
        model = self.build_model()
        self.assertEqual(model.backbone.lsconv_replacements, [
            'stage2.1.blocks.0.conv1', 'stage3.1.blocks.0.conv1', 'stage4.2.blocks.0.conv1'])
        self.assertEqual(sum(isinstance(m, LSConv) for m in model.modules()), 3)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4517130)
        model.eval()
        with torch.no_grad():
            features = model.backbone(torch.randn(1, 3, 640, 640))
            cls, reg = model.bbox_head(model.neck(features))
        self.assertEqual([tuple(t.shape) for t in features], [
            (1, 96, 80, 80), (1, 192, 40, 40), (1, 384, 20, 20)])
        self.assertEqual([tuple(t.shape) for t in cls], [
            (1, 20, 80, 80), (1, 20, 40, 40), (1, 20, 20, 20)])
        self.assertEqual([tuple(t.shape) for t in reg], [
            (1, 4, 80, 80), (1, 4, 40, 40), (1, 4, 20, 20)])

    def test_no_replacements_matches_official_backbone(self):
        config = copy.deepcopy(self.cfg.model.backbone)
        config.lsconv_stages = ()
        ours = MODELS.build(config)
        config.type = 'CSPNeXt'
        config.pop('lsconv_stages')
        baseline = MODELS.build(config)
        ours.load_state_dict(baseline.state_dict(), strict=True)
        ours.eval(); baseline.eval()
        with torch.no_grad():
            image = torch.randn(1, 3, 129, 161)
            for actual, expected in zip(ours(image), baseline(image)):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_new_parameters_receive_detection_gradients_and_update(self):
        model = self.build_model()
        model.train()
        watched = {n: p for n, p in model.named_parameters() if '.conv1.0.' in n}
        self.assertEqual(len(watched), 45)
        before = {n: p.detach().clone() for n, p in watched.items()}
        with torch.no_grad():
            for conv in model.bbox_head.head_module.rtm_reg:
                conv.bias.fill_(.4)
        samples = dict(
            bboxes_labels=torch.tensor([[0, 0, 8, 8, 32, 24], [1, 18, 56, 48, 80, 96]],
                                       dtype=torch.float32),
            img_metas=[dict(img_shape=(128, 128)) for _ in range(2)])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        losses = model.loss(torch.randn(2, 3, 128, 128), samples)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, p in watched.items():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        optimizer.step()
        for name, p in watched.items():
            self.assertGreater((p - before[name]).abs().max().item(), 0, name)

    def test_matches_baseline_batch36_protocol(self):
        baseline = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        for key in ('train_pipeline', 'train_pipeline_stage2', 'train_cfg',
                    'val_dataloader', 'val_cfg', 'val_evaluator', 'test_dataloader',
                    'test_evaluator', 'custom_hooks', 'default_hooks', 'randomness'):
            self.assertEqual(self.cfg[key], baseline[key], key)
        self.assertEqual(self.cfg.train_dataloader.dataset, baseline.train_dataloader.dataset)
        self.assertEqual(self.cfg.model.neck, baseline.model.neck)
        self.assertEqual(self.cfg.model.bbox_head, baseline.model.bbox_head)
        self.assertEqual(self.cfg.train_dataloader.batch_size, 36)
        self.assertAlmostEqual(self.cfg.optim_wrapper.optimizer.lr, .0005625)
        self.assertEqual(self.cfg.optim_wrapper.get('accumulative_counts', 1), 1)
        self.assertEqual(self.cfg.param_scheduler[0].end, 1000)
        self.assertIsNone(self.cfg.model.backbone.init_cfg)
        self.assertIsNone(self.cfg.load_from)
        self.assertFalse(self.cfg.resume)


if __name__ == '__main__':
    unittest.main()
