"""FDW weight reconstruction, independent mixtures, gradients and integration."""
import copy
import io
import logging
import os
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmyolo.models.layers.ema import ExpMomentumEMA
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.fdw.operator import FDWConv2d

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def explicit_mixture(module, x, attention):
    # Construct every individual kernel, then sum kernels and convolve each image.
    # Independent of the production implementation's batched spectrum scatter.
    spectrum = torch.view_as_complex(module.dft_weight[0].contiguous())
    kernels = []
    for group in range(module.kernel_num):
        mask = torch.zeros_like(spectrum.real)
        mask[module.indices[0, group], module.indices[1, group]] = 1
        matrix = torch.fft.irfft2(spectrum * mask * module.alpha,
                                 s=module.spectrum_shape)
        kernels.append(matrix.reshape(module.out_channels, 3, module.in_channels, 3)
                       .permute(0, 2, 1, 3))
    kernels = torch.stack(kernels)
    outputs = []
    for sample in range(len(x)):
        weight = (attention[sample, :, None, None, None, None] * kernels).sum(0)
        outputs.append(F.conv2d(x[sample:sample + 1], weight.to(x.dtype),
                                bias=module.bias, padding=module.padding))
    return torch.cat(outputs)


class FDWNumerics(unittest.TestCase):
    def test_groups_cover_spectrum_and_uniform_recovers_original(self):
        for channels, size, count in ((48, 80, 22098), (96, 40, 85362), (192, 20, 336306)):
            torch.manual_seed(19)
            original = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            module = FDWConv2d(original)
            rows, columns = module.spectrum_shape
            linear = module.indices[0] * (columns // 2 + 1) + module.indices[1]
            self.assertEqual(linear.unique().numel(), rows * (columns // 2 + 1))
            self.assertEqual(sum(p.numel() for p in module.parameters()), count)
            uniform = torch.full((2, 16), 1 / 16)
            reconstructed = module.aggregate_weight(uniform)
            torch.testing.assert_close(reconstructed[0], original.weight, atol=1e-7, rtol=2e-5)
            with torch.no_grad():
                module.router.kernel_fc.weight.zero_()
                module.router.kernel_fc.bias.zero_()
            module.eval()
            x = torch.randn(2, channels, size, size)
            torch.testing.assert_close(module(x), original(x), atol=3e-6, rtol=2e-5)

    def test_independent_mixture_forward_and_all_gradients(self):
        torch.manual_seed(23)
        module = FDWConv2d(nn.Conv2d(8, 8, 3, padding=1, bias=False).double(), 4).eval()
        # Exercise input-dependent kernels beyond their near-uniform initialization.
        nn.init.normal_(module.router.kernel_fc.weight, std=.1)
        x = torch.randn(2, 8, 5, 7, dtype=torch.float64, requires_grad=True)
        x = x.transpose(-1, -2).detach().requires_grad_()
        actual = module(x)
        expected = explicit_mixture(module, x, module.router(x))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        inputs = (x, *module.parameters())
        gradient = torch.randn_like(actual)
        ours = torch.autograd.grad(actual, inputs, gradient, retain_graph=True)
        reference = torch.autograd.grad(expected, inputs, gradient)
        for a, b in zip(ours, reference):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
        self.assertTrue(torch.autograd.gradcheck(
            lambda values, attention: F.conv2d(
                values, module.aggregate_weight(attention)[0], padding=1),
            (x[:1].detach().requires_grad_(), torch.randn(1, 4, dtype=torch.float64,
                                                        requires_grad=True)), fast_mode=True))

    def test_checkpoint_and_ema_restore_frequency_parameters(self):
        module = FDWConv2d(nn.Conv2d(8, 8, 3, padding=1, bias=False), 4).eval()
        ema = ExpMomentumEMA(module, momentum=.0002, gamma=100, update_buffers=True)
        ema.update_parameters(module)
        with torch.no_grad():
            module.dft_weight.add_(.01)
        ema.update_parameters(module)
        state = io.BytesIO()
        torch.save(ema.module.state_dict(), state)
        state.seek(0)
        restored = FDWConv2d(nn.Conv2d(8, 8, 3, padding=1, bias=False), 4).eval()
        restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
        self.assertNotIn('indices', restored.state_dict())
        x = torch.randn(2, 8, 7, 9)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), ema.module.eval()(x), atol=0, rtol=0)
            self.assertTrue(torch.isfinite(restored(x)).all())

    def test_invalid_division_and_convolution_fail_explicitly(self):
        with self.assertRaises(ValueError):
            FDWConv2d(nn.Conv2d(48, 48, 3), 64)
        with self.assertRaises(ValueError):
            FDWConv2d(nn.Conv2d(48, 48, 3, groups=48))

    @unittest.skipUnless(os.environ.get('PAPER1_RUN_CUDA_TESTS') == '1', 'Enable CUDA comparisons')
    def test_cuda_fp32_and_amp_forward_and_gradients(self):
        torch.cuda.set_per_process_memory_fraction(.06)
        for channels in (48, 96, 192):
            for amp in (False, True):
                torch.manual_seed(31)
                module = FDWConv2d(nn.Conv2d(channels, channels, 3, padding=1, bias=False)).cuda().eval()
                nn.init.normal_(module.router.kernel_fc.weight, std=.05)
                x = torch.randn(2, channels, 9, 11, device='cuda', requires_grad=True)
                x = x.transpose(-1, -2).detach().requires_grad_()
                with torch.autocast('cuda', enabled=amp), torch.backends.cudnn.flags(allow_tf32=False):
                    actual = module(x)
                    expected = explicit_mixture(module, x, module.router(x))
                    loss = actual.float().square().mean()
                    reference_loss = expected.float().square().mean()
                tolerance = .015 if amp else 2e-5
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                inputs = (x, *module.parameters())
                gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
                reference = torch.autograd.grad(reference_loss, inputs)
                for a, b in zip(gradients, reference):
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


class FDWIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(ROOT / 'configs/fdw/rtmdet_tiny_dior_640.py'))

    def build(self, config):
        torch.manual_seed(42)
        model = MODELS.build(copy.deepcopy(config))
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            model.init_weights()
        finally:
            logging.disable(previous)
        return model

    def test_locations_shapes_count_and_other_initial_weights(self):
        model = self.build(self.cfg.model).eval()
        self.assertEqual(model.backbone.fdw_replacements, [
            'stage2.1.blocks.0.conv1.conv', 'stage3.1.blocks.0.conv1.conv',
            'stage4.2.blocks.0.conv1.conv'])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4887018)
        base_cfg = copy.deepcopy(self.cfg.model)
        base_cfg.backbone.type = 'CSPNeXt'
        base_cfg.backbone.pop('fdw_stages'); base_cfg.backbone.pop('fdw_kernel_num')
        baseline = self.build(base_cfg).eval()
        baseline_state = baseline.state_dict()
        for name, value in model.state_dict().items():
            if name in baseline_state:
                torch.testing.assert_close(value, baseline_state[name], atol=0, rtol=0)
        for path in model.backbone.fdw_replacements:
            module = model.backbone.get_submodule(path)
            uniform = torch.full((1, 16), 1 / 16)
            torch.testing.assert_close(module.aggregate_weight(uniform)[0],
                                       baseline.backbone.get_submodule(path).weight,
                                       atol=1e-7, rtol=2e-5)
        with torch.no_grad():
            features = model.backbone(torch.randn(1, 3, 640, 640))
            cls, reg = model.bbox_head(model.neck(features))
        self.assertEqual([tuple(f.shape) for f in features], [
            (1, 96, 80, 80), (1, 192, 40, 40), (1, 384, 20, 20)])
        self.assertEqual([tuple(f.shape) for f in cls], [
            (1, 20, 80, 80), (1, 20, 40, 40), (1, 20, 20, 20)])
        self.assertEqual([tuple(f.shape) for f in reg], [
            (1, 4, 80, 80), (1, 4, 40, 40), (1, 4, 20, 20)])

    def test_disabled_replacement_matches_official(self):
        config = copy.deepcopy(self.cfg.model.backbone)
        config.fdw_stages = ()
        ours = self.build(config)
        ours.eval()
        config.type = 'CSPNeXt'
        config.pop('fdw_stages'); config.pop('fdw_kernel_num')
        baseline = self.build(config)
        baseline.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 129, 161)
            for a, b in zip(ours(x), baseline(x)):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_all_new_parameters_receive_detection_gradients_and_updates(self):
        model = self.build(self.cfg.model).train()
        watched = {n: p for n, p in model.named_parameters()
                   if '.dft_weight' in n or '.conv1.conv.router.' in n}
        self.assertEqual(len(watched), 24)
        initial = {n: p.detach().clone() for n, p in watched.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
        with torch.no_grad():
            for conv in model.bbox_head.head_module.rtm_reg:
                conv.bias.fill_(.4)
        samples = dict(bboxes_labels=torch.tensor(
            [[0, 0, 8, 8, 32, 24], [1, 18, 56, 48, 80, 96]], dtype=torch.float32),
            img_metas=[dict(img_shape=(128, 128)) for _ in range(2)])
        loss = sum(model.loss(torch.randn(2, 3, 128, 128), samples).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in watched.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        optimizer.step()
        for name, parameter in watched.items():
            self.assertGreater((parameter - initial[name]).abs().max().item(), 0, name)

    def test_same_completed_baseline_protocol(self):
        baseline = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        baseline.train_dataloader.batch_size = 36
        baseline.optim_wrapper.optimizer.lr = .004 * 36 / 256
        baseline.param_scheduler[1].eta_min = baseline.optim_wrapper.optimizer.lr * .05
        for key in ('train_dataloader', 'val_dataloader', 'train_cfg', 'val_cfg',
                    'val_evaluator', 'train_pipeline', 'train_pipeline_stage2',
                    'optim_wrapper', 'param_scheduler', 'custom_hooks', 'default_hooks',
                    'randomness', 'load_from', 'resume'):
            self.assertEqual(self.cfg[key], baseline[key], key)
        self.assertEqual(self.cfg.model.neck, baseline.model.neck)
        self.assertEqual(self.cfg.model.bbox_head, baseline.model.bbox_head)
        self.assertIsNone(self.cfg.model.backbone.init_cfg)


if __name__ == '__main__':
    unittest.main()
