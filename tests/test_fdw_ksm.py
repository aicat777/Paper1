"""Local KSM ordering, numerical gradients, checkpoint/EMA and detector integration."""
from copy import deepcopy
import io
import logging
import os
from pathlib import Path
import unittest

from mmengine.config import Config
from mmyolo.models.layers.ema import ExpMomentumEMA
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
import torch
from torch import nn
from torch.nn import functional as F

from custom_models.fdw.operator import FDWConv2d as PreviousFDW
from custom_models.fdw_ksm.operator import FDWKSMConv2d

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def explicit_local(module, x):
    """Index each (output, input, row, column) explicitly, avoiding permute."""
    local = module.local
    descriptor = x.mean(dim=(-2, -1)).unsqueeze(1)
    descriptor = F.layer_norm(descriptor, (module.in_channels,), local.norm.weight,
                              local.norm.bias, local.norm.eps)
    logits = F.conv1d(descriptor, local.conv.weight, padding=local.conv.padding)
    outputs = []
    for output in range(module.out_channels):
        channels = []
        for channel in range(module.in_channels):
            entries = logits[:, output * 9:(output + 1) * 9, channel]
            channels.append(entries.unflatten(-1, (3, 3)))
        outputs.append(torch.stack(channels, dim=1))
    return 2 * torch.stack(outputs, dim=1).sigmoid()


def explicit_forward(module, x):
    # Independently reconstruct individual disjoint kernels before mixing them.
    spectrum = torch.view_as_complex(module.dft_weight[0].contiguous())
    kernels = []
    for group in range(module.kernel_num):
        mask = torch.zeros_like(spectrum.real)
        mask[module.indices[0, group], module.indices[1, group]] = 1
        matrix = torch.fft.irfft2(spectrum * mask * module.alpha, s=module.spectrum_shape)
        kernels.append(matrix.reshape(module.out_channels, 3, module.in_channels, 3)
                       .permute(0, 2, 1, 3))
    kernels = torch.stack(kernels)
    attention, modulation = module.router(x), explicit_local(module, x)
    outputs = []
    for sample in range(len(x)):
        weight = (attention[sample, :, None, None, None, None] * kernels).sum(0)
        weight = weight * modulation[sample]
        outputs.append(F.conv2d(x[sample:sample + 1], weight.to(x.dtype),
                                bias=module.bias, padding=module.padding))
    return torch.cat(outputs)


class FDWKSMNumerics(unittest.TestCase):
    def test_neutral_local_recovers_previous_fdw_and_author_initialization(self):
        for channels, neighborhood, additional in ((48, 3, 1392), (96, 5, 4512), (192, 5, 9024)):
            original = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            module = FDWKSMConv2d(original).eval()
            previous = PreviousFDW(original).eval()
            previous.load_state_dict({k: v for k, v in module.state_dict().items()
                                      if not k.startswith('local.')}, strict=True)
            self.assertEqual(module.local.conv.kernel_size, (neighborhood,))
            self.assertEqual(sum(p.numel() for p in module.local.parameters()), additional)
            torch.testing.assert_close(module.local.conv.weight,
                                       torch.full_like(module.local.conv.weight, 1e-6), atol=0, rtol=0)
            x = torch.randn(2, channels, 9, 11)
            with torch.no_grad():
                self.assertLess((module.local(x) - 1).abs().max().item(), 1e-5)
                module.local.conv.weight.zero_()
                torch.testing.assert_close(module(x), previous(x), atol=0, rtol=0)

    def test_nonsquare_channel_mapping_nonuniform_gates_and_all_gradients(self):
        torch.manual_seed(23)
        module = FDWKSMConv2d(nn.Conv2d(4, 8, 3, padding=1, bias=True).double(), 4).eval()
        nn.init.normal_(module.router.kernel_fc.weight, std=.1)
        nn.init.normal_(module.local.conv.weight, std=.2)
        x = torch.randn(2, 4, 7, 5, dtype=torch.float64).transpose(-1, -2).requires_grad_()
        modulation = module.local(x)
        self.assertEqual(modulation.shape, (2, 8, 4, 3, 3))
        self.assertGreater((modulation[0] - modulation[1]).abs().max().item(), .1)
        torch.testing.assert_close(modulation, explicit_local(module, x), atol=1e-14, rtol=1e-12)
        actual, expected = module(x), explicit_forward(module, x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        inputs, gradient = (x, *module.parameters()), torch.randn_like(actual)
        ours = torch.autograd.grad(actual, inputs, gradient, retain_graph=True)
        reference = torch.autograd.grad(expected, inputs, gradient)
        for a, b in zip(ours, reference):
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
        self.assertTrue(torch.autograd.gradcheck(module, (x.detach().requires_grad_(),),
                                                fast_mode=True))

    def test_checkpoint_and_ema_include_local_and_frequency_parameters(self):
        module = FDWKSMConv2d(nn.Conv2d(8, 8, 3, padding=1, bias=False), 4).eval()
        ema = ExpMomentumEMA(module, momentum=.0002, gamma=2000, update_buffers=True)
        ema.update_parameters(module)
        with torch.no_grad():
            module.dft_weight.add_(.01)
            module.local.conv.weight.add_(.02)
            module.local.norm.bias.add_(.1)
        ema.update_parameters(module)
        state = io.BytesIO()
        torch.save(ema.module.state_dict(), state)
        state.seek(0)
        restored = FDWKSMConv2d(nn.Conv2d(8, 8, 3, padding=1, bias=False), 4).eval()
        restored.load_state_dict(torch.load(state, weights_only=True), strict=True)
        self.assertNotIn('indices', restored.state_dict())
        x = torch.randn(2, 8, 7, 9)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), ema.module.eval()(x), atol=0, rtol=0)

    @unittest.skipUnless(os.environ.get('PAPER1_RUN_CUDA_TESTS') == '1', 'Enable CUDA comparisons')
    def test_cuda_fp32_amp_forward_and_all_gradients(self):
        torch.cuda.set_per_process_memory_fraction(.08)
        for channels in (48, 96, 192):
            for amp in (False, True):
                torch.manual_seed(31)
                module = FDWKSMConv2d(nn.Conv2d(channels, channels, 3, padding=1, bias=False)).cuda().eval()
                nn.init.normal_(module.router.kernel_fc.weight, std=.05)
                nn.init.normal_(module.local.conv.weight, std=.1)
                x = torch.randn(2, channels, 9, 11, device='cuda').transpose(-1, -2).requires_grad_()
                with torch.autocast('cuda', enabled=amp), torch.backends.cudnn.flags(allow_tf32=False):
                    actual, expected = module(x), explicit_forward(module, x)
                    loss, reference_loss = actual.float().square().mean(), expected.float().square().mean()
                tolerance = .015 if amp else 2e-5
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                inputs = (x, *module.parameters())
                ours = torch.autograd.grad(loss, inputs, retain_graph=True)
                reference = torch.autograd.grad(reference_loss, inputs)
                for a, b in zip(ours, reference):
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


class FDWKSMIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config.fromfile(str(ROOT / 'configs/fdw_ksm/rtmdet_tiny_dior_640.py'))

    def build(self, config):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(config))
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            model.init_weights()
        finally:
            logging.disable(previous)
        return model

    def test_locations_shapes_count_and_existing_fdw_initialization(self):
        model = self.build(self.cfg.model).eval()
        fdw = self.build(Config.fromfile(str(ROOT / 'configs/fdw/rtmdet_tiny_dior_640.py')).model).eval()
        self.assertEqual(model.backbone.fdw_replacements, fdw.backbone.fdw_replacements)
        self.assertEqual(len(model.backbone.fdw_replacements), 3)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4901946)
        # The only state entries added relative to FDW are local branch weights.
        state = model.state_dict()
        for name, value in fdw.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)
        self.assertEqual(len(set(state) - set(fdw.state_dict())), 9)
        with torch.no_grad():
            features = model.backbone(torch.randn(1, 3, 640, 640))
            cls, reg = model.bbox_head(model.neck(features))
        self.assertEqual([list(f.shape) for f in features], [
            [1, 96, 80, 80], [1, 192, 40, 40], [1, 384, 20, 20]])
        self.assertEqual([list(f.shape) for f in cls], [[1, 20, s, s] for s in (80, 40, 20)])
        self.assertEqual([list(f.shape) for f in reg], [[1, 4, s, s] for s in (80, 40, 20)])

    def test_detection_gradients_and_updates_include_all_local_parameters(self):
        model = self.build(self.cfg.model).train()
        ids = {id(p) for layer in model.modules() if isinstance(layer, FDWKSMConv2d)
               for p in layer.parameters()}
        watched = {n: p for n, p in model.named_parameters() if id(p) in ids}
        self.assertEqual(len(watched), 33)
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

    def test_same_fdw_training_protocol_and_unchanged_neck_head(self):
        previous = Config.fromfile(str(ROOT / 'configs/fdw/rtmdet_tiny_dior_640.py'))
        for key in ('train_dataloader', 'val_dataloader', 'train_cfg', 'val_cfg',
                    'val_evaluator', 'train_pipeline', 'train_pipeline_stage2',
                    'optim_wrapper', 'param_scheduler', 'custom_hooks', 'default_hooks',
                    'randomness', 'load_from', 'resume'):
            self.assertEqual(self.cfg[key], previous[key], key)
        self.assertEqual(self.cfg.model.neck, previous.model.neck)
        self.assertEqual(self.cfg.model.bbox_head, previous.model.bbox_head)
        self.assertIsNone(self.cfg.model.backbone.init_cfg)


if __name__ == '__main__':
    unittest.main()
