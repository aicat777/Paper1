"""Independent coordinate/kernel checks and unchanged-PAN/baseline tests."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.freq_fusion.operator import FreqFusion, reassemble
from custom_models.freq_fusion.resample import LocalSimGuidedSampler, compute_similarity

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


class FreqFusionTests(unittest.TestCase):
    def test_carafe_center_border_order_and_gradients(self):
        x = torch.arange(24., dtype=torch.double).reshape(1, 2, 3, 4).requires_grad_()
        mask = torch.zeros(1, 9, 6, 8, dtype=torch.double)
        mask[:, 4] = 1
        torch.testing.assert_close(reassemble(x, mask, 3, 2), F.interpolate(x, scale_factor=2))
        mask[:, 4], mask[:, 0] = 0, 1
        expected = F.interpolate(F.pad(x, (1, 0, 1, 0))[:, :, :3, :4], scale_factor=2)
        torch.testing.assert_close(reassemble(x, mask, 3, 2), expected)
        # Uniform kernels must have zero-padding attenuation at the border.
        ones = torch.ones(1, 1, 3, 4)
        smooth = reassemble(ones, torch.full((1, 9, 3, 4), 1 / 9), 3)
        self.assertAlmostEqual(smooth[0, 0, 0, 0].item(), 4 / 9, places=6)
        self.assertEqual(smooth[0, 0, 1, 1].item(), 1)
        weights = torch.randn(1, 9, 3, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: reassemble(a, b, 3), (x, weights)))

    def test_sampler_identity_one_pixel_translation_and_group_order(self):
        sampler = LocalSimGuidedSampler(24, 4)
        # Odd, non-square low resolution exposes x/y and pixel-unshuffle errors.
        h, w = 3, 5
        value = torch.arange(8 * 2 * h * 2 * w, dtype=torch.float32).reshape(1, 8, 2 * h, 2 * w)
        offset = sampler.init_pos.expand(1, -1, h, w).clone()
        torch.testing.assert_close(sampler.sample(value, offset), value, atol=3e-5, rtol=1e-6)
        # +.5 coarse pixels = +1 high-resolution pixel. Move only group 1 in x.
        offset[:, 4:8] += .5
        expected = value.clone()
        expected[:, 2:4, :, :-1] = value[:, 2:4, :, 1:]
        torch.testing.assert_close(sampler.sample(value, offset), expected, atol=3e-5, rtol=1e-6)

    def test_similarity_dilation_and_normalized_kernels(self):
        x = torch.randn(1, 3, 9, 11)
        similarity = compute_similarity(x)
        expected = F.cosine_similarity(x[:, :, 4, 5], x[:, :, 2, 3], dim=1)
        torch.testing.assert_close(similarity[:, 0, 4, 5], expected)
        block = FreqFusion(96, 96, 24)
        logits = torch.randn(2, 25, 7, 11)
        mask = block.kernel_normalizer(logits, 5, block.hamming_lowpass)
        self.assertTrue((mask > 0).all())
        torch.testing.assert_close(mask.sum(1), torch.ones(2, 7, 11))
        with self.assertRaises(ValueError):
            block(torch.zeros(1, 96, 15, 20), torch.zeros(1, 96, 7, 10))

    def test_init_keeps_baseline_state_and_custom_initialization(self):
        base = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        ours = Config.fromfile(str(ROOT / 'configs/freq_fusion/rtmdet_tiny_dior_640.py'))
        models = []
        for cfg in (base, ours):
            torch.manual_seed(42)
            model = MODELS.build(deepcopy(cfg.model))
            logging.disable(logging.CRITICAL)
            model.init_weights()
            models.append(model)
        state = models[1].state_dict()
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0, msg=name)
        for layer, width, compressed in zip(models[1].neck.freq_fusions, (192, 96), (48, 24)):
            self.assertEqual((layer.hr_channels, layer.lr_channels, layer.compressed_channels),
                             (width, width, compressed))
            for generator in (layer.content_encoder, layer.content_encoder2):
                self.assertTrue(.0009 < generator.weight.std() < .0011)
            self.assertGreater(layer.dysampler.direct_scale.weight.std().item(), .01)
        initial = {k: v.clone() for k, v in models[1].state_dict().items()}
        models[1].neck.init_weights()
        for name, value in models[1].state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)

    def test_bypass_is_exact_baseline_and_real_concat_widths(self):
        base_cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        cfg = Config.fromfile(str(ROOT / 'configs/freq_fusion/rtmdet_tiny_dior_640.py'))
        base, neck = MODELS.build(deepcopy(base_cfg.model.neck)), MODELS.build(deepcopy(cfg.model.neck))
        neck.load_state_dict(base.state_dict(), strict=False)
        class NearestPair(nn.Module):
            def forward(self, hr, lr):
                return None, hr, F.interpolate(lr, scale_factor=2, mode='nearest')
        neck.freq_fusions = nn.ModuleList([NearestPair(), NearestPair()])
        base.eval(); neck.eval()
        features = [torch.randn(2, c, s, s) for c, s in ((96, 20), (192, 10), (384, 5))]
        seen = []
        handles = [layer.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[1]))
                   for layer in (*neck.top_down_layers, *neck.bottom_up_layers)]
        with torch.no_grad():
            for actual, expected in zip(neck(features), base(features)):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for handle in handles:
            handle.remove()
        self.assertEqual(seen, [384, 192, 192, 384])


if __name__ == '__main__':
    unittest.main()
