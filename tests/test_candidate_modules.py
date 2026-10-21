"""Numerical and integration contracts for the independent candidate methods.

Run from the repository root: python -m unittest discover -s tests -v
"""
import copy
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.state_guided.context import CompressedStateContext
from custom_models.state_guided.upsample import StateGuidedUpsample
from custom_models.state_guided.neck import StateGuidedPAFPN
from custom_models.magnitude_detail.context import magnitude_linear_attention
from custom_models.magnitude_detail.detail import GatedShallowDetail
from custom_models.magnitude_detail.neck import MagnitudeDetailPAFPN
from custom_models.frequency_guided.frequency import (
    SpatialFrequencySelector, split_frequency_bands)
from custom_models.frequency_guided.upsample import FrequencyGuidedUpsample
from custom_models.frequency_guided.neck import FrequencyGuidedPAFPN

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
register_all_modules()


class NumericalContracts(unittest.TestCase):
    def test_frequency_partition_and_identity(self):
        # Includes odd/non-square FFTs; neutral modulation must not alter x.
        for shape in ((2, 8, 13, 17), (1, 8, 16, 16)):
            x = torch.randn(*shape, requires_grad=True)
            torch.testing.assert_close(sum(split_frequency_bands(x)), x)
            module = SpatialFrequencySelector(8, groups=2)
            y, energy = module(x)
            torch.testing.assert_close(y, x)
            torch.testing.assert_close(energy.sum(1), torch.ones_like(energy[:, 0]))
            (y.square().mean() + energy[:, -1].mean()).backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(module.band_logits.weight.grad.abs().sum().item(), 0)

    def test_frequency_partition_distinguishes_dc_and_checkerboard(self):
        dc = torch.ones(1, 4, 16, 16)
        low, *high = split_frequency_bands(dc)
        torch.testing.assert_close(low, dc)
        self.assertLess(sum(x.abs().sum() for x in high).item(), 1e-5)
        grid = torch.arange(16)
        checker = ((grid[:, None] + grid[None, :]) % 2 * 2 - 1).float()
        bands = split_frequency_bands(checker[None, None])
        torch.testing.assert_close(bands[-1], checker[None, None])
        self.assertLess(sum(x.abs().sum() for x in bands[:-1]).item(), 1e-5)

    def test_local_upsamplers_preserve_constant_values(self):
        value = torch.full((2, 8, 7, 9), 2.5, requires_grad=True)
        lateral = torch.randn(2, 8, 13, 17, requires_grad=True)
        states = torch.randn(2, 4, 6, requires_grad=True)
        energy = torch.rand(2, 4, 13, 17)
        a = StateGuidedUpsample(8, 8, state_channels=6, num_states=4)
        c = FrequencyGuidedUpsample(8, 8)
        for y in (a(value, lateral, states), c(value, lateral, energy)):
            torch.testing.assert_close(y, torch.full_like(y, 2.5))
        # A spatially varying value must send gradients through state guidance.
        y = a(torch.randn_like(value), lateral, states)
        y.square().mean().backward()
        self.assertGreater(states.grad.abs().sum().item(), 0)

    def test_magnitude_attention_matches_explicit_attention(self):
        q, k = torch.rand(2, 2, 11, 6) + .1, torch.rand(2, 2, 11, 6) + .1
        v = torch.randn(2, 2, 11, 6, requires_grad=True)
        # An independent small NxN reference validates the linear factorization.
        score = (q @ k.transpose(-2, -1)) * (6 ** -.5 / 11)
        z = score.sum(-1, keepdim=True)
        attention = score * (1 + 1 / z) - z / 11
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 2, 11))
        result = magnitude_linear_attention(q, k, v)
        torch.testing.assert_close(result, attention @ v, atol=2e-6, rtol=2e-5)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        constant = torch.ones_like(v) * 3
        torch.testing.assert_close(magnitude_linear_attention(q, k, constant), constant)

    def test_attention_control_is_parameter_free_and_magnitude_invariant(self):
        torch.manual_seed(13)
        q, k = torch.rand(1, 2, 9, 4) + .1, torch.rand(1, 2, 9, 4) + .1
        v = torch.randn(1, 2, 9, 4)
        ordinary = magnitude_linear_attention(q, k, v, magnitude_aware=False)
        scores = q @ k.transpose(-2, -1)
        expected = (scores / scores.sum(-1, keepdim=True)) @ v
        torch.testing.assert_close(ordinary, expected)
        torch.testing.assert_close(ordinary, magnitude_linear_attention(
            3 * q, k, v, magnitude_aware=False))
        difference = (magnitude_linear_attention(3 * q, k, v)
                      - magnitude_linear_attention(q, k, v)).abs().max()
        self.assertGreater(difference.item(), 1e-3)

    def test_local_resampling_matches_coordinate_reference(self):
        # Nonconstant values expose neighbor order, nearest indexing and border
        # errors that constant-preservation tests cannot detect.
        value = torch.arange(24).float().reshape(1, 2, 3, 4)
        lateral = torch.randn(1, 2, 5, 7)
        probabilities = torch.arange(1, 10).float() / 45
        operators = (StateGuidedUpsample(2, 4),
                     FrequencyGuidedUpsample(2, 4, use_frequency=False))
        expected = torch.empty_like(lateral)
        for oy in range(5):
            for ox in range(7):
                iy, ix = oy * 3 // 5, ox * 4 // 7
                weighted = sum(probabilities[j] * value[:, :,
                    min(max(iy + j // 3 - 1, 0), 2),
                    min(max(ix + j % 3 - 1, 0), 3)] for j in range(9))
                expected[:, :, oy, ox] = .5 * (value[:, :, iy, ix] + weighted)
        for module in operators:
            with torch.no_grad():
                module.kernel_logits.weight.zero_()
                module.kernel_logits.bias.copy_(probabilities.log())
                module.mix_logit.zero_()
                torch.testing.assert_close(module(value, lateral), expected)

    def test_direct_guides_receive_nonzero_gradients(self):
        torch.manual_seed(17)
        value, lateral = torch.randn(2, 8, 4, 5), torch.randn(2, 8, 8, 10)
        states = torch.randn(2, 4, 6, requires_grad=True)
        bands = torch.rand(2, 4, 8, 10, requires_grad=True)
        cases = ((StateGuidedUpsample(8, 8, 6, 4), states),
                 (FrequencyGuidedUpsample(8, 8), bands))
        for module, guide in cases:
            module(value, lateral, guide).square().mean().backward()
            self.assertTrue(torch.isfinite(guide.grad).all())
            self.assertGreater(guide.grad.abs().sum().item(), 0)

    def test_state_context_and_shallow_detail_gradients(self):
        context = CompressedStateContext(16, 8, 4)
        # A bias shared by spatial positions is unidentifiable under this softmax.
        self.assertIsNone(context.assignment.bias)
        x = torch.randn(2, 16, 7, 9, requires_grad=True)
        y, states = context(x)
        self.assertEqual(states.shape, (2, 4, 8))
        y.square().mean().backward()
        self.assertGreater(context.assignment.weight.grad.abs().sum().item(), 0)
        detail = GatedShallowDetail(8, 16, 4)
        shallow = torch.randn(2, 8, 13, 17, requires_grad=True)
        y = detail(shallow, x.detach())
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertGreater(shallow.grad.abs().sum().item(), 0)


class ConfigurationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))

    def test_all_configs_keep_the_training_protocol(self):
        paths = sorted(path for method in (
            'state_guided', 'magnitude_detail', 'frequency_guided')
            for path in (ROOT / 'configs' / method).glob('*.py'))
        self.assertEqual(len(paths), 12)
        work_dirs = set()
        for path in paths:
            cfg = Config.fromfile(str(path))
            with self.subTest(config=str(path.relative_to(ROOT))):
                for key in ('train_dataloader', 'val_dataloader', 'val_evaluator',
                            'train_cfg', 'optim_wrapper', 'param_scheduler',
                            'custom_hooks', 'randomness', 'default_hooks',
                            'test_dataloader', 'test_evaluator', 'val_cfg',
                            'test_cfg'):
                    self.assertEqual(cfg[key], self.base[key])
                self.assertIsNone(cfg.model.backbone.init_cfg)
                self.assertIsNone(cfg.load_from)
                self.assertFalse(cfg.resume)
                self.assertEqual(cfg.model.bbox_head, self.base.model.bbox_head)
                work_dirs.add(cfg.work_dir)
                neck = MODELS.build(cfg.model.neck)
                self.assertEqual(neck.out_channels, 96)
        self.assertEqual(len(work_dirs), len(paths))

    def test_controls_retain_both_modules_or_identical_context_parameters(self):
        a = MODELS.build(Config.fromfile(str(ROOT /
            'configs/state_guided/no_guidance.py')).model.neck)
        self.assertIsNotNone(a.context)
        self.assertIsNotNone(a.local_upsample)
        self.assertIsNone(a.local_upsample.state_query)
        c = MODELS.build(Config.fromfile(str(ROOT /
            'configs/frequency_guided/no_guidance.py')).model.neck)
        self.assertIsNotNone(c.frequency)
        self.assertIsNotNone(c.local_upsample)
        self.assertIsNone(c.local_upsample.energy_projection)
        self.assertFalse(c.use_frequency_guidance)
        necks = [MODELS.build(Config.fromfile(str(ROOT /
                 f'configs/magnitude_detail/{name}.py')).model.neck)
                 for name in ('context_only', 'linear_context_only')]
        signatures = [dict((n, tuple(p.shape)) for n, p in m.named_parameters())
                      for m in necks]
        self.assertEqual(signatures[0], signatures[1])
        self.assertTrue(necks[0].context.magnitude_aware)
        self.assertFalse(necks[1].context.magnitude_aware)
        self.assertTrue(all(m.detail is None for m in necks))

    def test_frequency_identity_after_framework_initialization(self):
        cfg = Config.fromfile(str(ROOT / 'configs/frequency_guided/full.py'))
        neck = MODELS.build(cfg.model.neck)
        neck.init_weights()
        for x in (torch.randn(1, 96, 13, 17), torch.zeros(1, 96, 13, 17)):
            y, statistics = neck.frequency(x)
            torch.testing.assert_close(y, x)
            self.assertTrue(torch.isfinite(statistics).all())

    def test_disabling_modules_recovers_original_neck(self):
        options = copy.deepcopy(dict(self.base.model.neck))
        options.pop('type')
        baseline = MODELS.build(self.base.model.neck)
        baseline.eval()
        inputs = [torch.randn(1, c, s, s) for c, s in
                  ((96, 16), (192, 8), (384, 4))]
        shallow = torch.randn(1, 48, 32, 32)
        candidates = [
            (StateGuidedPAFPN(**options, use_context=False, use_upsample=False), inputs),
            (MagnitudeDetailPAFPN(**options, use_context=False, use_detail=False),
             [shallow] + inputs),
            (FrequencyGuidedPAFPN(**options, use_frequency=False, use_upsample=False), inputs)]
        with torch.no_grad():
            expected = baseline(inputs)
            for neck, neck_inputs in candidates:
                neck.load_state_dict(baseline.state_dict(), strict=True)
                neck.eval()
                actual = neck(neck_inputs)
                for x, y in zip(actual, expected):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_full_necks_accept_odd_rectangular_pyramids(self):
        inputs = [torch.randn(1, c, h, w) for c, h, w in
                  ((96, 25, 19), (192, 13, 10), (384, 7, 5))]
        for method in ('state_guided', 'magnitude_detail', 'frequency_guided'):
            cfg = Config.fromfile(str(ROOT / f'configs/{method}/full.py'))
            neck = MODELS.build(cfg.model.neck)
            neck.eval()
            data = ([torch.randn(1, 48, 50, 38)] + inputs
                    if method == 'magnitude_detail' else inputs)
            with torch.no_grad():
                outputs = neck(data)
            self.assertEqual([x.shape[-2:] for x in outputs],
                             [x.shape[-2:] for x in inputs])
            self.assertTrue(all(torch.isfinite(x).all() for x in outputs))


if __name__ == '__main__':
    unittest.main()
