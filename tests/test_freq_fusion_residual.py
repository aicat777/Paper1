"""Check raw signed detail, post-CSP routing, and unchanged experiment state."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.freq_fusion.operator import FreqFusion as FullFreqFusion
from custom_models.freq_fusion_residual.operator import SplitFreqFusion

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


class FreqFusionResidualTests(unittest.TestCase):
    def test_signed_detail_matches_independent_spatial_filter(self):
        block = SplitFreqFusion(8, 8, 8)
        nn.init.zeros_(block.content_encoder2.weight)
        nn.init.zeros_(block.content_encoder2.bias)
        high = torch.ones(1, 8, 14, 22)
        high[:, :, 6, 10] += 3
        high.requires_grad_()
        low = torch.randn(1, 8, 7, 11)
        _, detail, _ = block(high, low)
        # Zero generator logits give the known normalized 2D Hamming kernel.
        # This uses grouped convolution, independent of CARAFE/unfold code.
        window = torch.hamming_window(3, periodic=False)
        kernel = window[:, None] * window[None, :]
        kernel = (kernel / kernel.sum())[None, None].repeat(8, 1, 1, 1)
        reference_input = high.detach().clone().requires_grad_()
        expected = reference_input - F.conv2d(reference_input, kernel, padding=1, groups=8)
        torch.testing.assert_close(detail, expected, atol=2e-6, rtol=2e-6)
        self.assertGreater(detail.max().item(), 0)
        self.assertLess(detail.min().item(), 0)
        seed = torch.randn_like(detail)
        actual_grad = torch.autograd.grad(detail, high, seed)[0]
        expected_grad = torch.autograd.grad(expected, reference_input, seed)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)

    def test_recombined_core_matches_previous_full_core_and_gradients(self):
        torch.manual_seed(17)
        split = SplitFreqFusion(96, 96, 24)
        for layer in (split.content_encoder, split.content_encoder2):
            nn.init.normal_(layer.weight, std=.06)
        original = FullFreqFusion(96, 96, 24)
        original.load_state_dict(split.state_dict(), strict=True)
        high = torch.randn(2, 96, 6, 10, requires_grad=True)
        low = torch.randn(2, 96, 3, 5, requires_grad=True)
        ref_high, ref_low = high.detach().clone().requires_grad_(), low.detach().clone().requires_grad_()
        mask, detail, upsampled = split(high, low)
        actual = (mask, high + detail, upsampled)
        expected = original(ref_high, ref_low)
        for first, second in zip(actual, expected):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        names, parameters = zip(*split.named_parameters())
        refs = dict(original.named_parameters())
        seeds = [torch.randn_like(value) / value.numel() ** .5 for value in actual]
        grads = torch.autograd.grad(actual, (high, low, *parameters), seeds)
        ref_grads = torch.autograd.grad(expected, (ref_high, ref_low, *(refs[n] for n in names)), seeds)
        for name, first, second in zip(('high', 'low', *names), grads, ref_grads):
            torch.testing.assert_close(first, second, atol=3e-6, rtol=3e-5, msg=name)

    def test_protocol_initial_state_and_projection_parameters(self):
        original_cfg, new_cfg = config('freq_fusion'), config('freq_fusion_residual')
        for name in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                     'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                     'optim_wrapper', 'param_scheduler', 'randomness', 'default_hooks'):
            self.assertEqual(new_cfg[name], original_cfg[name], name)
        for name in ('backbone', 'bbox_head', 'train_cfg', 'test_cfg', 'data_preprocessor'):
            self.assertEqual(new_cfg.model[name], original_cfg.model[name], name)
        self.assertIsNone(new_cfg.model.backbone.init_cfg)
        self.assertIsNone(new_cfg.load_from)
        self.assertFalse(new_cfg.resume)
        self.assertEqual(new_cfg.train_dataloader.batch_size, 36)
        self.assertEqual(new_cfg.optim_wrapper.get('accumulative_counts', 1), 1)
        models, rng_states = [], []
        for cfg in (original_cfg, new_cfg):
            torch.manual_seed(42)
            model = MODELS.build(deepcopy(cfg.model))
            logging.disable(logging.CRITICAL)
            model.init_weights()
            models.append(model)
            rng_states.append(torch.get_rng_state())
        original_state, new_state = [model.state_dict() for model in models]
        for name, value in original_state.items():
            torch.testing.assert_close(value, new_state[name], atol=0, rtol=0, msg=name)
        torch.testing.assert_close(*rng_states, atol=0, rtol=0)
        extras = set(new_state) - set(original_state)
        self.assertEqual(extras, {'neck.detail_projections.0.weight', 'neck.detail_projections.1.weight'})
        self.assertEqual(sum(p.numel() for p in models[1].parameters()), 5015448)
        for layer, dims in zip(models[1].neck.detail_projections, ((192, 96), (96, 96))):
            self.assertIs(type(layer), nn.Conv2d)
            self.assertEqual((layer.in_channels, layer.out_channels), dims)
            self.assertEqual(layer.kernel_size, (1, 1))
            self.assertIsNone(layer.bias)
            self.assertGreater(layer.weight.std().item(), .05)
            detail = torch.randn(1, dims[0], 3, 5)
            torch.testing.assert_close(layer(-detail), -layer(detail), atol=0, rtol=0)
        initial = {name: value.clone() for name, value in new_state.items()}
        models[1].neck.init_weights()
        for name, value in models[1].state_dict().items():
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0, msg=name)

    def test_parent_initialization_failure_restores_both_module_lists(self):
        neck = MODELS.build(deepcopy(config('freq_fusion_residual').model.neck))
        before_order = list(neck._modules)
        fusion, projections = neck.freq_fusions, neck.detail_projections
        before = {name: value.clone() for name, value in neck.state_dict().items()}
        with patch.object(CSPNeXtPAFPN, 'init_weights', side_effect=RuntimeError('parent init sentinel')):
            with self.assertRaisesRegex(RuntimeError, 'parent init sentinel'):
                neck.init_weights()
        self.assertEqual(list(neck._modules), before_order)
        self.assertIs(neck.freq_fusions, fusion)
        self.assertIs(neck.detail_projections, projections)
        for name, value in neck.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)

    def test_nearest_upsampling_and_zero_detail_recover_exact_baseline(self):
        baseline_cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
        baseline = MODELS.build(deepcopy(baseline_cfg.model.neck))
        neck = MODELS.build(deepcopy(config('freq_fusion_residual').model.neck))
        neck.load_state_dict(baseline.state_dict(), strict=False)

        class NearestWithoutDetail(nn.Module):
            def forward(self, high, low):
                return None, torch.zeros_like(high), F.interpolate(low, scale_factor=2, mode='nearest')

        neck.freq_fusions = nn.ModuleList([NearestWithoutDetail(), NearestWithoutDetail()])
        baseline.eval(); neck.eval()
        features = [torch.randn(2, c, s, s) for c, s in ((96, 20), (192, 10), (384, 5))]
        for order in (True, False):
            neck.upsample_feats_cat_first = baseline.upsample_feats_cat_first = order
            with torch.no_grad():
                actual, expected = neck(features), baseline(features)
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_detail_bypasses_complete_csp_block_and_reaches_next_fusion_and_pan(self):
        neck = MODELS.build(deepcopy(config('freq_fusion_residual').model.neck))
        neck.eval()

        class KnownDetail(nn.Module):
            def forward(self, high, low):
                self.high, self.low = high.detach().clone(), low.detach().clone()
                self.detail = torch.linspace(-2, 2, high.numel()).reshape_as(high)
                self.up = F.interpolate(low, scale_factor=2, mode='nearest')
                return None, self.detail, self.up

        neck.freq_fusions = nn.ModuleList([KnownDetail(), KnownDetail()])
        features = [torch.randn(1, c, s, s) for c, s in ((96, 20), (192, 10), (384, 5))]
        for order in (True, False):
            neck.upsample_feats_cat_first = order
            seen, handles = {}, []
            for kind, layers in (('csp', neck.top_down_layers), ('projection', neck.detail_projections),
                                 ('pan', neck.bottom_up_layers), ('out', neck.out_layers)):
                for idx, layer in enumerate(layers):
                    key = (kind, idx)
                    handles.append(layer.register_forward_pre_hook(
                        lambda m, args, key=key: seen.__setitem__((key, 'input'), args[0].detach().clone())))
                    handles.append(layer.register_forward_hook(
                        lambda m, args, result, key=key: seen.__setitem__((key, 'output'), result.detach().clone())))
            with torch.no_grad():
                neck(features)
            for handle in handles:
                handle.remove()
            for idx, block in enumerate(neck.freq_fusions):
                first, second = seen[(('csp', idx), 'input')].chunk(2, dim=1)
                up, high = (first, second) if order else (second, first)
                torch.testing.assert_close(up, block.up, atol=0, rtol=0)
                torch.testing.assert_close(high, block.high, atol=0, rtol=0)
                torch.testing.assert_close(seen[(('projection', idx), 'input')], block.detail, atol=0, rtol=0)
            fused_middle = seen[(('csp', 0), 'output')] + seen[(('projection', 0), 'output')]
            fused_finest = seen[(('csp', 1), 'output')] + seen[(('projection', 1), 'output')]
            torch.testing.assert_close(neck.freq_fusions[1].low, fused_middle, atol=0, rtol=0)
            torch.testing.assert_close(seen[(('pan', 0), 'input')][:, 96:], fused_middle, atol=0, rtol=0)
            torch.testing.assert_close(seen[(('out', 0), 'input')], fused_finest, atol=0, rtol=0)
            self.assertEqual([seen[(('csp', i), 'input')].shape[1] for i in range(2)], [384, 192])
            self.assertEqual([seen[(('pan', i), 'input')].shape[1] for i in range(2)], [192, 384])


if __name__ == '__main__':
    unittest.main()
