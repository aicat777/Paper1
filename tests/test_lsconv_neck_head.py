"""New neck/head adapters: scope, exact surviving initialization and sharing."""
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

from custom_models.lsconv.operator import LSConv as BackboneLSConv
from custom_models.lsconv_neck.block import LSFeatureBlock as NeckBlock
from custom_models.lsconv_head.block import LSFeatureBlock as HeadBlock

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('lsconv_neck', 'lsconv_head')
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method, initialize=True):
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(config(method).model))
    if initialize:
        model.init_weights()
    return model


class NeckHeadAdapters(unittest.TestCase):
    def test_verified_operator_copies_and_identical_protocol(self):
        baseline = config('lsconv')
        for method in METHODS:
            cfg = config(method)
            for file in ('operator.py', 'aggregation.py'):
                self.assertEqual(hashlib.sha256((ROOT / 'custom_models' / method / file).read_bytes()).digest(),
                                 hashlib.sha256((ROOT / 'custom_models/lsconv' / file).read_bytes()).digest())
            for key in ('train_dataloader', 'val_dataloader', 'test_dataloader', 'train_cfg',
                        'val_evaluator', 'test_evaluator', 'custom_hooks', 'optim_wrapper',
                        'param_scheduler', 'default_hooks', 'randomness'):
                self.assertEqual(cfg[key], baseline[key], (method, key))
            self.assertEqual(cfg.model.backbone, baseline.model.backbone)
            self.assertIsNone(cfg.load_from)
            self.assertFalse(cfg.resume)

    def test_untouched_parameters_and_rng_match_three_site_foundation_exactly(self):
        baseline = build('lsconv')
        baseline_rng = torch.get_rng_state().clone()
        state = baseline.state_dict()
        for method in METHODS:
            model = build(method)
            torch.testing.assert_close(torch.get_rng_state(), baseline_rng, atol=0, rtol=0)
            common = set(state) & set(model.state_dict())
            for name in common:
                torch.testing.assert_close(model.state_dict()[name], state[name], atol=0, rtol=0, msg=name)
            self.assertFalse(any(hasattr(m, '_params_init_info') for m in model.modules()))
            self.assertFalse(model.neck._initialization_slots if method == 'lsconv_neck'
                             else model.bbox_head.head_module._initialization_slots)

    def test_lsblock_constructor_initialization_survives_root_and_repeated_init(self):
        for method, kind in (('lsconv_neck', NeckBlock), ('lsconv_head', HeadBlock)):
            model = build(method, initialize=False)
            blocks = [(n, m) for n, m in model.named_modules() if isinstance(m, kind)]
            before = {n: deepcopy(m.state_dict()) for n, m in blocks}
            model.init_weights()
            for name, block in blocks:
                for key, value in block.state_dict().items():
                    torch.testing.assert_close(value, before[name][key], atol=0, rtol=0)
                self.assertEqual(int(torch.count_nonzero(block.ffn.m.pw2.bn.weight)), 0)
            state = deepcopy(model.state_dict())
            model.init_weights()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_exception_restores_active_modules(self):
        for method, parent in (('lsconv_neck', CSPNeXtPAFPN),
                               ('lsconv_head', RTMDetSepBNHeadModule)):
            model = build(method, initialize=False)
            adapter = model.neck if method == 'lsconv_neck' else model.bbox_head.head_module
            slots = adapter._initialization_slots
            with patch.object(parent, 'init_weights', side_effect=RuntimeError('injected failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    adapter.init_weights()
            for path, _, replacement in slots:
                self.assertIs(adapter.get_submodule(path), replacement)
            self.assertFalse(adapter.is_init)
            adapter.init_weights()
            self.assertFalse(adapter._initialization_slots)

    def test_neck_scope_and_original_outputs(self):
        model = build('lsconv_neck')
        paths = model.neck.lsblock_replacements
        self.assertEqual(paths, ['top_down_layers.0.0.blocks.0', 'top_down_layers.1.blocks.0',
                                 'bottom_up_layers.0.blocks.0', 'bottom_up_layers.1.blocks.0'])
        self.assertEqual([model.neck.get_submodule(p).mixer.bn.num_features for p in paths],
                         [96, 48, 96, 192])
        self.assertEqual(sum(isinstance(m, BackboneLSConv) for m in model.backbone.modules()), 3)
        self.assertEqual(sum(isinstance(m, NeckBlock) for m in model.neck.modules()), 4)
        self.assertFalse(any(isinstance(m, CSPNeXtBlock) for m in model.neck.modules()))
        self.assertIs(type(model.bbox_head.head_module), RTMDetSepBNHeadModule)
        for layer, width in zip(model.neck.out_layers, (96, 192, 384)):
            self.assertEqual((layer.conv.in_channels, layer.conv.out_channels), (width, 96))
            self.assertEqual(layer.conv.kernel_size, (3, 3))

    def test_head_scope_shared_conv_independent_norm_and_branches(self):
        model = build('lsconv_head')
        head = model.bbox_head.head_module
        self.assertIs(type(model.neck), CSPNeXtPAFPN)
        self.assertEqual(sum(isinstance(m, HeadBlock) for m in head.modules()), 6)
        for branch in ('cls_convs', 'reg_convs'):
            towers = getattr(head, branch)
            self.assertEqual(towers[0][0].conv.kernel_size, (3, 3))
            self.assertIs(towers[0][0].conv, towers[2][0].conv)
            refs = dict(towers[0][1].named_modules())
            for level in (1, 2):
                for name, layer in towers[level][1].named_modules():
                    if isinstance(layer, nn.Conv2d):
                        self.assertIs(layer, refs[name])
                    elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                        self.assertIsNot(layer, refs[name])
                        self.assertIsNot(layer.weight, refs[name].weight)
            self.assertFalse(any(isinstance(m, nn.SiLU) for m in towers[0][1].modules()))
        cls_ids = {id(p) for p in head.cls_convs.parameters()}
        self.assertFalse(cls_ids & {id(p) for p in head.reg_convs.parameters()})
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))

    def test_head_shared_gradient_is_sum_and_survives_serialization(self):
        model = build('lsconv_head')
        head = model.bbox_head.head_module
        losses = []
        for level, side in enumerate((9, 5, 3)):
            block = head.cls_convs[level][1]
            with torch.no_grad():
                block.ffn.m.pw2.bn.weight.fill_(.3)
            losses.append(block(torch.randn(2, 96, side, side)).square().mean())
        parameter = head.cls_convs[0][1].mixer.lkp.cv4.weight
        grads = [torch.autograd.grad(loss, parameter, retain_graph=True)[0] for loss in losses]
        total = torch.autograd.grad(sum(losses), parameter)[0]
        torch.testing.assert_close(total, sum(grads), atol=2e-6, rtol=2e-5)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build('lsconv_head')
        restored.load_state_dict(torch.load(buffer, weights_only=False), strict=True)
        self.assertIs(restored.bbox_head.head_module.cls_convs[0][1].ffn.m.pw1.c,
                      restored.bbox_head.head_module.cls_convs[2][1].ffn.m.pw1.c)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

    def test_zero_ffn_receives_grad_then_activates(self):
        for kind in (NeckBlock, HeadBlock):
            block = kind(48)
            optimizer = torch.optim.SGD(block.parameters(), lr=.1)
            optimizer.zero_grad()
            block(torch.randn(2, 48, 7, 9)).square().mean().backward()
            self.assertEqual(float(block.ffn.m.pw1.c.weight.grad.abs().max()), 0.)
            self.assertGreater(float(block.ffn.m.pw2.bn.weight.grad.abs().max()), 0.)
            optimizer.step()
            optimizer.zero_grad()
            block(torch.randn(2, 48, 7, 9)).square().mean().backward()
            self.assertGreater(float(block.ffn.m.pw1.c.weight.grad.abs().max()), 0.)


if __name__ == '__main__':
    unittest.main()
