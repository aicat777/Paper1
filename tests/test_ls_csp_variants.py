"""Independent structure, initialization, legacy-equivalence and learning checks."""
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
from mmyolo.models.necks import CSPNeXtPAFPN
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.lsblock_csp.block import LSFeatureBlock
from custom_models.lsblock_csp.operator import LSConv as BlockLSConv
from custom_models.lsconv_csp.operator import LSConv as PlainLSConv

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('lsconv_csp', 'lsblock_csp')
register_all_modules()
torch.set_num_threads(2)


def build(variant, cfg=None):
    cfg = cfg or Config.fromfile(str(ROOT / 'configs' / variant / 'rtmdet_tiny_dior_640.py'))
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        model.init_weights()
    finally:
        logging.disable(previous)
    return model


def watched(model):
    ids = {id(p) for module in model.modules()
           if isinstance(module, (PlainLSConv, LSFeatureBlock)) for p in module.parameters()}
    return {n: p for n, p in model.named_parameters() if id(p) in ids}


class LSCSPVariants(unittest.TestCase):
    def test_operator_sources_are_exact_copies_of_verified_legacy_core(self):
        for variant in VARIANTS:
            for file in ('operator.py', 'aggregation.py'):
                actual = (ROOT / 'custom_models' / variant / file).read_bytes()
                reference = (ROOT / 'custom_models/lsconv' / file).read_bytes()
                self.assertEqual(hashlib.sha256(actual).hexdigest(), hashlib.sha256(reference).hexdigest())

    def test_legacy_three_stage_special_case_is_bitwise_identical(self):
        old_cfg = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
        cfg = Config.fromfile(str(ROOT / 'configs/lsconv_csp/rtmdet_tiny_dior_640.py'))
        cfg.model.backbone.lsconv_stages = (2, 3, 4)
        cfg.model.neck.enable_lsconv = False
        old, new = build('lsconv', old_cfg).eval(), build('lsconv_csp', cfg).eval()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], atol=0, rtol=0, msg=name)
        with torch.no_grad():
            image = torch.randn(1, 3, 128, 160)
            for group_a, group_b in zip(old._forward(image), new._forward(image)):
                for a, b in zip(group_a, group_b):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_exact_scope_channels_call_count_and_unchanged_detection_head(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = build(variant).eval()
                core_type = PlainLSConv if variant == 'lsconv_csp' else BlockLSConv
                cores = [(name, module) for name, module in model.named_modules() if isinstance(module, core_type)]
                self.assertEqual(len(cores), 8)
                self.assertEqual(sum(name.startswith('backbone.') for name, _ in cores), 4)
                self.assertEqual(sum(name.startswith('neck.') for name, _ in cores), 4)
                for name, _ in cores:
                    self.assertIn('.blocks.', name)
                if variant == 'lsconv_csp':
                    blocks = [m for m in model.modules() if isinstance(m, CSPNeXtBlock)]
                    self.assertEqual(len(blocks), 8)
                    for block in blocks:
                        self.assertIsInstance(block.conv1[0], PlainLSConv)
                        self.assertIsInstance(block.conv1[1], nn.SiLU)
                        self.assertEqual(block.conv2.depthwise_conv.conv.kernel_size, (5, 5))
                else:
                    self.assertFalse(any(isinstance(m, CSPNeXtBlock) for m in model.modules()))
                    self.assertEqual(sum(isinstance(m, LSFeatureBlock) for m in model.modules()), 8)
                    for block in (m for m in model.modules() if isinstance(m, LSFeatureBlock)):
                        self.assertFalse(any(isinstance(m, nn.SiLU) for m in block.modules()))
                        self.assertFalse(any(isinstance(m, nn.Conv2d) and m.kernel_size == (5, 5)
                                             for m in block.modules()))
                self.assertEqual([(layer.conv.in_channels, layer.conv.out_channels, layer.conv.kernel_size)
                                  for layer in model.neck.out_layers],
                                 [(96, 96, (3, 3)), (192, 96, (3, 3)), (384, 96, (3, 3))])
                head = model.bbox_head.head_module
                self.assertEqual(type(head).__name__, 'RTMDetSepBNHeadModule')
                self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in head.modules()), 12)
                self.assertFalse(any(isinstance(m, nn.GroupNorm) for m in head.modules()))
                for towers in (head.cls_convs, head.reg_convs):
                    for depth in range(2):
                        self.assertEqual(len({id(tower[depth].conv) for tower in towers}), 1)
                        self.assertEqual(len({id(tower[depth].bn) for tower in towers}), 3)
                calls = []
                handles = [m.register_forward_hook(lambda m, args, out: calls.append(tuple(out.shape))) for _, m in cores]
                with torch.no_grad():
                    cls, reg = model._forward(torch.randn(1, 3, 640, 640))
                for handle in handles:
                    handle.remove()
                self.assertEqual(len(calls), 8)
                self.assertEqual([c[1] for c in calls], [24, 48, 96, 192, 96, 48, 96, 192])
                self.assertEqual([tuple(x.shape) for x in cls], [(1, 20, s, s) for s in (80, 40, 20)])
                self.assertEqual([tuple(x.shape) for x in reg], [(1, 4, s, s) for s in (80, 40, 20)])
                priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls])
                self.assertEqual(sum(len(p) for p in priors), 8400)

    def test_parent_initialization_preserves_new_states_and_zero_ffn_scale(self):
        for variant in VARIANTS:
            cfg = Config.fromfile(str(ROOT / 'configs' / variant / 'rtmdet_tiny_dior_640.py'))
            model = MODELS.build(deepcopy(cfg.model))
            blocks = [(n, m) for n, m in model.named_modules()
                      if isinstance(m, (PlainLSConv, LSFeatureBlock))]
            initial = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in blocks}
            previous = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                model.init_weights()
                for n, m in blocks:
                    for k, value in m.state_dict().items():
                        torch.testing.assert_close(value, initial[n][k], atol=0, rtol=0, msg=n + '.' + k)
                state = {k: v.clone() for k, v in model.state_dict().items()}
                model.neck.init_weights()
                for k, v in model.state_dict().items():
                    torch.testing.assert_close(v, state[k], atol=0, rtol=0)
            finally:
                logging.disable(previous)
            for _, block in blocks:
                core = block.mixer if isinstance(block, LSFeatureBlock) else block
                self.assertGreater(float(core.lkp.cv4.bias.abs().max()), 0)
                self.assertEqual(float(core.bn.weight.min()), 1)
                if isinstance(block, LSFeatureBlock):
                    self.assertEqual(int(torch.count_nonzero(block.ffn.m.pw2.bn.weight)), 0)

    def test_neck_restores_modules_if_parent_initialization_raises(self):
        for variant in VARIANTS:
            cfg = Config.fromfile(str(ROOT / 'configs' / variant / 'rtmdet_tiny_dior_640.py'))
            neck = MODELS.build(deepcopy(cfg.model.neck))
            before = list(neck.named_modules())
            with patch.object(CSPNeXtPAFPN, 'init_weights', side_effect=RuntimeError('injected failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    neck.init_weights()
            self.assertEqual([(n, id(m)) for n, m in before], [(n, id(m)) for n, m in neck.named_modules()])

    def test_zero_initialized_ffn_is_identity_then_learns_all_parameters(self):
        torch.manual_seed(37)
        block = LSFeatureBlock(24).train()
        image = torch.randn(2, 24, 7, 9)
        with torch.no_grad():
            z = block.mixer(image)
            torch.testing.assert_close(block(image), z, atol=0, rtol=0)
        opt = torch.optim.AdamW(block.parameters(), lr=.001, weight_decay=0)
        initial = {n: p.clone() for n, p in block.named_parameters()}
        grads = {}
        for step in range(3):
            opt.zero_grad()
            loss = (block(image) * torch.randn_like(image)).mean()
            loss.backward()
            if step == 0:
                self.assertEqual(float(block.ffn.m.pw1.c.weight.grad.abs().max()), 0)
                self.assertGreater(float(block.ffn.m.pw2.bn.weight.grad.abs().max()), 0)
            for n, p in block.named_parameters():
                self.assertTrue(torch.isfinite(p.grad).all(), n)
                grads[n] = max(grads.get(n, 0), float(p.grad.abs().max()))
            opt.step()
        for n, p in block.named_parameters():
            self.assertGreater(grads[n], 0, n)
            self.assertGreater(float((p - initial[n]).abs().max()), 0, n)

    def test_full_detection_gradients_and_state_roundtrip(self):
        for variant in VARIANTS:
            model = build(variant).train()
            params = watched(model)
            initial = {n: p.clone() for n, p in params.items()}
            # Controlled coverage test only: decoded boxes span all GT scales.
            with torch.no_grad():
                for conv in model.bbox_head.head_module.rtm_reg:
                    conv.bias.fill_(2.)
            rows = [[i, label, *box] for i in range(2) for label, box in enumerate(
                    [(32, 32, 48, 48), (64, 64, 144, 144), (8, 8, 248, 248)])]
            samples = dict(bboxes_labels=torch.tensor(rows, dtype=torch.float32),
                           img_metas=[dict(img_shape=(256, 256)) for _ in range(2)])
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0005625)
            gradients = {}
            for _ in range(3):
                optimizer.zero_grad()
                loss = sum(model.loss(torch.randn(2, 3, 256, 256), samples).values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for n, p in params.items():
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.isfinite(p.grad).all(), n)
                    gradients[n] = max(gradients.get(n, 0), float(p.grad.abs().max()))
                optimizer.step()
            for n, p in params.items():
                self.assertGreater(gradients[n], 0, n)
                self.assertGreater(float((p - initial[n]).abs().max()), 0, n)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = build(variant)
            restored.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
            model.eval(); restored.eval()
            with torch.no_grad():
                image = torch.randn(1, 3, 128, 160)
                for a, b in zip(model._forward(image), restored._forward(image)):
                    for x, y in zip(a, b):
                        torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_both_configs_match_legacy_batch36_training_protocol(self):
        base = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
        for variant in VARIANTS:
            cfg = Config.fromfile(str(ROOT / 'configs' / variant / 'rtmdet_tiny_dior_640.py'))
            for key in ('train_dataloader', 'train_pipeline', 'train_pipeline_stage2', 'test_pipeline',
                        'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator', 'test_dataloader',
                        'test_evaluator', 'custom_hooks', 'optim_wrapper', 'param_scheduler',
                        'default_hooks', 'randomness'):
                self.assertEqual(cfg[key], base[key], (variant, key))
            self.assertEqual(cfg.model.bbox_head, base.model.bbox_head)
            self.assertEqual(cfg.model.train_cfg, base.model.train_cfg)
            self.assertEqual(cfg.train_dataloader.batch_size, 36)
            self.assertEqual(cfg.optim_wrapper.get('accumulative_counts', 1), 1)
            self.assertIsNone(cfg.model.backbone.init_cfg)
            self.assertIsNone(cfg.load_from)
            self.assertFalse(cfg.resume)


if __name__ == '__main__':
    unittest.main()
