"""Audit task/scale ownership, shared gradients and actual model cost."""
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
from custom_models.factorized_head.head import FactorizedTowerBlock

ROOT = Path(__file__).resolve().parents[1]
register_all_modules()
torch.set_num_threads(2)


class FactorizedHeadTests(unittest.TestCase):
    def build(self):
        cfg = Config.fromfile(str(ROOT / 'configs/factorized_head/rtmdet_tiny_dior_640.py'))
        model = MODELS.build(deepcopy(cfg.model))
        logging.disable(logging.CRITICAL); model.init_weights()
        return model

    def assert_ownership(self, head):
        for name in ('cls_convs', 'reg_convs'):
            towers = getattr(head, name)
            for depth in range(2):
                for part in ('expand', 'spatial', 'project'):
                    layers = [getattr(tower[depth], part) for tower in towers]
                    self.assertEqual(len({id(layer.conv.weight) for layer in layers}), 1)
                    self.assertEqual(len({id(layer.norm) for layer in layers}), 3)
                    self.assertEqual(len({id(layer.norm.weight) for layer in layers}), 3)
        cls_ids = {id(p) for p in head.cls_convs.parameters()}
        self.assertFalse(cls_ids & {id(p) for p in head.reg_convs.parameters()})
        self.assertEqual(len({id(layer.weight) for layer in head.rtm_cls}), 3)

    def test_scope_parameters_macs_and_outputs(self):
        model = self.build().eval()
        self.assert_ownership(model.bbox_head.head_module)
        self.assertEqual(sum(isinstance(m, FactorizedTowerBlock) for m in model.modules()), 12)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4710516)
        macs, handles = [], []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                handles.append(layer.register_forward_hook(lambda m, a, y:
                    macs.append(y.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])))
        with torch.no_grad():
            cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
        for handle in handles: handle.remove()
        self.assertEqual(sum(macs), 6497475840)
        self.assertEqual([list(x.shape) for x in cls], [[1, 20, s, s] for s in (80, 40, 20)])
        self.assertEqual([list(x.shape) for x in reg], [[1, 4, s, s] for s in (80, 40, 20)])

    def test_cross_scale_gradient_sum_and_checkpoint_ema_sharing(self):
        head = self.build().bbox_head.head_module.eval()
        features = [torch.randn(2, 96, s, s) for s in (16, 8, 4)]
        weight = head.cls_convs[0][0].expand.conv.weight
        scores, _ = head(features)
        individual = [torch.autograd.grad(x.square().sum(), weight, retain_graph=True)[0] for x in scores]
        combined = torch.autograd.grad(sum(x.square().sum() for x in scores), weight)[0]
        torch.testing.assert_close(combined, sum(individual), atol=1e-10, rtol=2e-5)
        self.assertGreater(combined.abs().sum().item(), 0)
        ema = ExpMomentumEMA(head, momentum=.0002, update_buffers=True)
        ema.update_parameters(head)
        saved = io.BytesIO(); torch.save(ema.module.state_dict(), saved); saved.seek(0)
        restored = self.build().bbox_head.head_module.eval()
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        self.assert_ownership(restored); self.assert_ownership(ema.module)
        with torch.no_grad():
            for a, b in zip(restored(features), ema.module.eval()(features)):
                for x, y in zip(a, b): torch.testing.assert_close(x, y, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
