"""Check the actual alpha=1 config, fourfold ROI derivative and AdamW reference."""
from copy import deepcopy
import unittest

import torch

from tests.test_instance_semantic_clstower import (
    assert_gradient, config, data, grads)
from scripts.instance_semantic_neckhead_probe import build

METHOD = 'scdown_p3_g4_semantic_clstower_alpha1'
PREVIOUS = 'scdown_p3_g4_semantic_clstower'
REFERENCE = 'scdown_p3_g4_semantic_gradscale'


class ClassificationTowerAlphaOneTests(unittest.TestCase):
    def test_actual_config_changes_only_alpha_and_output_location(self):
        actual, previous = config(METHOD), config(PREVIOUS)
        expected = deepcopy(previous.model)
        expected.bbox_head.semantic_feature_gradient_scale = 1.
        self.assertEqual(actual.model, expected)
        excluded = {'model', 'work_dir'}
        for key in previous:
            if key not in excluded:
                self.assertEqual(actual[key], previous[key], key)
        first, second = build(actual).eval(), build(previous).eval()
        self.assertEqual(first.bbox_head.instance_semantic.feature_gradient_scale, 1.)
        self.assertEqual(first.bbox_head.instance_semantic.loss_weight, .1)
        self.assertEqual(sum(p.numel() for p in first.parameters()), 3105704)
        for name, value in second.state_dict().items():
            torch.testing.assert_close(first.state_dict()[name], value, atol=0, rtol=0)
        images, _ = data()
        with torch.no_grad():
            for a, b in zip(first._forward(images), second._forward(images)):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_fourfold_cls_roi_same_aux_classifier_and_main_gradients(self):
        images, samples = data()
        model, previous = build(config(METHOD)).train(), build(config(PREVIOUS)).train()
        actual, old = model.loss(images, samples), previous.loss(images, samples)
        for key in actual:
            torch.testing.assert_close(actual[key], old[key], atol=0, rtol=0)
        for name in ('loss_cls', 'loss_bbox'):
            ag = grads(actual[name], model, retain_graph=True)
            eg = grads(old[name], previous, retain_graph=True)
            for (key, _), a, e in zip(model.named_parameters(), ag, eg):
                assert_gradient(self, a, e, key, exact=True)
        ag, eg = grads(actual['loss_instance_semantic'], model), grads(old['loss_instance_semantic'], previous)
        active = 0
        for (name, _), a, e in zip(model.named_parameters(), ag, eg):
            if name.startswith('bbox_head.head_module.cls_convs.'):
                assert_gradient(self, a, 4 * e, name, exact=True)
                active += int(a is not None and a.abs().sum() > 0)
            elif name.startswith('bbox_head.instance_semantic.'):
                assert_gradient(self, a, e, name, exact=True)
            else:
                assert_gradient(self, a, None, name)
        self.assertEqual(active, 14)

    def test_independent_unrouted_alpha1_gradient_and_adamw_update(self):
        images, samples = data()
        cfg = config(REFERENCE)
        cfg.model.bbox_head.semantic_feature_gradient_scale = 1.
        reference, model = build(cfg).train(), build(config(METHOD)).train()
        expected = reference.loss(images, samples)
        main = grads(expected['loss_cls'] + expected['loss_bbox'], reference, retain_graph=True)
        aux = grads(expected['loss_instance_semantic'], reference)
        actual = model.loss(images, samples)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        sum(actual.values()).backward()
        for (name, ref), actual_p, gm, ga in zip(
                reference.named_parameters(), model.parameters(), main, aux):
            selected = name.startswith(('bbox_head.head_module.cls_convs.', 'bbox_head.instance_semantic.'))
            wanted = gm if not selected or ga is None else (ga if gm is None else gm + ga)
            assert_gradient(self, actual_p.grad, wanted, name)
            ref.grad = wanted
        for candidate in (reference, model):
            torch.optim.AdamW(candidate.parameters(), lr=.0005625).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=2e-7, rtol=3e-5, msg=name)


if __name__ == '__main__':
    unittest.main()
