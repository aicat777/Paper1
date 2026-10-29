"""Integration checks for the original ROI task on the four-site SCDown and seven-site G4 model."""
from copy import deepcopy
import logging
from pathlib import Path
import unittest

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.instance_semantic.roi import InstanceSemanticLoss

ROOT = Path(__file__).resolve().parents[1]
SITES = ('backbone.stage2.0', 'backbone.stage3.0', 'backbone.stage4.0', 'neck.downsample_layers.1')
GROUP_SITES = ('backbone.stage2.1.blocks.0.conv1', 'backbone.stage3.1.blocks.0.conv1',
               'backbone.stage4.2.blocks.0.conv1', 'neck.top_down_layers.0.0.blocks.0.conv1',
               'neck.top_down_layers.1.blocks.0.conv1', 'neck.bottom_up_layers.0.blocks.0.conv1',
               'neck.bottom_up_layers.1.blocks.0.conv1')
register_all_modules()
torch.set_num_threads(2)
logging.disable(logging.CRITICAL)


def config(method):
    return Config.fromfile(str(ROOT / 'configs' / method / 'rtmdet_tiny_dior_640.py'))


def build(method, weight=.1):
    cfg = config(method)
    if method == 'scdown_p3_g4_semantic':
        cfg.model.bbox_head.semantic_loss_weight = weight
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model


def data(empty=False):
    torch.manual_seed(13)
    images = torch.randn(2, 3, 256, 256)
    # Include an unmatched duplicate GT, interleaved image rows, every ROI
    # level, fractional sub-stride ROI, and a degenerate row excluded only by CE.
    rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
                         [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
                         [0, 3, 8, 8, 248, 248], [1, 4, 4.25, 5.5, 5.25, 6.5],
                         [0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    return images, dict(bboxes_labels=rows[:0] if empty else rows,
                        img_metas=[dict(img_shape=(256, 256))] * 2)


class SCDownP3G4SemanticTests(unittest.TestCase):
    def test_protocol_and_model_are_exact_composition_of_existing_methods(self):
        combined, scdown, roi = [config(m) for m in ('scdown_p3_g4_semantic', 'scdown_p3_g4', 'instance_semantic')]
        expected = deepcopy(scdown.model)
        expected.bbox_head = deepcopy(roi.model.bbox_head)
        self.assertEqual(combined.model, expected)
        for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                    'custom_hooks', 'param_scheduler', 'optim_wrapper', 'default_hooks', 'randomness'):
            self.assertEqual(combined[key], scdown[key], key)
            self.assertEqual(combined[key], roi[key], key)
        self.assertIsNone(combined.load_from)
        self.assertFalse(combined.resume)
        self.assertIsNone(combined.model.backbone.init_cfg)

    def test_initialization_inference_scdown_sites_and_parameter_sharing(self):
        reference = build('scdown_p3_g4').eval()
        rng = torch.get_rng_state().clone()
        model = build('scdown_p3_g4_semantic').eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
        extras = set(model.state_dict()) - set(reference.state_dict())
        self.assertEqual(extras, {'bbox_head.instance_semantic.classifier.weight', 'bbox_head.instance_semantic.classifier.bias'})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3105704)
        self.assertEqual(tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)), SITES)
        grouped = tuple(n.removesuffix('.conv') for n, layer in model.named_modules()
                        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 4)
        self.assertEqual(set(grouped), set(GROUP_SITES))
        for site, channels in zip(GROUP_SITES, (48, 96, 192, 96, 48, 96, 192)):
            layer = model.get_submodule(site).conv
            self.assertEqual((layer.in_channels, layer.out_channels, layer.groups), (channels, channels, 4))
            self.assertEqual(layer.kernel_size, (3, 3))
            self.assertEqual(layer.stride, (1, 1))
        self.assertIs(type(model.bbox_head.instance_semantic), InstanceSemanticLoss)
        self.assertEqual(model.bbox_head.instance_semantic.output_size, 3)
        head = model.bbox_head.head_module
        for level in (1, 2):
            for i in range(2):
                self.assertIs(head.cls_convs[0][i].conv, head.cls_convs[level][i].conv)
                self.assertIsNot(head.cls_convs[0][i].bn, head.cls_convs[level][i].bn)
        images, _ = data()
        with torch.no_grad():
            for first, second in zip(reference._forward(images), model._forward(images)):
                for a, b in zip(first, second):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_main_losses_unchanged_and_full_gt_ce_independent_reference(self):
        images, samples = data()
        reference, model = build('scdown_p3_g4').train(), build('scdown_p3_g4_semantic').train()
        captured = {}
        assign_calls, tower_calls = [], []
        handles = [model.bbox_head.assigner.register_forward_hook(lambda *args: assign_calls.append(1)),
                   model.bbox_head.head_module.register_forward_hook(lambda *args: tower_calls.append(1)),
                   model.bbox_head.instance_semantic.register_forward_pre_hook(
                       lambda module, inputs: captured.update(features=inputs[0], rows=inputs[1]))]
        actual, expected = model.loss(images, samples), reference.loss(images, samples)
        for handle in handles:
            handle.remove()
        self.assertEqual(assign_calls, [1]); self.assertEqual(tower_calls, [1])
        for key in ('loss_cls', 'loss_bbox'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        rows = captured['rows']; rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(-1)]
        self.assertEqual(len(rows), 6)
        module = model.bbox_head.instance_semantic
        rois = torch.cat((rows[:, :1], rows[:, 2:]), 1)
        pooled = module.pool_instances(captured['features'], rois)
        expected_aux = .1 * torch.nn.functional.cross_entropy(module.classifier(pooled), rows[:, 1].long())
        torch.testing.assert_close(actual['loss_instance_semantic'], expected_aux, atol=2e-7, rtol=2e-6)
        self.assertEqual(module.loss_weight, .1)
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                self.assertEqual(int(layer.num_batches_tracked), 1, name)

    def test_auxiliary_gradient_reaches_all_scdown_and_g4_sites_but_excludes_predictors(self):
        images, samples = data()
        model = build('scdown_p3_g4_semantic').train()
        loss = model.loss(images, samples)['loss_instance_semantic']
        loss.backward()
        for site in SITES + GROUP_SITES:
            parameters = list(model.get_submodule(site).parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), site)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters), 0, site)
        head = model.bbox_head.head_module
        self.assertTrue(all(p.grad is None for p in (*head.rtm_cls.parameters(), *head.rtm_reg.parameters(), *head.reg_convs.parameters())))
        self.assertGreater(float(model.bbox_head.instance_semantic.classifier.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.cls_convs[0][0].conv.weight.grad.abs().sum()), 0)

    def test_zero_weight_preserves_detection_gradients_and_sgd_step(self):
        images, samples = data()
        reference, model = build('scdown_p3_g4').train(), build('scdown_p3_g4_semantic', 0).train()
        for candidate in (reference, model):
            losses = candidate.loss(images, samples)
            sum(v for k, v in losses.items() if 'loss' in k).backward()
        parameters = dict(model.named_parameters())
        for name, p in reference.named_parameters():
            if p.grad is None:
                self.assertIsNone(parameters[name].grad)
            else:
                torch.testing.assert_close(parameters[name].grad, p.grad, atol=2e-7, rtol=3e-5, msg=name)
        for candidate in (reference, model):
            torch.optim.SGD(candidate.parameters(), lr=.001).step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], atol=2e-7, rtol=2e-6, msg=name)

    def test_empty_gt_preserves_original_detector_losses_and_connected_auxiliary_zero(self):
        images, samples = data(empty=True)
        a, b = build('scdown_p3_g4').train(), build('scdown_p3_g4_semantic').train()
        first, second = a.loss(images, samples), b.loss(images, samples)
        for key in ('loss_cls', 'loss_bbox'):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        self.assertEqual(float(second['loss_instance_semantic']), 0)
        second['loss_instance_semantic'].backward()
        self.assertTrue(all(p.grad is not None and not p.grad.any() for p in b.bbox_head.instance_semantic.parameters()))


if __name__ == '__main__':
    unittest.main()
