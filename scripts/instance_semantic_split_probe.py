"""Checks for all-GT classifier learning with matched-only feature gradients."""
import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from custom_models.instance_semantic_split.supervision import MatchedFeatureGradientGate, split_roi_loss
from custom_models.roi_supervision_common.supervision import selected_roi_loss
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.roi_semantic_ablation_probe import config_identity, require_memory, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'instance_semantic_split'


def preflight(cfg, output, path):
    original = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks', 'train_pipeline',
                'train_pipeline_stage2', 'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == original[key], key
    expected = deepcopy(original.model)
    expected.bbox_head.type = 'SplitGradientInstanceSemanticRTMDetHead'
    expected.train_cfg.assigner.type = 'ROISemanticIndexedAssigner'
    assert cfg.model == expected
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    models = []
    for source in (original, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(source.model))
        model.init_weights(); model.eval()
        models.append(model)
    reference, model = models
    assert set(reference.state_dict()) == set(model.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    macs, handles = [0], []

    def count(layer, inputs, result):
        per_output = (layer.in_channels // layer.groups) * math.prod(layer.kernel_size) \
            if isinstance(layer, torch.nn.Conv2d) else layer.in_features
        macs[0] += result.numel() * per_output

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        actual = model._forward(torch.zeros(1, 3, 640, 640))
        expected_outputs = reference._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    for group, expected_group in zip(actual, expected_outputs):
        for a, b in zip(group, expected_group):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert [list(x.shape) for x in actual[0]] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in actual[1]] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert sum(p.numel() for p in model.parameters()) == 4880648
    assert math.isclose(macs[0] / 1e9, 7.98770304, abs_tol=1e-10)
    for loader, expected_count in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected_count
    report_to(output, dict(status='passed', method=METHOD, parameters=4880648,
        new_parameters=0, inference_conv_linear_gmac=macs[0] / 1e9,
        initial_parameters_and_predictions_exact=True, train_images=5862, val_images=5863,
        prediction_points=8400, input_size=[640, 640], formal_batch=36, accumulation=1,
        scratch=True, unchanged_training_protocol=True, semantic_weight=.1,
        classifier_coverage='all_valid_gt', feature_gradient_coverage='matched_valid_gt',
        all_valid_gt_denominator=True, formal_batch_capacity_tested=False, **config_identity(path)))


def fidelity(output, fraction):
    require_memory(fraction)
    cases = []
    for amp in (False, True):
        dtype = torch.float16 if amp else torch.float32
        for name, flags in [('all', [1, 1, 1, 1, 1]), ('mixed', [0, 1, 0, 1, 0]), ('none', [0, 0, 0, 0, 0])]:
            torch.manual_seed(8)
            module = InstanceSemanticLoss(96, 20).cuda()
            features = [torch.randn(2, 96, s, s, device='cuda', dtype=dtype, requires_grad=True)
                        for s in (80, 40, 20)]
            rows = torch.tensor([[1, 1, 8.25, 8.5, 9.25, 9.5], [0, 1, 16, 16, 48, 48],
                                 [0, 2, 16, 16, 48, 48], [1, 0, 32, 32, 160, 160],
                                 [0, 2, 48, 48, 368, 368]], device='cuda', dtype=torch.float32)
            matched = torch.tensor(flags, device='cuda', dtype=torch.bool)
            with torch.autocast('cuda', enabled=amp):
                actual = split_roi_loss(module, MatchedFeatureGradientGate(), features, rows, matched)
                full = module(features, rows)
                selected = selected_roi_loss(module, features, rows, matched)
            torch.testing.assert_close(actual, full, atol=0, rtol=0)
            params = (*features, *module.parameters())
            groups = [torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                      for loss in (actual, full, selected)]
            maximum = 0
            for i, a in enumerate(groups[0]):
                expected = groups[1][i] if i >= len(features) else groups[2][i]
                if expected is None:
                    assert a is None or not a.any()
                elif a is None:
                    assert not expected.any()
                else:
                    torch.testing.assert_close(a, expected, atol=2e-6, rtol=5e-4)
                    maximum = max(maximum, float((a.float()-expected.float()).abs().max()))
            assert all(torch.isfinite(g).all() for g in groups[0] if g is not None)
            assert groups[0][-2].abs().sum() > 0
            cases.append(dict(amp=amp, mask=name, forward_error=0, gradient_max_error=maximum))
    report_to(output, dict(status='passed', method=METHOD, cases=cases,
        classifier_gradient_reference='original_all_gt', feature_gradient_reference='matched_only_all_gt_denominator'))


@HOOKS.register_module()
class SplitROISemanticAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {n:p.detach().cpu().clone() for n,p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, value in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            value.register_hook(capture)
        self.assignment_calls, self.coverage = 0, []

        def assignment(module, inputs, result):
            self.assignment_calls += 1
            assert torch.equal(result['assigned_gt_indices'] >= 0,
                               result['assigned_labels'] < self.model.bbox_head.num_classes)

        def routing(module, inputs, routed):
            pooled, matched = inputs
            torch.testing.assert_close(pooled, routed, atol=0, rtol=0)
            flags = matched.detach().clone()
            row = dict(all_gt=len(flags), classifier_gt=len(flags), feature_gt=int(flags.sum()),
                       unmatched_gt=int((~flags).sum()), gradient_checked=False)
            self.coverage.append(row)
            expected_gradient = []

            def after_gate(grad):
                assert torch.isfinite(grad).all()
                expected_gradient.append(grad.detach() * flags[:, None])
                row['unmatched_classifier_input_gradient_l1'] = float(grad[~flags].abs().sum())

            def before_gate(grad):
                assert expected_gradient
                torch.testing.assert_close(grad, expected_gradient.pop(), atol=0, rtol=0)
                assert not grad[~flags].any()
                row['gradient_checked'] = True

            routed.register_hook(after_gate)
            pooled.register_hook(before_gate)

        head = self.model.bbox_head
        self.handles = [head.assigner.register_forward_hook(assignment),
                        head.gradient_gate.register_forward_hook(routing)]
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def after_train(self, runner):
        super().after_train(runner)
        for handle in self.handles:
            handle.remove()
        assert self.assignment_calls == len(self.coverage) == 16
        assert all(row['gradient_checked'] for row in self.coverage)
        assert any(row['unmatched_gt'] and row['unmatched_classifier_input_gradient_l1'] > 0
                   for row in self.coverage)
        report = json.loads(Path(self.report_path).read_text())
        report.update(assignment_calls=self.assignment_calls, one_assignment_per_train_step=True,
            coverage=self.coverage, unmatched_descriptor_gradients_zero=True,
            unmatched_gt_retained_in_classifier_loss=True, all_valid_gt_denominator=True,
            no_new_parameters=True, inference_unchanged=True)
        report_to(self.report_path, report)


def smoke(cfg, output, fraction):
    directory = Path(output).parent / 'smoke'
    if directory.exists():
        raise RuntimeError('Existing short-test outputs would be overwritten')
    directory.mkdir(parents=True)
    cfg.work_dir = str(directory)
    cfg.train_cfg.update(max_epochs=2, val_interval=1, dynamic_intervals=[])
    cfg.param_scheduler = []
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger.interval = 1
    cfg.optim_wrapper.loss_scale = dict(init_scale=32)
    cfg.custom_hooks[1].switch_epoch = 1
    cfg.train_dataloader.update(batch_size=2, num_workers=0, persistent_workers=False)
    cfg.train_dataloader.dataset.indices = list(range(16))
    cfg.val_dataloader.update(batch_size=4, num_workers=0, persistent_workers=False)
    annotation = json.loads((Path(cfg.data_root) / cfg.val_dataloader.dataset.ann_file).read_text())
    categories = {row['id'] for row in annotation['categories']}
    by_image = {row['id']:set() for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].add(row['category_id'])
    selected = []
    while categories:
        image = max(by_image, key=lambda key:len(by_image[key] & categories))
        assert by_image[image] & categories
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.instance_semantic_split_probe')
    cfg.custom_hooks.append(dict(type='SplitROISemanticAuditHook', method=METHOD,
        report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (METHOD + '_smoke.py')
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', METHOD, flush=True)


def capacity(cfg, output, fraction, batch, path):
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or 36
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    optim_cfg = deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model)
    initial = {n:p.detach().cpu().clone() for n,p in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients = {}
    for name, parameter in watched.items():
        def capture(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        parameter.register_hook(capture)
    started, losses = time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        values = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in values.values())
        assert values['semantic_classifier_gt'] == values['semantic_all_gt']
        assert values['semantic_feature_gt'] == values['semantic_matched_gt']
        losses.append({key:float(value) for key,value in values.items()})
        torch.cuda.synchronize()
    updates = {name:float((p.detach().cpu()-initial[name]).abs().max()) for name,p in watched.items()}
    assert all(gradients.get(name, 0)>0 and updates[name]>0 for name in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=METHOD, microbatch=batch, accumulation=1,
        optimizer_updates=3, gt_per_image=1200, different_image_ids=image_ids,
        repeated_annotations=True, synthetic_multiscale_boxes=True, formal_batch=36,
        formal_batch_capacity_tested=batch==36, allocator_fraction=fraction,
        loss_scale_init=32, warmup_enabled=False, amp_skipped_updates=0, saved_checkpoints=False,
        watched_parameter_tensors=len(watched), gradient_max=gradients, parameter_updates_max=updates,
        losses=losses, cuda_device=torch.cuda.get_device_name(), cuda_total_gib=total/2**30,
        initial_free_gib=free/2**30, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        elapsed_seconds=time.perf_counter()-started, **config_identity(path)))
    print('CAPACITY_PASSED', batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.28)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            fidelity(args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.config)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=METHOD, mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
