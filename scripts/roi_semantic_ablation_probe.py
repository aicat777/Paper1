"""Isolated checks for three ROI mechanism controls; official training entrypoint.

Methods use separate model/config packages. This script only shares validation
plumbing and never edits experiments or schedules full training."""
import argparse
from copy import deepcopy
import hashlib
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

from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('instance_semantic_center', 'instance_semantic_matched', 'instance_semantic_random')
METHOD = None


def watched_parameters(model):
    prefixes = ('bbox_head.head_module.rtm_cls.', 'bbox_head.head_module.cls_convs.',
                'bbox_head.instance_semantic.')
    return {name: p for name, p in model.named_parameters() if name.startswith(prefixes)}


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Allocator limit must leave GPU memory headroom')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    return free, total


def config_identity(path):
    path = Path(path).resolve()
    return dict(config=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preflight(cfg, output, path):
    original = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks', 'train_pipeline',
                'train_pipeline_stage2', 'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == original[key], key
    expected_model = deepcopy(original.model)
    head_types = dict(instance_semantic_center='CenterInstanceSemanticRTMDetHead',
                      instance_semantic_matched='MatchedInstanceSemanticRTMDetHead',
                      instance_semantic_random='RandomInstanceSemanticRTMDetHead')
    expected_model.bbox_head.type = head_types[METHOD]
    if METHOD != 'instance_semantic_center':
        expected_model.train_cfg.assigner.type = 'ROISemanticIndexedAssigner'
    if METHOD == 'instance_semantic_random':
        expected_model.bbox_head.semantic_selection_seed = 42
    assert cfg.model == expected_model
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    models = []
    for candidate in (original, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(candidate.model))
        model.init_weights(); model.eval()
        models.append(model)
    reference, model = models
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    macs, handles = [0], []

    def count(module, inputs, result):
        per_output = (module.in_channels // module.groups) * math.prod(module.kernel_size) \
            if isinstance(module, torch.nn.Conv2d) else module.in_features
        macs[0] += result.numel() * per_output

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
        expected = reference._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    for a, b in zip((cls, reg), expected):
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert sum(p.numel() for p in model.parameters()) == 4880648
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    assert math.isclose(macs[0] / 1e9, 7.98770304, abs_tol=1e-10)
    for loader, expected_count in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected_count
    report_to(output, dict(status='passed', method=METHOD, parameters=4880648,
        new_parameters=0, inference_conv_linear_gmac=macs[0] / 1e9,
        inference_matches_original_exactly=True, train_images=5862, val_images=5863,
        prediction_points=8400, input_size=[640, 640], formal_batch=36, accumulation=1,
        scratch=True, unchanged_training_protocol=True, semantic_weight=.1, all_valid_gt_denominator=True,
        different_run_matched_counts_may_diverge=True,
        formal_batch_capacity_tested=False, **config_identity(path)))


@HOOKS.register_module()
class ROISemanticAblationAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, value in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            value.register_hook(capture)
        self.assignment_calls, self.coverage = 0, []
        head = self.model.bbox_head

        def assignment(module, inputs, result):
            self.assignment_calls += 1
            if self.method != 'instance_semantic_center':
                assert torch.equal(result['assigned_gt_indices'] >= 0,
                                   result['assigned_labels'] < head.num_classes)

        def selection(module, inputs, selected):
            matched = inputs[0]
            assert int(selected.sum()) == int(matched.sum())
            if self.method == 'instance_semantic_matched':
                assert torch.equal(selected, matched)
            self.coverage.append(dict(all_gt=len(matched), matched=int(matched.sum()),
                selected=int(selected.sum()), selected_unmatched=int((selected & ~matched).sum())))

        def center(module, inputs):
            rows = inputs[1]
            valid = (rows[:, 4:6] > rows[:, 2:4]).all(-1)
            self.coverage.append(dict(all_gt=int(valid.sum()), selected=int(valid.sum())))

        self.handles = [head.assigner.register_forward_hook(assignment)]
        if self.method == 'instance_semantic_center':
            self.handles.append(head.instance_semantic.register_forward_pre_hook(center))
        else:
            self.handles.append(head.selection.register_forward_hook(selection))
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def after_train(self, runner):
        super().after_train(runner)
        for handle in self.handles:
            handle.remove()
        assert self.assignment_calls == len(self.coverage) == 16
        assert all('loss_instance_semantic' in values for values in self.losses)
        report = json.loads(Path(self.report_path).read_text())
        report.update(assignment_calls=self.assignment_calls, one_assignment_per_train_step=True,
                      coverage=self.coverage, all_valid_gt_denominator=True,
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
    by_image = {row['id']: set() for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].add(row['category_id'])
    selected = []
    while categories:
        image = max(by_image, key=lambda key: len(by_image[key] & categories))
        assert by_image[image] & categories
        selected.append(image)
        categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.roi_semantic_ablation_probe')
    cfg.custom_hooks.append(dict(type='ROISemanticAblationAuditHook', method=METHOD,
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
    batch = batch or 36
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    optim_cfg = deepcopy(cfg.optim_wrapper)
    optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model)
    initial = {name: p.detach().cpu().clone() for name, p in watched.items()}
    started, losses = time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {name: float((p.detach().cpu() - initial[name]).abs().max()) for name, p in watched.items()}
    assert all(updates[name] > 0 for name in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=METHOD, microbatch=batch, accumulation=1,
        optimizer_updates=3, gt_per_image=1200, different_image_ids=image_ids,
        repeated_annotations=True, synthetic_multiscale_boxes=True,
        formal_batch=36, formal_batch_capacity_tested=batch == 36,
        allocator_fraction=fraction, loss_scale_init=32, warmup_enabled=False,
        amp_skipped_updates=0, saved_checkpoints=False, losses=losses,
        watched_parameter_tensors=len(watched), parameter_updates_max=updates,
        cuda_device=torch.cuda.get_device_name(), cuda_total_gib=total / 2**30,
        initial_free_gib=free / 2**30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        elapsed_seconds=time.perf_counter() - started, **config_identity(path)))
    print('CAPACITY_PASSED', batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'capacity'), required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config')
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    args = parser.parse_args()
    global METHOD
    METHOD = args.method
    args.config = args.config or str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py')
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
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
