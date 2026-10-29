"""Isolated ROI gradient-route checks; no full training or queue mutation."""
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
from custom_models.instance_semantic_gradscale import GradientScaledInstanceSemanticLoss
from custom_models.scdown import SCDown
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ('instance_semantic', 'scdown_p3_semantic', 'scdown_p3_g4_semantic')
METHODS = tuple(name + '_gradscale' for name in REFERENCES)


def identity(cfg):
    backbone = cfg.model.backbone.type
    if backbone == 'SCDownP3G4CSPNeXt':
        return METHODS[2], REFERENCES[2], 3105704, 6.78524544
    if backbone == 'SCDownP3CSPNeXt':
        return METHODS[1], REFERENCES[1], 3821096, 7.48197504
    if backbone == 'CSPNeXt':
        return METHODS[0], REFERENCES[0], 4880648, 7.98770304
    raise ValueError('Unsupported reference backbone: ' + backbone)


def watched_parameters(model):
    prefixes = ['bbox_head.head_module.rtm_cls.', 'bbox_head.head_module.cls_convs.',
                'bbox_head.instance_semantic.']
    for name, module in model.named_modules():
        if isinstance(module, SCDown):
            prefixes.append(name + '.')
        elif isinstance(module, torch.nn.Conv2d) and module.groups == 4 and module.kernel_size == (3, 3):
            prefixes.append(name.rpartition('.')[0] + '.')
    return {name: p for name, p in model.named_parameters() if name.startswith(tuple(prefixes))}


def preflight(cfg, output, path):
    method, ref, total, gmac = identity(cfg)
    reference_cfg = Config.fromfile(str(ROOT / 'configs' / ref / 'rtmdet_tiny_dior_640.py'))
    for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                'train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'optim_wrapper',
                'param_scheduler', 'default_hooks', 'randomness', 'env_cfg'):
        assert cfg[key] == reference_cfg[key], key
    expected = deepcopy(reference_cfg.model)
    expected.bbox_head.update(type='GradientScaledSemanticRTMDetHead',
                              semantic_feature_gradient_scale=.25)
    assert cfg.model == expected
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    models = []
    for source in (reference_cfg, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(source.model))
        model.init_weights(); model.eval()
        models.append(model)
    reference, model = models
    assert set(reference.state_dict()) == set(model.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)
    macs, handles = [0], []
    def count(module, inputs, result):
        per_output = (module.in_channels // module.groups) * math.prod(module.kernel_size) \
            if isinstance(module, torch.nn.Conv2d) else module.in_features
        macs[0] += result.numel() * per_output
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        inputs = torch.zeros(1, 3, 640, 640)
        actual, wanted = model._forward(inputs), reference._forward(inputs)
    for handle in handles:
        handle.remove()
    for a, b in zip(actual, wanted):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert [list(x.shape) for x in actual[0]] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in actual[1]] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert sum(p.numel() for p in model.parameters()) == total
    assert math.isclose(macs[0] / 1e9, gmac, abs_tol=1e-10)
    for loader, count_images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        ann = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(ann['images']) == count_images
    report_to(output, dict(status='passed', method=method, reference=ref,
        parameters=total, inference_parameters=total - 1940, new_parameters=0,
        inference_conv_linear_gmac=gmac, state_and_initialization_match_reference=True,
        inference_matches_reference_exactly=True, main_losses_and_assignment_unchanged=True,
        semantic_weight=.1, feature_gradient_scale=.25, all_valid_gt_preserved=True,
        classifier_weight_unchanged=True, single_detection_tower_forward=True,
        train_images=5862, val_images=5863, input_size=[640, 640], formal_batch=36,
        accumulation=1, scratch=True, unchanged_training_protocol=True,
        watched_parameter_tensors=len(watched_parameters(model)),
        scdown_sites=[n for n, m in model.named_modules() if isinstance(m, SCDown)],
        g4_sites=[n for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)
                  and m.groups == 4 and m.kernel_size == (3, 3)],
        formal_batch_capacity_tested=False, **config_identity(path)))


def fidelity(output, fraction):
    require_memory(fraction)
    torch.cuda.set_per_process_memory_fraction(fraction)
    records = []
    for amp in (False, True):
        for alpha in (0., .25, 1.):
            torch.manual_seed(7)
            reference = InstanceSemanticLoss(96, 20).cuda()
            model = GradientScaledInstanceSemanticLoss(deepcopy(reference), alpha)
            dtype = torch.float16 if amp else torch.float32
            features = [torch.randn(2, 96, s, s, device='cuda', dtype=dtype, requires_grad=True)
                        for s in (80, 40, 20)]
            copies = [x.detach().clone().requires_grad_() for x in features]
            rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
                [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
                [0, 3, 8, 8, 600, 600], [1, 4, 4.25, 5.5, 5.25, 6.5],
                [0, 0, 0, 0, 0, 0]], dtype=torch.float32, device='cuda')
            with torch.autocast('cuda', enabled=amp):
                actual, wanted = model(features, rows), reference(copies, rows)
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
            grads = torch.autograd.grad(actual, (*features, *model.parameters()))
            refs = torch.autograd.grad(wanted, (*copies, *reference.parameters()))
            feature_error = classifier_error = 0.
            for i, (a, b) in enumerate(zip(grads, refs)):
                expected = alpha * b if i < len(features) else b
                torch.testing.assert_close(a, expected, atol=2e-7, rtol=3e-3 if amp else 3e-5)
                error = float((a.float() - expected.float()).abs().max())
                if i < len(features):
                    feature_error = max(feature_error, error)
                else:
                    classifier_error = max(classifier_error, error)
            records.append(dict(amp=amp, alpha=alpha, forward_error=float((actual-wanted).abs()),
                feature_gradient_max_error=feature_error, classifier_gradient_max_error=classifier_error))
    report_to(output, dict(status='passed', cases=records, all_gt_and_three_scales=True,
                          comparisons='Original ROI autograd; closed-form vector derivatives in unit tests'))


@HOOKS.register_module()
class ROIGradientScaleAuditHook(CandidateAuditHook):
    def __init__(self, actual_method, **kwargs):
        super().__init__(method='instance_semantic', **kwargs)
        self.actual_method = actual_method

    def before_train(self, runner):
        super().before_train(runner)
        # Reuse the common training/EMA/state audit; extend its watched set
        # rather than changing the original auditor or other experiments.
        additional = {n: p for n, p in watched_parameters(self.model).items() if n not in self.watched}
        self.watched.update(additional)
        self.initial.update({n: p.detach().cpu().clone() for n, p in additional.items()})
        optimizer_ids = {id(p) for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']}
        assert {id(p) for p in self.watched.values()} <= optimizer_ids
        for name, p in additional.items():
            def capture(g, name=name):
                assert g.isfinite().all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(g.abs().max()))
            p.register_hook(capture)
        self.calls = dict(tower=0, assignment=0, roi=0)
        self.coverage, self.handles = [], []
        for key, module in [('tower', self.model.bbox_head.head_module),
                            ('assignment', self.model.bbox_head.assigner)]:
            def count(*args, key=key):
                if self.model.training:
                    self.calls[key] += 1
            self.handles.append(module.register_forward_hook(count))
        def record_roi(module, inputs):
            rows = inputs[1]
            self.calls['roi'] += 1
            self.coverage.append(int((rows[:, 4:6] > rows[:, 2:4]).all(-1).sum()))
        self.handles.append(self.model.bbox_head.instance_semantic.register_forward_pre_hook(record_roi))
        self.classifier_rows = []
        self.handles.append(self.model.bbox_head.instance_semantic.classifier.register_forward_pre_hook(
            lambda module, inputs: self.classifier_rows.append(len(inputs[0]))))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.bn_before = {n: int(m.num_batches_tracked) for n, m in self.model.named_modules()
                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        for name, module in self.model.named_modules():
            if name in self.bn_before:
                assert int(module.num_batches_tracked) == self.bn_before[name] + 1, name

    def after_train(self, runner):
        super().after_train(runner)
        for handle in self.handles:
            handle.remove()
        assert set(self.calls.values()) == {16}, self.calls
        assert self.coverage == self.classifier_rows and min(self.coverage) > 0
        report = json.loads(Path(self.report_path).read_text())
        report.update(method=self.actual_method, feature_gradient_scale=.25,
                      all_valid_gt_coverage=self.coverage, forward_counts=self.calls,
                      classifier_uses_all_gt=True, one_bn_update_per_train_step=True)
        report_to(self.report_path, report)


def smoke(cfg, output, fraction):
    require_memory(fraction)
    method, _, _, _ = identity(cfg)
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
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.instance_semantic_gradscale_probe')
    cfg.custom_hooks.append(dict(type='ROIGradientScaleAuditHook', actual_method=method,
                                report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (method + '_smoke.py')
    cfg.dump(str(snapshot))
    done = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                          cwd=ROOT, stdin=subprocess.DEVNULL)
    if done.returncode:
        raise RuntimeError(f'Official short training failed: {done.returncode}')
    assert json.loads(Path(output).read_text())['status'] == 'passed'
    assert not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)


def capacity(cfg, output, fraction, batch, path):
    method, _, _, _ = identity(cfg)
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    batch = batch or 36
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    inputs, ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    options = deepcopy(cfg.optim_wrapper); options.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, options)
    watched = watched_parameters(model)
    initial = {n: p.detach().cpu().clone() for n, p in watched.items()}
    grads = {}
    for name, p in watched.items():
        def record(g, name=name):
            assert g.isfinite().all(), name
            grads[name] = max(grads.get(name, 0), float(g.abs().max()))
        p.register_hook(record)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []
    for _ in range(3):
        values = model.train_step(inputs, optimizer)
        assert all(v.isfinite().all() for v in values.values())
        losses.append({k: float(v) for k, v in values.items()})
        torch.cuda.synchronize()
    updates = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in watched.items()}
    assert all(grads.get(n, 0) > 0 and updates[n] > 0 for n in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=method, microbatch=batch, formal_batch=36,
        accumulation=1, feature_gradient_scale=.25, optimizer_updates=3, gt_per_image=1200,
        different_image_ids=ids, repeated_annotations=True, synthetic_multiscale_boxes=True,
        formal_batch_capacity_tested=batch == 36, allocator_fraction=fraction,
        loss_scale_init=32, warmup_enabled=False, amp_skipped_updates=0, saved_checkpoints=False,
        watched_parameter_tensors=len(watched), gradient_max=grads, parameter_updates_max=updates,
        losses=losses, cuda_total_gib=total / 2**30, initial_free_gib=free / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        cudnn_benchmark=torch.backends.cudnn.benchmark, elapsed_seconds=time.perf_counter() - started,
        **config_identity(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default=METHODS[1])
    parser.add_argument('--config')
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allocator-fraction', type=float, default=.28)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    path = args.config or str(ROOT / 'configs' / args.method / 'rtmdet_tiny_dior_640.py')
    cfg = Config.fromfile(path)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, path)
        elif args.mode == 'fidelity':
            fidelity(args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch, path)
    except Exception as error:
        report_to(args.report, dict(status='failed', mode=args.mode, config=str(path),
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
