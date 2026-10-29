"""Isolated checks for selective ROI-to-point transfer; formal training stays scratch."""
import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_transfer import RegionToPointTransfer
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.roi_semantic_ablation_probe import config_identity, require_memory, watched_parameters

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'instance_semantic_transfer'


def preflight(cfg, output, path):
    original = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
    for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks', 'train_pipeline',
                'train_pipeline_stage2', 'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == original[key], key
    expected = deepcopy(original.model)
    expected.bbox_head.update(type='TransferInstanceSemanticRTMDetHead', transfer_loss_weight=.1,
                              transfer_min_iou=.5, transfer_temperature=1.)
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
    assert set(model.state_dict()) == set(reference.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    assert sum(p.numel() for p in model.parameters()) == 4880648
    assert len(watched_parameters(model)) == 22
    assert not list(model.bbox_head.local_transfer.parameters())
    macs, handles = [0], []

    def count(layer, inputs, result):
        per_output = (layer.in_channels // layer.groups) * math.prod(layer.kernel_size) \
            if isinstance(layer, torch.nn.Conv2d) else layer.in_features
        macs[0] += result.numel() * per_output

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    calls = []
    handles.append(model.bbox_head.local_transfer.register_forward_hook(lambda *a: calls.append(1)))
    with torch.no_grad():
        actual = model._forward(torch.zeros(1, 3, 640, 640))
        wanted = reference._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert calls == []
    for first, second in zip(actual, wanted):
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert math.isclose(macs[0] / 1e9, 7.98770304, abs_tol=1e-10)
    for loader, expected_count in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected_count
    report_to(output, dict(status='passed', method=METHOD, parameters=4880648,
        inference_parameters=4878708, new_parameters=0, inference_conv_linear_gmac=macs[0] / 1e9,
        initial_parameters_and_predictions_exact=True, inference_skips_transfer=True,
        unchanged_training_protocol=True, input_size=[640, 640], formal_batch=36, accumulation=1,
        train_images=5862, val_images=5863, scratch=True, semantic_weight=.1,
        transfer_weight=.1, transfer_min_iou=.5, transfer_temperature=1.,
        transfer_normalization='mean_within_each_selected_gt_then_mean_over_selected_gt',
        original_roi_coverage='all_valid_gt', original_roi_denominator='all_valid_gt',
        formal_batch_capacity_tested=False, **config_identity(path)))


def fidelity(output, fraction):
    require_memory(fraction)
    helpers = runpy.run_path(str(ROOT / 'tests/test_instance_semantic_transfer.py'))
    cases = []
    for amp in (False, True):
        for temperature in (1., 2.):
            values = helpers['fixture']('cuda', torch.float16 if amp else torch.float32)
            module = RegionToPointTransfer(temperature=temperature).cuda()
            with torch.autocast('cuda', enabled=amp):
                actual, stats = module(*values)
            expected, _, _ = helpers['reference_points'](*values, temperature=temperature)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
            params = (*values[0], *values[1], values[5], *values[6].parameters())
            first = torch.autograd.grad(actual, params, retain_graph=True, allow_unused=True)
            second = torch.autograd.grad(expected, params, allow_unused=True)
            errors = []
            for a, b in zip(first, second):
                if b is None:
                    assert a is None
                else:
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-3)
                    assert torch.isfinite(a).all()
                    errors.append(float((a.float() - b.float()).abs().max()))
            assert all(g is None for g in first[3:])
            cases.append(dict(amp=amp, temperature=temperature, selected_points=7, selected_gt=3,
                forward_max_error=float((actual - expected).abs()), gradient_max_error=max(errors),
                classifier_teacher_and_detector_score_gradients_absent=True))
    report_to(output, dict(status='passed', method=METHOD, cases=cases))


@HOOKS.register_module()
class TransferSemanticAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {n: p.detach().cpu().clone() for n, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, p in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            p.register_hook(capture)
        self.calls = dict(head=0, assignment=0, roi=0, transfer=0)

        def counter(key):
            def hook(*args):
                if self.model.training:
                    self.calls[key] += 1
            return hook

        head = self.model.bbox_head
        self.handles = [head.head_module.register_forward_hook(counter('head')),
                        head.assigner.register_forward_hook(counter('assignment')),
                        head.instance_semantic.classifier.register_forward_hook(counter('roi')),
                        head.local_transfer.register_forward_hook(counter('transfer'))]
        self.started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.bn_before = {n: int(m.num_batches_tracked) for n, m in self.model.named_modules()
                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                assert int(layer.num_batches_tracked) == self.bn_before[name] + 1, name

    def after_train(self, runner):
        super().after_train(runner)
        for handle in self.handles:
            handle.remove()
        assert self.calls == dict(head=16, assignment=16, roi=16, transfer=16)
        report = json.loads(Path(self.report_path).read_text())
        report.update(single_pass_calls=self.calls, one_bn_update_per_train_step=True,
            all_valid_gt_roi_preserved=True, formal_min_iou_used=True, scratch=True,
            transfer_selected_points=sum(x['transfer_selected_points'] for x in self.losses),
            new_parameters=0)
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
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.instance_semantic_transfer_probe')
    cfg.custom_hooks.append(dict(type='TransferSemanticAuditHook', method=METHOD,
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


def active_real(cfg, output, fraction, checkpoint):
    """Exercise the actual .5 selection on an isolated converged ROI model.

    Own completed ROI weights are used ONLY here to obtain reliable boxes now.
    This does not alter load_from/resume in the source/frozen formal config.
    """
    require_memory(fraction)
    assert cfg.load_from is None and not cfg.resume
    random.seed(431); np.random.seed(431); torch.manual_seed(431)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k, v in saved['ema_state_dict'].items()
             if k.startswith('module.')}
    assert state and set(state) == set(saved['state_dict'])
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.load_state_dict(state, strict=True); model.cuda().train()
    del saved, state
    optim_cfg = deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    loader_cfg = deepcopy(cfg.train_dataloader)
    loader_cfg.update(batch_size=8, num_workers=0, persistent_workers=False,
                      sampler=dict(type='DefaultSampler', shuffle=False))
    loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 32, replace=False).tolist()
    loader = Runner.build_dataloader(loader_cfg, seed=431)
    losses, routes = [], []
    params = dict(model.named_parameters())
    selected_total = 0
    for index, raw in enumerate(loader):
        processed = model.data_preprocessor(raw, training=True)
        if index == 0:
            continue
        with optimizer.optim_context(model):
            values = model.loss(processed['inputs'], processed['data_samples'])
            total = sum(value for key, value in values.items() if 'loss' in key)
            assert all(torch.isfinite(value).all() for value in values.values())
            count = int(values['transfer_selected_points'])
            selected_total += count
            if count:
                grads = torch.autograd.grad(values['loss_instance_transfer'], tuple(params.values()),
                                            retain_graph=True, allow_unused=True)
                active = []
                for (name, p), gradient in zip(params.items(), grads):
                    if name.startswith(('bbox_head.instance_semantic.', 'bbox_head.head_module.rtm_cls.',
                                        'bbox_head.head_module.rtm_reg.', 'bbox_head.head_module.reg_convs.')):
                        assert gradient is None, name
                    if gradient is not None:
                        assert torch.isfinite(gradient).all(), name
                        if gradient.abs().sum() > 0:
                            active.append(name)
                assert any(n.startswith('bbox_head.head_module.cls_convs.') for n in active)
                assert any(n.startswith('backbone.') for n in active)
                assert any(n.startswith('neck.') for n in active)
                routes.append(dict(step=index, selected_points=count, active_feature_parameters=len(active),
                                   auxiliary_classifier_and_prediction_layer_gradients_absent=True))
        optimizer.update_params(total)
        losses.append({key: float(value) for key, value in values.items()})
    assert len(losses) == 3 and selected_total > 0 and routes
    assert optimizer.loss_scaler.get_scale() == 32
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched_parameters(model).values())
    report_to(output, dict(status='passed', method=METHOD, microbatch=8, optimizer_updates=3,
        transfer_min_iou=.5, transfer_selected_points=selected_total, gradient_routes=routes,
        loss_scale_init=32, amp_skipped_updates=0, saved_checkpoints=False,
        isolated_check_only=True, own_roi_weights_for_activation_only=str(Path(checkpoint).resolve()),
        formal_config_still_scratch=True, losses=losses,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))


def stress_selection(module, inputs):
    """Capacity-only: activate every point without changing actual DSL losses."""
    features, scores, assigned, rows, lookup, teacher, classifier = inputs
    batch_size, total_points = assigned['assigned_gt_indices'].shape
    slots = torch.empty_like(assigned['assigned_gt_indices'])
    for image in range(batch_size):
        valid_slots = (lookup[image] >= 0).nonzero().flatten()
        assert len(valid_slots)
        slots[image] = valid_slots[torch.arange(total_points, device=slots.device) % len(valid_slots)]
    fake_assigned = dict(assigned, assigned_gt_indices=slots,
                         assign_metrics=torch.ones_like(assigned['assign_metrics']))
    teacher = F.one_hot(rows[:, 1].long(), classifier.out_features).to(teacher) * 8 - 4
    fake_scores, offset = [], 0
    for feature, score in zip(features, scores):
        height, width = feature.shape[-2:]
        size = height * width
        images = torch.arange(batch_size, device=slots.device)[:, None]
        row = lookup[images, slots[:, offset:offset + size]]
        wrong_label = (rows[row, 1].long() + 1) % classifier.out_features
        fake = F.one_hot(wrong_label.reshape(batch_size, height, width), classifier.out_features)
        fake_scores.append(fake.permute(0, 3, 1, 2).to(score) * 8 - 4)
        offset += size
    return features, tuple(fake_scores), fake_assigned, rows, lookup, teacher, classifier


def capacity(cfg, output, fraction, batch, path):
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or 36
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    model.bbox_head.local_transfer.register_forward_pre_hook(stress_selection)
    optim_cfg = deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model)
    initial = {n: p.detach().cpu().clone() for n, p in watched.items()}
    gradients = {}
    for name, p in watched.items():
        def capture(gradient, name=name):
            assert torch.isfinite(gradient).all(), name
            gradients[name] = max(gradients.get(name, 0), float(gradient.abs().max()))
        p.register_hook(capture)
    started, losses = time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        values = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in values.values())
        assert int(values['transfer_selected_points']) == batch * 8400
        assert int(values['transfer_selected_gt']) == batch * 1200
        losses.append({key: float(value) for key, value in values.items()})
        torch.cuda.synchronize()
    updates = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in watched.items()}
    assert all(gradients.get(n, 0) > 0 and updates[n] > 0 for n in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=METHOD, microbatch=batch, accumulation=1,
        optimizer_updates=3, gt_per_image=1200, different_image_ids=image_ids,
        repeated_annotations=True, synthetic_multiscale_boxes=True, formal_batch=36,
        formal_batch_capacity_tested=batch == 36, allocator_fraction=fraction,
        synthetic_full_transfer_selection=True, active_transfer_points_per_step=batch * 8400,
        actual_dsl_and_main_losses_unchanged=True, capacity_hook_not_in_formal_config=True,
        loss_scale_init=32, warmup_enabled=False, amp_skipped_updates=0, saved_checkpoints=False,
        watched_parameter_tensors=len(watched), gradient_max=gradients, parameter_updates_max=updates,
        losses=losses, cuda_device=torch.cuda.get_device_name(), cuda_total_gib=total / 2**30,
        initial_free_gib=free / 2**30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        elapsed_seconds=time.perf_counter() - started, **config_identity(path)))
    print('CAPACITY_PASSED', batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'active', 'capacity'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.30)
    parser.add_argument('--checkpoint', default=str(ROOT / 'work_dirs/instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36/epoch_300.pth'))
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
        elif args.mode == 'active':
            active_real(cfg, args.report, args.allocator_fraction, args.checkpoint)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.config)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=METHOD, mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
