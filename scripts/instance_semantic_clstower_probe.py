"""Isolated checks for classification-only ROI, retaining the queued models."""
import argparse
from copy import deepcopy
import gc
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

from custom_models.instance_semantic_clstower import ClassificationOnlyROISemanticDetector
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import report_to
from scripts.instance_semantic_neckhead_probe import ROINeckHeadAuditHook, watched_parameters, build
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown_p3_g4_semantic_clstower'
REFERENCE = 'scdown_p3_g4_semantic_gradscale'


def preflight(cfg, output, path):
    reference_cfg = Config.fromfile(str(ROOT / 'configs' / REFERENCE / 'rtmdet_tiny_dior_640.py'))
    for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                'train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'optim_wrapper',
                'param_scheduler', 'default_hooks', 'randomness', 'env_cfg'):
        assert cfg[key] == reference_cfg[key], key
    expected = deepcopy(reference_cfg.model); expected.type = 'ClassificationOnlyROISemanticDetector'
    assert cfg.model == expected
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.img_scale == (640, 640) and cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    reference, model = build(reference_cfg).eval(), build(cfg).eval()
    assert isinstance(model, ClassificationOnlyROISemanticDetector)
    assert set(reference.state_dict()) == set(model.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0, msg=name)
    selected = {id(p) for p in model.roi_gradient_parameters()}
    assert not selected & {id(p) for p in model.backbone.parameters()}
    assert not selected & {id(p) for p in model.neck.parameters()}
    assert {id(p) for p in model.bbox_head.head_module.cls_convs.parameters()} <= selected
    assert len(selected) == 16
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
    assert sum(p.numel() for p in model.parameters()) == 3105704
    assert math.isclose(macs[0] / 1e9, 6.78524544, abs_tol=1e-10)
    for loader, count_images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == count_images
    report_to(output, dict(status='passed', method=METHOD, reference=REFERENCE,
        parameters=3105704, inference_parameters=3103764, new_parameters=0,
        inference_conv_linear_gmac=6.78524544, initialization_and_state_match=True,
        inference_matches_reference_exactly=True, original_roi_preserved=True,
        backbone_roi_gradient_scale=0., neck_roi_gradient_scale=0., classification_roi_gradient_scale=.25,
        auxiliary_classifier_weight=.1, all_gt_and_original_normalization_preserved=True,
        main_losses_and_matching_unchanged=True, formal_batch=36, accumulation=1,
        input_size=[640, 640], train_images=5862, val_images=5863, scratch=True,
        training_protocol_unchanged=True, watched_parameter_tensors=len(watched_parameters(model)),
        auxiliary_parameter_tensors=len(model.roi_gradient_parameters()),
        formal_batch_capacity_tested=False, **config_identity(path)))


def fidelity(cfg, output, fraction):
    require_memory(fraction)
    reference_cfg = Config.fromfile(str(ROOT / 'configs' / REFERENCE / 'rtmdet_tiny_dior_640.py'))
    records = []
    for amp, scale in ((False, 1.), (False, 32.), (True, 32.), (True, 128.)):
        reference, model = build(reference_cfg, 'cuda').train(), build(cfg, 'cuda').train()
        torch.manual_seed(17)
        images = torch.randn(2, 3, 256, 256, device='cuda')
        rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
            [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
            [0, 3, 8, 8, 248, 248], [1, 4, 4.25, 5.5, 5.25, 6.5]], device='cuda', dtype=torch.float32)
        samples = dict(bboxes_labels=rows, img_metas=[dict(img_shape=(256, 256))] * 2)
        with torch.autocast('cuda', enabled=amp):
            expected = reference.loss(images, samples)
        parameters = tuple(reference.parameters())
        gm = torch.autograd.grad(scale * (expected['loss_cls'] + expected['loss_bbox']),
                                 parameters, allow_unused=True, retain_graph=True)
        ga = torch.autograd.grad(scale * expected['loss_instance_semantic'],
                                 parameters, allow_unused=True)
        with torch.autocast('cuda', enabled=amp):
            actual = model.loss(images, samples)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        (scale * sum(actual.values())).backward()
        max_error, square_error, square_reference = 0., 0., 0.
        upstream_norms = dict(backbone=0., neck=0.)
        for (name, p), main, aux in zip(model.named_parameters(), gm, ga):
            wanted = main if name.startswith(('backbone.', 'neck.')) or aux is None else (
                aux if main is None else main + aux)
            if wanted is None:
                assert p.grad is None or not p.grad.any(), name
                continue
            assert p.grad is not None and p.grad.isfinite().all(), name
            error = (p.grad.float() - wanted.float()) / scale
            max_error = max(max_error, float(error.abs().max()))
            square_error += float(error.double().square().sum())
            square_reference += float((wanted.double() / scale).square().sum())
            torch.testing.assert_close(p.grad / scale, wanted / scale, atol=5e-6, rtol=3e-4, msg=name)
            for key in upstream_norms:
                if name.startswith(key + '.'):
                    upstream_norms[key] += float(p.grad.float().norm() / scale)
        assert all(value > 0 for value in upstream_norms.values())
        # Inspect the pure auxiliary route separately: exactly no backbone/neck
        # contribution even though all backbone parameters remain trainable.
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=amp):
            auxiliary = model.loss(images, samples)['loss_instance_semantic']
        (auxiliary * scale).backward()
        assert all(p.grad is None or not p.grad.any() for p in (*model.backbone.parameters(), *model.neck.parameters()))
        record = dict(amp=amp, outer_loss_multiplier=scale, forward_error=0.,
            maximum_unscaled_parameter_gradient_error=max_error,
            gradient_relative_l2_error=math.sqrt(square_error / max(square_reference, 1e-30)),
            main_backbone_gradient_nonzero=True, main_neck_gradient_nonzero=True,
            roi_backbone_gradient_exactly_zero=True, roi_neck_gradient_exactly_zero=True)
        records.append(record); print(json.dumps(record), flush=True)
        del reference, model, parameters, expected, actual, gm, ga, images, samples, rows, auxiliary
        gc.collect(); torch.cuda.empty_cache()
    report_to(output, dict(status='passed', method=METHOD,
        independent_reference='Separate main/auxiliary autograd on the unchanged alpha=.25 GSCE; discard auxiliary backbone and neck derivatives',
        cases=records, no_parameter_updates=True, saved_checkpoints=False))


@HOOKS.register_module()
class ROIClassificationTowerAuditHook(ROINeckHeadAuditHook):
    def after_train(self, runner):
        super().after_train(runner)
        report = json.loads(Path(self.report_path).read_text())
        report.update(neck_roi_gradient_scale=0., classification_roi_gradient_scale=.25,
                      auxiliary_parameter_tensors=len(self.model.roi_gradient_parameters()))
        report_to(self.report_path, report)


def smoke(cfg, output, fraction):
    require_memory(fraction)
    method = METHOD
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
    cfg.custom_imports.imports.append('scripts.instance_semantic_clstower_probe')
    cfg.custom_hooks.append(dict(type='ROIClassificationTowerAuditHook', actual_method=method,
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
    method = METHOD
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
        accumulation=1, feature_gradient_scale=.25, backbone_roi_gradient_scale=0., neck_roi_gradient_scale=0., optimizer_updates=3, gt_per_image=1200,
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
    parser.add_argument('--config', default=str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allocator-fraction', type=float, default=.28)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            fidelity(cfg, args.report, args.allocator_fraction)
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
