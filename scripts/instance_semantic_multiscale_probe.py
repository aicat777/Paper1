"""Isolated checks for clean multiscale object supervision; no full training."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_multiscale.data import square_object_crop
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'instance_semantic_multiscale'
SCALES = (32, 64, 128, 256, 512, 800)


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Allocator limit must leave GPU memory headroom')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    return free, total


def build(cfg, cuda=False):
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    if cuda:
        model.cuda()
    return model


def make_optimizer(cfg, model):
    options = deepcopy(cfg.optim_wrapper)
    options.loss_scale = dict(init_scale=32)
    return build_optim_wrapper(model, options)


def watched_parameters(model):
    modules = [model.backbone.stem, model.backbone.stage2, model.backbone.stage3,
               model.backbone.stage4, *model.neck.top_down_layers,
               model.bbox_head.head_module.cls_convs[0][0]]
    selected = {id(next(module.parameters())) for module in modules}
    selected.update(id(p) for p in model.bbox_head.instance_semantic.parameters())
    return {name: p for name, p in model.named_parameters() if id(p) in selected}


def bn_state(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def clean_crops(cfg, image_ids):
    """Attach ORIGINAL-image GT crops, never crops of synthetic stress boxes."""
    root = Path(cfg.data_root)
    annotation = json.loads((root / cfg.train_dataloader.dataset.ann_file).read_text())
    metadata = {row['id']: row for row in annotation['images']}
    by_image = {image: [] for image in image_ids}
    labels = {row['id']: cfg.classes.index(row['name']) for row in annotation['categories']}
    for row in annotation['annotations']:
        if row['image_id'] in by_image and not row.get('iscrowd', 0):
            by_image[row['image_id']].append(row)
    images, boxes, classes = [], [], []
    for image_id in image_ids:
        meta = metadata[image_id]
        image = cv2.imread(str(root / meta['file_name']))
        if image is None:
            raise RuntimeError('Cannot read original crop image: ' + meta['file_name'])
        row = by_image[image_id][0]
        x, y, width, height = row['bbox']
        crop, box, _ = square_object_crop(image, (x, y, x + width, y + height))
        images.append(torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))))
        boxes.append(torch.from_numpy(box))
        classes.append(labels[row['category_id']])
    return dict(images=images, boxes=torch.stack(boxes), labels=torch.tensor(classes),
                batch_indices=torch.arange(len(images)), metadata=[])


def preflight(cfg, output, source):
    old = Config.fromfile(str(ROOT / 'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
    # Compare the entire effective protocol after removing ONLY declared changes.
    for key in ('train_cfg', 'val_cfg', 'val_dataloader', 'val_evaluator', 'test_cfg',
                'test_dataloader', 'test_evaluator', 'param_scheduler', 'default_hooks',
                'randomness', 'test_pipeline'):
        assert cfg[key] == old[key], key
    for name in ('train_pipeline', 'train_pipeline_stage2'):
        pipeline = deepcopy(cfg[name])
        assert pipeline.pop(2)['type'] == 'SnapshotObjectCrop'
        pipeline[-1]['type'] = old[name][-1]['type']
        assert pipeline == old[name], name
    loader = deepcopy(cfg.train_dataloader)
    loader.dataset.pipeline = old.train_pipeline
    loader.collate_fn = old.train_dataloader.collate_fn
    assert loader == old.train_dataloader
    hooks = deepcopy(cfg.custom_hooks)
    hooks[1].switch_pipeline = old.train_pipeline_stage2
    assert hooks == old.custom_hooks
    optim = deepcopy(cfg.optim_wrapper)
    optim.type = 'AmpOptimWrapper'
    assert optim == old.optim_wrapper
    restored_model = deepcopy(cfg.model)
    for key in ('crop_scales', 'crop_scale_seed'):
        restored_model.pop(key)
    restored_model.type = old.model.type
    restored_model.data_preprocessor.type = old.model.data_preprocessor.type
    restored_model.bbox_head.type = old.model.bbox_head.type
    restored_model.bbox_head.head_module.type = old.model.bbox_head.head_module.type
    assert restored_model == old.model
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    model, previous = build(cfg), build(old)
    for name, value in previous.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    model.eval()
    macs, handles = [0], []
    def count(layer, inputs, result):
        macs[0] += result.numel() * (layer.in_channels // layer.groups) * math.prod(layer.kernel_size)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [list(x.shape) for x in cls] == [[1, 20, size, size] for size in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, size, size] for size in (80, 40, 20)]
    assert sum(x.shape[-2] * x.shape[-1] for x in cls) == 8400
    assert math.isclose(macs[0] / 1e9, 7.98770304, abs_tol=1e-10)
    assert sum(p.numel() for p in model.parameters()) == 4880648
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        data = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(data['images']) == expected
    assert tuple(cfg.model.crop_scales) == SCALES
    report_to(output, dict(status='passed', method=METHOD, config=str(Path(source).resolve()),
        config_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        previous_initial_state_exact=True, parameters=4880648, auxiliary_parameters=1940,
        inference_conv_linear_gmac=macs[0] / 1e9, prediction_points=8400,
        formal_batch=36, accumulation=1, input_size=[640, 640], scratch=True,
        train_images=5862, val_images=5863, crop_scales=list(SCALES),
        previous_auxiliary_replaced=True, training_protocol_unchanged=True,
        formal_batch_capacity_tested=False))


def equivalence(cfg, output, fraction):
    require_memory(fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    separate, joint = build(cfg, True), build(cfg, True)
    separate.train()
    joint.train()
    first, second = make_optimizer(cfg, separate), make_optimizer(cfg, joint)
    data, ids = capacity_data(cfg, 2, gt_per_image=20)
    data['data_samples']['object_crops'] = clean_crops(cfg, ids)
    for model, wrapper, split in ((separate, first, True), (joint, second, False)):
        with wrapper.optim_context(model):
            prepared = model.data_preprocessor(deepcopy(data), training=True)
            auxiliary = model.auxiliary_loss(prepared.pop('object_crops'), model.sample_crop_scale())
        if split:
            wrapper.backward_auxiliary(auxiliary)
        with wrapper.optim_context(model):
            losses = model._run_forward(prepared, mode='loss')
            main, _ = model.parse_losses(losses)
        wrapper.backward(main if split else main + auxiliary)
    maximum = 0.
    for (name, left), (other, right) in zip(separate.named_parameters(), joint.named_parameters()):
        assert name == other and left.grad is not None and right.grad is not None, name
        a, b = left.grad / 32, right.grad / 32
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-3, msg=name)
        maximum = max(maximum, float((a - b).abs().max()))
    for wrapper in (first, second):
        wrapper.step()
        wrapper.zero_grad()
        assert wrapper._inner_count == 1
        assert wrapper.loss_scaler.state_dict()['_growth_tracker'] == 1
        assert wrapper.loss_scaler.get_scale() == 32
    for name, value in separate.state_dict().items():
        torch.testing.assert_close(value, joint.state_dict()[name], atol=2e-6, rtol=2e-5, msg=name)
    assert all(int(value) == 1 for name, value in bn_state(separate).items() if name.endswith('num_batches_tracked'))
    report_to(output, dict(status='passed', method=METHOD, amp=True, actual_train_images=ids,
        gradient_max_error=maximum, separate_and_summed_losses_match=True,
        optimizer_steps=1, scaler_updates=1, mmengine_iterations=1, main_only_bn_updates=True))


@HOOKS.register_module()
class ObjectCropAuditHook(CandidateAuditHook):
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
        self.scales, self.crop_counts = [], []
        original = self.model.auxiliary_loss
        def observed(model, crops, scale):
            before = bn_state(model)
            result = original(crops, scale)
            for name, value in bn_state(model).items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0, msg=name)
            assert len(crops['images']) == len(crops['batch_indices'].unique())
            self.scales.append(scale)
            self.crop_counts.append(len(crops['images']))
            return result
        self.model.auxiliary_loss = MethodType(observed, self.model)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def after_train(self, runner):
        self.restore_check()
        updates = {name: float((p.detach().cpu() - self.initial[name]).abs().max())
                   for name, p in self.watched.items()}
        assert all(self.gradients.get(name, 0) > 0 and updates[name] > 0 for name in self.watched)
        wrapper = runner.optim_wrapper
        assert wrapper._inner_count == 24
        assert all(int(wrapper.optimizer.state[p]['step']) == 24 for p in self.watched.values())
        assert wrapper.loss_scaler.get_scale() == 32
        assert wrapper.loss_scaler.state_dict()['_growth_tracker'] == 24
        assert len(self.losses) == 24 and len(self.metrics) == 2
        assert self.predictions == 64 and self.raw_predictions == 8 and self.ema_roundtrips == 2
        assert set(self.scales) == set(SCALES) and len(self.scales) == 24
        assert all(count == 4 for count in self.crop_counts)
        assert all(int(value) == 24 for name, value in bn_state(self.model).items()
                   if name.endswith('num_batches_tracked'))
        ema = next(hook for hook in runner.hooks if type(hook).__name__ == 'EMAHook')
        assert int(ema.ema_model.steps) == 24
        checkpoint = dict(state_dict=deepcopy(self.model.state_dict()), optimizer=wrapper.state_dict())
        ema.before_save_checkpoint(runner, checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        restored = MODELS.build(deepcopy(runner.cfg.model))
        resume_cfg = deepcopy(runner.cfg.custom_hooks[0])
        resume_cfg.pop('priority')
        resume_ema = HOOKS.build(resume_cfg)
        stub = SimpleNamespace(model=restored, _resume=True)
        resume_ema.before_run(stub)
        resume_ema.after_load_checkpoint(stub, saved)
        restored.load_state_dict(saved['state_dict'], strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name].cpu(), atol=0, rtol=0, msg=name)
        for name, value in resume_ema.ema_model.state_dict().items():
            torch.testing.assert_close(value, ema.ema_model.state_dict()[name].cpu(), atol=0, rtol=0, msg=name)
        wrapper.load_state_dict(saved['optimizer'])
        state = self.model.crop_scale_rng_state.clone()
        expected = [self.model.sample_crop_scale() for _ in range(12)]
        self.model.crop_scale_rng_state.copy_(state)
        assert expected == [restored.sample_crop_scale() for _ in range(12)]
        assert wrapper.loss_scaler.state_dict()['_growth_tracker'] == 24
        report_to(self.report_path, dict(status='passed', method=METHOD,
            official_train_entrypoint=True, input_size=[640, 640], microbatch=4,
            optimizer_updates=24, mmengine_iterations=24, scaler_updates=24,
            accumulation=1, crop_scales=self.scales, crop_counts=self.crop_counts,
            crop_bn_buffers_unchanged=True, full_state_ema_roundtrips=2,
            model_optimizer_scaler_serialization=True, crop_rng_restored=True,
            ema_updates=24, actual_ema_checkpoint_swap_and_resume=True,
            strong_and_weak_augmentation=True, val_passes=2, val_images_per_pass=32,
            predictions_checked=64, raw_predictions_checked=8, finite_losses=True,
            finite_predictions=True, amp_skipped_updates=0, loss_scale_init=32,
            warmup_enabled=False, saved_checkpoints=False, losses=self.losses,
            metrics=self.metrics, gradient_max=self.gradients, parameter_updates_max=updates,
            peak_allocated_gib=self.peak_allocated / 2**30,
            peak_reserved_gib=self.peak_reserved / 2**30,
            elapsed_seconds=time.perf_counter() - self.started))


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
    # At scale32 the deepest map is 1x1. Batch2 produces poorly conditioned
    # crop BN derivatives in AMP; use four real images, preserving formal36.
    cfg.train_dataloader.update(batch_size=4, num_workers=0, persistent_workers=False)
    cfg.train_dataloader.dataset.indices = list(range(48))
    cfg.val_dataloader.update(batch_size=4, num_workers=0, persistent_workers=False)
    annotation = json.loads((Path(cfg.data_root) / cfg.val_dataloader.dataset.ann_file).read_text())
    remaining = {row['id'] for row in annotation['categories']}
    by_image = {row['id']: set() for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].add(row['category_id'])
    selected = []
    while remaining:
        image = max(by_image, key=lambda key: len(by_image[key] & remaining))
        assert by_image[image] & remaining
        selected.append(image)
        remaining -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    assert len(selected) == 32
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.instance_semantic_multiscale_probe')
    cfg.custom_hooks.append(dict(type='ObjectCropAuditHook', method=METHOD,
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


def capacity(cfg, output, fraction, batch, auxiliary_only=False):
    free, total = require_memory(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    assert cfg.train_dataloader.batch_size == 36
    data, ids = capacity_data(cfg, batch)
    data['data_samples']['object_crops'] = clean_crops(cfg, ids)
    model = build(cfg, True)
    model.train()
    wrapper = make_optimizer(cfg, model)
    initial = model.bbox_head.instance_semantic.classifier.weight.detach().clone()
    # Force the largest scale only in this process, not the formal sampler.
    model.sample_crop_scale = lambda: 800
    losses, started = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    test_scales = SCALES + (800, 800) if auxiliary_only else (800,) * 3
    crop_buffers = bn_state(model)
    for scale in test_scales:
        if auxiliary_only:
            crops = model.data_preprocessor.cast_data(data['data_samples']['object_crops'])
            with wrapper.optim_context(model):
                loss = model.auxiliary_loss(crops, scale)
            wrapper.update_params(loss)
            result = dict(loss_instance_semantic=loss.detach())
            for name, value in bn_state(model).items():
                torch.testing.assert_close(value, crop_buffers[name], atol=0, rtol=0, msg=name)
        else:
            result = model.train_step(data, wrapper)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    count = len(test_scales)
    assert wrapper._inner_count == count and wrapper.loss_scaler.get_scale() == 32
    assert wrapper.loss_scaler.state_dict()['_growth_tracker'] == count
    classifier = model.bbox_head.instance_semantic.classifier.weight
    assert not torch.equal(classifier, initial)
    assert int(wrapper.optimizer.state[classifier]['step']) == count
    report_to(output, dict(status='passed', method=METHOD, microbatch=batch,
        formal_batch=36, accumulation=1, crop_resolutions=list(test_scales), optimizer_updates=count,
        auxiliary_only=auxiliary_only, formal_batch_capacity_tested=batch == 36 and not auxiliary_only,
        gt_per_image=0 if auxiliary_only else 1200, different_image_ids=ids,
        repeated_detection_annotations=not auxiliary_only, auxiliary_crops_from_original_real_gt=True,
        allocator_fraction=fraction, loss_scale_init=32, amp_skipped_updates=0,
        saved_checkpoints=False, losses=losses, cuda_device=torch.cuda.get_device_name(),
        cuda_total_gib=total / 2**30, initial_free_gib=free / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        elapsed_seconds=time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'equivalence', 'smoke', 'capacity', 'crop_capacity'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int, default=12)
    parser.add_argument('--allocator-fraction', type=float, default=.4)
    args = parser.parse_args()
    register_all_modules()
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'equivalence':
            equivalence(cfg, args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.mode == 'crop_capacity')
    except Exception as error:
        report_to(args.report, dict(status='failed', method=METHOD, mode=args.mode,
            error_type=type(error).__name__, error=str(error)))
        raise
    print(args.mode.upper() + '_PASSED', flush=True)


if __name__ == '__main__':
    main()
