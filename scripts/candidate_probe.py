"""Read-only audits and isolated GPU short runs for four standalone candidates.

Uses the official tools/train.py entrypoint. Does not start full training,
modify queues, or retain smoke checkpoints. Run with CUDA_VISIBLE_DEVICES set
to a checked device; a shared-GPU allocator limit is applied before training.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import io
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
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('freq_fusion', 'instance_semantic', 'tiny_assignment', 'factorized_head')


def report_to(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def watched_parameters(model, method):
    prefix = dict(freq_fusion='neck.freq_fusions.',
                  instance_semantic='bbox_head.instance_semantic.',
                  tiny_assignment='bbox_head.head_module.',
                  factorized_head='bbox_head.head_module.')[method]
    return {name: value for name, value in model.named_parameters() if name.startswith(prefix)}


def preflight(cfg, method, output):
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert math.isclose(cfg.optim_wrapper.optimizer.lr, .0005625, rel_tol=1e-12)
    assert cfg.custom_hooks[0].get('gamma', 2000) == 2000
    assert cfg.custom_hooks[0].get('interval', 1) == 1
    assert cfg.param_scheduler[0].end == 1000
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    baseline = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
    for name in ('train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'val_dataloader',
                 'val_evaluator', 'custom_hooks', 'randomness', 'default_hooks'):
        assert cfg[name] == baseline[name], name
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.eval()
    cost, handles = [0], []
    def count(module, inputs, result):
        if isinstance(module, torch.nn.Conv2d):
            cost[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        else:
            cost[0] += result.numel() * module.in_features
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles: handle.remove()
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls], with_stride=True)
    assert sum(len(x) for x in priors) == 8400
    for loader, count_images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == count_images
    report_to(output, dict(status='passed', method=method, parameters=sum(p.numel() for p in model.parameters()),
                           inference_conv_linear_gmac=cost[0] / 1e9, prediction_points=8400,
                           excludes_dynamic_filtering_resampling_norm_and_postprocessing=True,
                           watched_parameter_tensors=len(watched_parameters(model, method)),
                           scratch=True, input_size=[640, 640], batch=36, accumulation=1,
                           formal_batch_capacity_tested=False, ema_gamma=2000,
                           class_count=20, train_images=5862, val_images=5863))


def freq_fidelity(reference, output, fraction):
    from custom_models.freq_fusion.operator import FreqFusion, reassemble
    path = Path(reference)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_hash == '886e2d0068d2c1ce8f054ad12b4337e990a43a64220b29c978c4eec8057c12d7'
    spec = importlib.util.spec_from_file_location('paper1_freqfusion_reference', str(path))
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.cuda.set_per_process_memory_fraction(fraction)
    cases = []
    for width, compressed, h, w in ((192, 48, 20, 20), (96, 24, 40, 40), (96, 24, 7, 11)):
        for half in (False, True):
            torch.manual_seed(19)
            ours = FreqFusion(width, width, compressed).cuda()
            # Nonuniform learned filters/offsets expose ordering mistakes which
            # near-uniform initialization could hide.
            for layer in (ours.content_encoder, ours.content_encoder2):
                torch.nn.init.normal_(layer.weight, std=.06)
            other = upstream.FreqFusion(width, width, compressed_channels=compressed,
                                         feature_resample=True, feature_resample_group=4).cuda()
            other.load_state_dict(ours.state_dict(), strict=True)
            dtype = torch.float16 if half else torch.float32
            x = torch.randn(2, width, 2 * h, 2 * w, device='cuda', dtype=dtype, requires_grad=True)
            y = torch.randn(2, width, h, w, device='cuda', dtype=dtype, requires_grad=True)
            rx, ry = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
            with torch.autocast('cuda', enabled=half):
                actual = ours(x, y)
            expected = other(rx.float(), ry.float())
            expected = (expected[0], expected[1].to(dtype), expected[2].to(dtype))
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-4)
            seeds = [torch.randn_like(value) / math.sqrt(value.numel()) for value in actual]
            names, parameters = zip(*ours.named_parameters())
            reference_parameters = dict(other.named_parameters())
            grads = torch.autograd.grad(actual, (x, y, *parameters), seeds)
            ref_grads = torch.autograd.grad(expected, (rx, ry, *(reference_parameters[n] for n in names)), seeds)
            for name, a, e in zip(('hr_input', 'lr_input', *names), grads, ref_grads):
                torch.testing.assert_close(a, e, atol=4e-5, rtol=4e-4, msg=name)
            cases.append(dict(width=width, guidance_width=compressed, lr_size=[h, w], amp=half,
                              forward_max_error=max(float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)),
                              gradient_max_error=max(float((a.float() - b.float()).abs().max()) for a, b in zip(grads, ref_grads))))
    carafe_cases = []
    for kernel in (3, 5):
        for scale in (1, 2):
            x = torch.randn(2, 8, 3, 5, requires_grad=True)
            masks = torch.randn(2, kernel * kernel, 3 * scale, 5 * scale).softmax(1).requires_grad_()
            gx, gm = x.detach().cuda().requires_grad_(), masks.detach().cuda().requires_grad_()
            expected, actual = reassemble(x, masks, kernel, scale), reassemble(gx, gm, kernel, scale)
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
            seed = torch.randn_like(expected)
            a = torch.autograd.grad(actual, (gx, gm), seed.cuda())
            e = torch.autograd.grad(expected, (x, masks), seed)
            for first, second in zip(a, e):
                torch.testing.assert_close(first.cpu(), second, atol=3e-6, rtol=1e-5)
            carafe_cases.append(dict(kernel=kernel, scale=scale, passed=True))
    report_to(output, dict(status='passed', reference=str(path), source_sha256=source_hash,
                           upstream_commit='3fb0c70637a3c194fb74294d3ce4681958b26241',
                           complete_generators_and_enhanced_initial_fusion=True,
                           cases=cases, cpu_cuda_carafe=carafe_cases))


@HOOKS.register_module()
class CandidateAuditHook(Hook):
    priority = 48

    def __init__(self, method, report_path, image_ids, allocator_fraction=.35):
        self.method, self.report_path, self.image_ids = method, report_path, image_ids
        self.fraction = allocator_fraction
        self.losses, self.metrics, self.gradients, self.raw_state = [], [], {}, None
        self.predictions = self.raw_predictions = self.ema_roundtrips = 0
        self.peak_allocated = self.peak_reserved = 0
        self.assignment = dict(fallback_positives=0, floored_regression_positives=0, unmatched_tiny=0)

    def before_run(self, runner):
        torch.cuda.set_per_process_memory_fraction(self.fraction)

    def capture_memory(self):
        # MMEngine's logger resets peak statistics after logging each interval.
        # Capture before the lower-priority logger, not just at after_train.
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model, self.method)
        self.initial = {n: p.detach().cpu().clone() for n, p in self.watched.items()}
        params = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(params) == len({id(p) for p in params})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in params}
        for name, value in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            value.register_hook(capture)
        if self.method == 'tiny_assignment':
            def capture_assignment(module, inputs, result):
                self.assignment['fallback_positives'] += int(result['fallback_mask'].sum())
                self.assignment['floored_regression_positives'] += int((result['regression_weights'] > result['assign_metrics']).sum())
                self.assignment['unmatched_tiny'] += int(result['unmatched_tiny'].sum())
            self.model.bbox_head.assigner.register_forward_hook(capture_assignment)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def restore_check(self):
        if self.raw_state is not None:
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(value.detach().cpu(), self.raw_state[name], atol=0, rtol=0, msg=name)
            self.raw_state = None
            self.ema_roundtrips += 1

    def before_train_epoch(self, runner):
        self.restore_check()

    @staticmethod
    def finite_predictions(samples):
        for sample in samples:
            assert torch.isfinite(sample.pred_instances.bboxes).all()
            assert torch.isfinite(sample.pred_instances.scores).all()

    def before_val_epoch(self, runner):
        self.raw_state = {n: p.detach().cpu().clone() for n, p in self.model.state_dict().items()}
        for metric in runner.val_evaluator.metrics:
            if hasattr(metric, 'img_ids'):
                metric.img_ids = self.image_ids
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            samples = self.model.test_step(next(iter(runner.val_dataloader)))
        self.finite_predictions(samples)
        self.capture_memory()
        self.raw_predictions += len(samples)
        self.model.train(training)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        values = {k: float(v) for k, v in outputs.items()}
        assert all(math.isfinite(v) for v in values.values()), values
        self.losses.append(values)
        self.capture_memory()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.finite_predictions(outputs)
        self.predictions += len(outputs)
        self.capture_memory()

    def after_val_epoch(self, runner, metrics=None):
        values = {k: metrics[k] for k in ('pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50')}
        assert all(math.isfinite(v) for v in values.values()), values
        self.metrics.append(values)

    def after_train(self, runner):
        self.restore_check()
        updates = {n: float((p.detach().cpu() - self.initial[n]).abs().max()) for n, p in self.watched.items()}
        inactive = [n for n in self.watched if not self.gradients.get(n, 0) > 0 or not updates[n] > 0]
        assert not inactive, inactive
        assert all(int(runner.optim_wrapper.optimizer.state[p]['step']) == 16 for p in self.watched.values())
        assert len(self.losses) == 16 and len(self.metrics) == 2 and self.predictions == 64
        assert self.ema_roundtrips == 2 and self.raw_predictions == 8
        assert runner.optim_wrapper.loss_scaler.get_scale() == 32
        # Serialize/restore the trained model, optimizer AND AMP scaler in
        # memory; verify state including new buffers without retaining a .pth.
        buffer = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(), optimizer=runner.optim_wrapper.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location='cpu', weights_only=False)
        restored = MODELS.build(deepcopy(runner.cfg.model))
        restored.load_state_dict(saved['model'], strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name].detach().cpu(), atol=0, rtol=0)
        runner.optim_wrapper.load_state_dict(saved['optimizer'])
        assert all(int(runner.optim_wrapper.optimizer.state[p]['step']) == 16 for p in self.watched.values())
        if self.method == 'tiny_assignment':
            assert self.assignment['floored_regression_positives'] > 0
        report_to(self.report_path, dict(status='passed', method=self.method, official_train_entrypoint=True,
                  input_size=[640, 640], microbatch=2, optimizer_updates=16, accumulation=1,
                  val_passes=2, val_images_per_pass=32, predictions_checked=64,
                  raw_predictions_checked=8, full_state_ema_roundtrips=2,
                  model_optimizer_scaler_serialization=True,
                  strong_and_weak_augmentation=True, coco_evaluation_scoped_to_subset=True,
                  watched_parameter_tensors=len(self.watched), finite_losses=True, finite_predictions=True,
                  amp_skipped_updates=0, gradient_max=self.gradients, parameter_updates_max=updates,
                  losses=self.losses, metrics=self.metrics, assignment=self.assignment,
                  loss_scale_init=32, warmup_enabled=False, saved_checkpoints=False,
                  peak_allocated_gib=self.peak_allocated / 2**30,
                  peak_reserved_gib=self.peak_reserved / 2**30,
                  elapsed_seconds=time.perf_counter() - self.started))


def smoke(cfg, method, output, fraction):
    directory = Path(output).parent / 'smoke'
    if directory.exists():
        raise RuntimeError('Existing short-test output would be overwritten')
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
        if not row.get('iscrowd', 0): by_image[row['image_id']].add(row['category_id'])
    selected = []
    while categories:
        image = max(by_image, key=lambda k: len(by_image[k] & categories))
        assert by_image[image] & categories
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32: break
        if image not in selected: selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.candidate_probe')
    cfg.custom_hooks.append(dict(type='CandidateAuditHook', method=method, report_path=str(output),
                                 image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (method + '_smoke.py')
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', method, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'fidelity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    register_all_modules(); torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(str(ROOT / 'configs' / args.method / 'rtmdet_tiny_dior_640.py'))
    try:
        if args.mode == 'preflight': preflight(cfg, args.method, args.report)
        elif args.mode == 'smoke': smoke(cfg, args.method, args.report, args.allocator_fraction)
        else:
            if args.method != 'freq_fusion' or not args.reference:
                raise ValueError('Fidelity mode requires FreqFusion and its pinned reference source')
            freq_fidelity(args.reference, args.report, args.allocator_fraction)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=args.method, mode=args.mode,
                                    error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
