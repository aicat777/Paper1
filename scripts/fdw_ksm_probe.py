"""Official-entrypoint smoke and capacity checks for FDW + local KSM."""
import argparse
import ast
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

import cv2
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules
import torch

from custom_models.fdw_ksm.operator import FDWKSMConv2d

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def watched_parameters(model):
    ids = {id(p) for layer in model.modules() if isinstance(layer, FDWKSMConv2d)
           for p in layer.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def official_fidelity(reference_path, output):
    """Run the upstream classes unchanged, excluding imports and registration."""
    reference_path = Path(reference_path)
    source = reference_path.read_text()
    wanted = {'StarReLU', 'KernelSpatialModulation_Global', 'KernelSpatialModulation_Local',
              'get_fft2freq', 'FDConv'}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    for node in nodes:
        node.decorator_list = []  # Do not register the old MMCV convolution type.
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference_path), 'exec'), scope)
    cases = []
    for channels, side in ((48, 80), (96, 40), (192, 20)):
        for training in (False, True):
            torch.manual_seed(71)
            ours = FDWKSMConv2d(torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            # Near-identity initialization could conceal ordering mistakes.
            torch.nn.init.normal_(ours.local.conv.weight, std=.15)
            torch.nn.init.normal_(ours.router.kernel_fc.weight, std=.1)
            original = scope['FDConv'](
                channels, channels, 3, padding=1, bias=False, kernel_num=16,
                ksm_only_kernel_att=True, use_ksm_local=True, use_fbm_if_k_in=[],
                use_fdconv_if_c_gt=0, use_fdconv_if_k_in=[3], kernel_temp=1,
                att_multi=2, param_ratio=1, param_reduction=1)
            def original_name(name):
                return name.replace('router.', 'KSM_Global.').replace('local.', 'KSM_Local.')
            original.load_state_dict({original_name(k): v for k, v in ours.state_dict().items()}, strict=True)
            torch.testing.assert_close(ours.indices, original.indices[0], atol=0, rtol=0)
            ours.train(training)
            original.train(training)
            x = torch.randn(2, channels, side, side, requires_grad=True)
            other_x = x.detach().clone().requires_grad_()
            actual, expected = ours(x), original(other_x)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            gradient = torch.randn_like(actual) / math.sqrt(actual.numel())
            names, parameters = zip(*ours.named_parameters())
            other_parameters = dict(original.named_parameters())
            ours_grads = torch.autograd.grad(actual, (x, *parameters), gradient)
            reference_grads = torch.autograd.grad(expected,
                (other_x, *(other_parameters[original_name(name)] for name in names)), gradient)
            errors = {}
            for name, a, b in zip(('input', *names), ours_grads, reference_grads):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
                errors[name] = float((a - b).abs().max())
            cases.append(dict(channels=channels, spatial_size=side, training=training,
                              forward_max_error=float((actual - expected).abs().max()),
                              gradient_max_error=max(errors.values()), gradient_errors=errors))
    write_report(output, dict(status='passed', source=str(reference_path),
                             source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                             upstream_commit='bf7a33d4f39ac7ea8172158f5628107fcaaba469',
                             local_ksm_enabled=True, global_ksm_axes_enabled=False,
                             fbm_enabled=False, nonuniform_routing_and_local_gates=True,
                             cases=cases))


def check_locations(model):
    operators = [m for m in model.modules() if isinstance(m, FDWKSMConv2d)]
    assert [m.in_channels for m in operators] == [48, 96, 192]
    assert len(watched_parameters(model)) == 33


def preflight(cfg, output):
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640)
    assert cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.custom_hooks[0].get('gamma', 2000) == 2000
    assert cfg.custom_hooks[0].get('interval', 1) == 1
    assert cfg.param_scheduler[0].end == 1000
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    model = MODELS.build(deepcopy(cfg.model)).eval()
    check_locations(model)
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 4901946
    macs, handles = [0, 0], []
    def convolution(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    def dynamic_convolution(module, inputs, result):
        macs[0] += result.numel() * module.in_channels * 9
    def modulation(module, inputs, result):
        macs[1] += result.numel() * module.in_channels * math.prod(module.kernel_size)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            handles.append(layer.register_forward_hook(convolution))
        elif isinstance(layer, torch.nn.Conv1d):
            handles.append(layer.register_forward_hook(modulation))
        elif isinstance(layer, FDWKSMConv2d):
            handles.append(layer.register_forward_hook(dynamic_convolution))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [list(t.shape) for t in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(t.shape) for t in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(loader.dataset.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected
    write_report(output, dict(status='passed', parameters=parameters,
                             added_parameters_vs_fdw=14928,
                             conv2d_gmac=macs[0] / 1e9, local_conv1d_gmac=macs[1] / 1e9,
                             counted_gmac=sum(macs) / 1e9,
                             excludes_fft_kernel_synthesis_and_elementwise_ops=True,
                             prediction_points=8400, watched_parameter_tensors=33,
                             scratch=True, microbatch=36, accumulation=1, effective_batch=36,
                             ema_gamma=2000, replacements=model.backbone.fdw_replacements))


def capacity_data(cfg, batch):
    """Dense real images and explicit multiscale GT coverage; never training data."""
    annotation = json.loads((Path(cfg.data_root) / cfg.train_dataloader.dataset.ann_file).read_text())
    by_image = {item['id']: [] for item in annotation['images']}
    for item in annotation['annotations']:
        if not item.get('iscrowd', 0):
            by_image[item['image_id']].append(item)
    metadata = {item['id']: item for item in annotation['images']}
    selected = sorted(by_image, key=lambda key: len(by_image[key]), reverse=True)[:batch]
    labels = {item['id']: cfg.classes.index(item['name']) for item in annotation['categories']}
    images, boxes = [], []
    for batch_index, image_id in enumerate(selected):
        meta = metadata[image_id]
        image = cv2.imread(str(Path(cfg.data_root) / meta['file_name']))
        if image is None:
            raise RuntimeError('Could not load the capacity-check image')
        image = cv2.resize(image, tuple(cfg.img_scale))
        images.append(torch.from_numpy(image.transpose(2, 0, 1).copy()))
        sx, sy = cfg.img_scale[0] / meta['width'], cfg.img_scale[1] / meta['height']
        rows = []
        for item in by_image[image_id]:
            x, y, w, h = item['bbox']
            rows.append([batch_index, labels[item['category_id']], x * sx, y * sy,
                         (x + w) * sx, (y + h) * sy])
        # Matcher's worst-case allocations depend on GT count. Added boxes also
        # exercise regression paths at all three scales in this short check.
        for side in (16, 48, 128, 320, 576):
            start = (640 - side) / 2
            rows.append([batch_index, 0, start, start, start + side, start + side])
        rows = torch.tensor(rows, dtype=torch.float32)
        boxes.append(rows[torch.arange(1200) % len(rows)])
    assert len(images) == batch and len(set(selected)) == batch
    return dict(inputs=torch.stack(images),
                data_samples=dict(bboxes_labels=torch.cat(boxes))), selected


def capacity(cfg, output, batch, fraction):
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or cfg.train_dataloader.batch_size
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    model.cuda().train()
    check_locations(model)
    cfg.optim_wrapper.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    watched = watched_parameters(model)
    initial = {name: p.detach().cpu().clone() for name, p in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients = {}
    for name, parameter in watched.items():
        def capture(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        parameter.register_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []
    accumulation = cfg.optim_wrapper.get('accumulative_counts', 1)
    for _ in range(3 * accumulation):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {name: float((p.detach().cpu() - initial[name]).abs().max())
               for name, p in watched.items()}
    assert all(gradients.get(name, 0) > 0 and updates[name] > 0 for name in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report = dict(status='passed', microbatch=batch, accumulation=accumulation,
                  optimizer_updates=3, training_microsteps=3 * accumulation,
                  input_size=list(cfg.img_scale), gt_per_image=1200,
                  different_image_ids=image_ids, repeated_annotations=True,
                  synthetic_multiscale_boxes=True, allocator_fraction=fraction,
                  loss_scale_init=32, amp_skipped_updates=0,
                  watched_parameter_tensors=len(watched), losses=losses,
                  gradient_max=gradients, parameter_updates_max=updates,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=time.perf_counter() - started)
    write_report(output, report)
    print('CAPACITY_PASSED', json.dumps({k: v for k, v in report.items()
                                      if k not in ('losses', 'gradient_max', 'parameter_updates_max')}), flush=True)


@HOOKS.register_module()
class FDWKSMAuditHook(Hook):
    """Check actual official training, prediction and EMA; save no checkpoints."""
    priority = 48

    def __init__(self, report_path, image_ids, allocator_fraction=.18):
        self.report_path, self.image_ids = report_path, image_ids
        self.allocator_fraction = allocator_fraction
        self.losses, self.metrics, self.gradients = [], [], {}
        self.predictions = self.raw_predictions = self.ema_roundtrips = 0
        self.peak_allocated = self.peak_reserved = 0
        self.raw_state = None

    def capture_memory(self):
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())

    def before_run(self, runner):
        torch.cuda.set_per_process_memory_fraction(self.allocator_fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        params = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(params) == len({id(p) for p in params})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in params}
        check_locations(self.model)
        for name, parameter in self.watched.items():
            def capture(grad, name=name):
                assert torch.isfinite(grad).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(grad.abs().max()))
            parameter.register_hook(capture)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def check_ema_restore(self):
        if self.raw_state is not None:
            for name, parameter in self.watched.items():
                torch.testing.assert_close(parameter.detach().cpu(), self.raw_state[name], atol=0, rtol=0)
            check_locations(self.model)
            self.ema_roundtrips += 1
            self.raw_state = None

    def before_train_epoch(self, runner):
        self.check_ema_restore()

    @staticmethod
    def check_predictions(outputs):
        for sample in outputs:
            assert torch.isfinite(sample.pred_instances.bboxes).all()
            assert torch.isfinite(sample.pred_instances.scores).all()

    def before_val_epoch(self, runner):
        self.raw_state = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        for metric in runner.val_evaluator.metrics:
            if hasattr(metric, 'img_ids'):
                metric.img_ids = self.image_ids
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.test_step(next(iter(runner.val_dataloader)))
        self.check_predictions(outputs)
        self.capture_memory()
        self.raw_predictions += len(outputs)
        self.model.train(was_training)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        values = {key: float(value) for key, value in outputs.items()}
        assert all(math.isfinite(value) for value in values.values()), values
        self.losses.append(values)
        self.capture_memory()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.check_predictions(outputs)
        self.predictions += len(outputs)
        self.capture_memory()

    def after_val_epoch(self, runner, metrics=None):
        values = {key: metrics[key] for key in ('pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50')}
        assert all(math.isfinite(value) for value in values.values()), values
        self.metrics.append(values)

    def after_train(self, runner):
        self.check_ema_restore()
        updates = {name: float((p.detach().cpu() - self.initial[name]).abs().max())
                   for name, p in self.watched.items()}
        assert all(self.gradients.get(name, 0) > 0 and updates[name] > 0 for name in self.watched)
        assert all(int(runner.optim_wrapper.optimizer.state[p]['step']) == 16 for p in self.watched.values())
        assert len(self.losses) == 16 and len(self.metrics) == 2 and self.predictions == 64
        assert self.ema_roundtrips == 2 and self.raw_predictions == 8
        assert runner.optim_wrapper.loss_scaler.get_scale() == 32
        report = dict(status='passed', official_train_entrypoint=True, input_size=[640, 640],
                      microbatch=2, accumulation=1, optimizer_updates=16, training_microsteps=16,
                      val_passes=2, val_images_per_pass=32, predictions_checked=64,
                      raw_predictions_checked=8, ema_validation_roundtrips=2, ema_gamma=2000,
                      coco_evaluation_scoped_to_subset=True, strong_and_weak_augmentation=True,
                      finite_losses=True, finite_predictions=True, amp_skipped_updates=0,
                      watched_parameter_tensors=len(self.watched), gradient_max=self.gradients,
                      parameter_updates_max=updates, loss_scale_init=32, warmup_enabled=False,
                      saved_checkpoints=False, losses=self.losses, validation_metrics=self.metrics,
                      peak_allocated_gib=self.peak_allocated / 2**30,
                      peak_reserved_gib=self.peak_reserved / 2**30,
                      elapsed_seconds=time.perf_counter() - self.started)
        write_report(self.report_path, report)


def smoke(cfg, output, fraction):
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
    by_image = {item['id']: set() for item in annotation['images']}
    for item in annotation['annotations']:
        if not item.get('iscrowd', 0):
            by_image[item['image_id']].add(item['category_id'])
    unseen, selected = {item['id'] for item in annotation['categories']}, []
    while unseen:
        image = max(by_image, key=lambda key: len(by_image[key] & unseen))
        assert by_image[image] & unseen
        selected.append(image)
        unseen -= by_image[image]
    for item in annotation['images']:
        if len(selected) >= 32:
            break
        if item['id'] not in selected:
            selected.append(item['id'])
    ids = [item['id'] for item in annotation['images']]
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.fdw_ksm_probe')
    cfg.custom_hooks.append(dict(type='FDWKSMAuditHook', report_path=str(output),
                                 image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / 'fdw_ksm_smoke.py'
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', json.dumps({k: v for k, v in report.items()
                                    if k not in ('gradient_max', 'parameter_updates_max', 'losses')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'capacity', 'smoke', 'fidelity'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference', help='Upstream FDConv operator source, for fidelity mode')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.88)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    register_all_modules()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'fidelity':
            if not args.reference:
                raise ValueError('Fidelity mode requires --reference pointing to upstream source')
            official_fidelity(args.reference, args.report)
        elif args.mode == 'preflight':
            preflight(cfg, args.report)
        elif args.mode == 'capacity':
            capacity(cfg, args.report, args.batch, args.allocator_fraction)
        else:
            smoke(cfg, args.report, args.allocator_fraction)
    except Exception as error:
        write_report(args.report, dict(status='failed', mode=args.mode,
                                      error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
