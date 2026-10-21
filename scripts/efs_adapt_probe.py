"""Preflight and short GPU checks for the independent EFS adaptation."""
import argparse
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]
WATCH_PREFIXES = (
    'neck.upsample_layers.2.', 'neck.top_down_layers.2.',
    'neck.bottom_up_layers.0.', 'neck.out_layers.0.',
    'bbox_head.head_module.cls_convs.0.',
    'bbox_head.head_module.reg_convs.0.',
    'bbox_head.head_module.rtm_cls.0.',
    'bbox_head.head_module.rtm_reg.0.')


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def watched_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters()
            if name.startswith(WATCH_PREFIXES)}


def check_sharing(head):
    for towers in (head.cls_convs, head.reg_convs):
        for idx in range(head.stacked_convs):
            assert towers[0][idx].conv.weight.data_ptr() != towers[1][idx].conv.weight.data_ptr()
            assert all(towers[1][idx].conv is towers[level][idx].conv for level in (2, 3))
            assert len({id(towers[level][idx].norm) for level in range(4)}) == 4


def preflight(cfg, output):
    import queue
    from queue import Empty, Queue
    assert Empty and Queue and Path(queue.__file__).name == 'queue.py'
    assert str(ROOT) not in str(Path(queue.__file__).resolve())
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640)
    assert cfg.train_dataloader.batch_size == 9 and cfg.optim_wrapper.accumulative_counts == 4
    assert cfg.custom_hooks[0].gamma == 8000 and cfg.custom_hooks[0].interval == 4
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    model = MODELS.build(deepcopy(cfg.model))
    model.eval()
    check_sharing(model.bbox_head.head_module)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5342340
    seen = {}
    macs = [0]
    handles = []
    def count_macs(module, inputs, result):
        if isinstance(module, torch.nn.Conv2d):
            macs[0] += result.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            macs[0] += result.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(count_macs))
    model.neck.top_down_layers[1].register_forward_hook(
        lambda m, a, out: seen.update(f3=list(out.shape)))
    model.neck.bottom_up_layers[0].register_forward_pre_hook(
        lambda m, a: seen.update(pan_input=list(a[0].shape)))
    with torch.no_grad():
        cls, reg = model.bbox_head.head_module(model.extract_feat(torch.zeros(1, 3, 640, 640)))
    for handle in handles:
        handle.remove()
    assert seen['f3'] == [1, 96, 80, 80] and seen['pan_input'] == [1, 144, 80, 80]
    assert [list(value.shape) for value in cls] == [[1, 20, s, s] for s in (160, 80, 40, 20)]
    assert [list(value.shape) for value in reg] == [[1, 4, s, s] for s in (160, 80, 40, 20)]
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(loader.dataset.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected
    write_report(output, dict(status='passed', parameters=5342340, f3=seen['f3'],
                             pan_input=seen['pan_input'], p2_independent=True,
                             p3_p5_shared=True, stdlib_queue=queue.__file__,
                             conv_linear_gmac=macs[0] / 1e9,
                             prediction_points=34000, scratch=True,
                             microbatch=9, accumulation=4, ema_gamma=8000))


def capacity(cfg, output, batch, fraction):
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    if batch is not None:
        cfg.train_dataloader.batch_size = batch
    batch = cfg.train_dataloader.batch_size
    annotation = json.loads((Path(cfg.data_root) / cfg.train_dataloader.dataset.ann_file).read_text())
    counts = {}
    for item in annotation['annotations']:
        if not item.get('iscrowd', 0):
            counts[item['image_id']] = counts.get(item['image_id'], 0) + 1
    densest = max(counts, key=counts.get)
    meta = next(item for item in annotation['images'] if item['id'] == densest)
    image = cv2.imread(str(Path(cfg.data_root) / meta['file_name']))
    if image is None:
        raise RuntimeError('Could not load the capacity-check image')
    image = cv2.resize(image, tuple(cfg.img_scale))
    inputs = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].repeat(batch, 1, 1, 1)
    labels = {item['id']: cfg.classes.index(item['name']) for item in annotation['categories']}
    boxes = []
    for item in annotation['annotations']:
        if item['image_id'] == densest and not item.get('iscrowd', 0):
            x, y, w, h = item['bbox']
            sx, sy = cfg.img_scale[0] / meta['width'], cfg.img_scale[1] / meta['height']
            boxes.append([0, labels[item['category_id']], x * sx, y * sy,
                          (x + w) * sx, (y + h) * sy])
    boxes = torch.tensor(boxes, dtype=torch.float32)
    boxes = boxes[torch.arange(1200) % len(boxes)].repeat(batch, 1)
    boxes[:, 0] = torch.arange(batch).repeat_interleave(1200)
    data = dict(inputs=inputs, data_samples=dict(bboxes_labels=boxes))
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    model.cuda().train()
    check_sharing(model.bbox_head.head_module)
    cfg.optim_wrapper.loss_scale = dict(init_scale=128)
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    watched = watched_parameters(model)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in watched.items()}
    optimizer_ids = {id(p) for group in optimizer.optimizer.param_groups for p in group['params']}
    assert all(id(parameter) in optimizer_ids for parameter in watched.values())
    gradients = {}
    for name, parameter in watched.items():
        def capture(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        parameter.register_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    accumulation = cfg.optim_wrapper.accumulative_counts
    for _ in range(3 * accumulation):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {name: float((p.detach().cpu() - initial[name]).abs().max())
               for name, p in watched.items()}
    assert all(gradients.get(name, 0) > 0 and updates[name] > 0 for name in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    report = dict(status='passed', microbatch=batch, accumulation=accumulation,
                  optimizer_updates=3, training_microsteps=3 * accumulation,
                  input_size=list(cfg.img_scale), gt_per_image=1200,
                  repeated_annotations=True, allocator_fraction=fraction,
                  loss_scale_init=128, watched_parameter_tensors=len(watched),
                  losses=losses, gradient_max=gradients, parameter_updates_max=updates,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=time.perf_counter() - started)
    write_report(output, report)
    print('CAPACITY_PASSED', json.dumps({k: v for k, v in report.items()
                                      if k not in ('losses', 'gradient_max', 'parameter_updates_max')}), flush=True)


@HOOKS.register_module()
class EFSAdaptAuditHook(Hook):
    """Audit the official training loop without saving short-test checkpoints."""
    priority = 48

    def __init__(self, report_path, image_ids, allocator_fraction=.18):
        self.report_path = report_path
        self.image_ids = image_ids
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
        optimizer_ids = {id(p) for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']}
        assert all(id(p) in optimizer_ids for p in self.watched.values())
        check_sharing(self.model.bbox_head.head_module)
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
            check_sharing(self.model.bbox_head.head_module)
            self.ema_roundtrips += 1
            self.raw_state = None

    def before_train_epoch(self, runner):
        self.check_ema_restore()

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

    @staticmethod
    def check_predictions(outputs):
        for sample in outputs:
            assert torch.isfinite(sample.pred_instances.bboxes).all()
            assert torch.isfinite(sample.pred_instances.scores).all()

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
        assert all(int(runner.optim_wrapper.optimizer.state[p]['step']) == 4 for p in self.watched.values())
        assert len(self.losses) == 16 and len(self.metrics) == 2 and self.predictions == 64
        assert self.ema_roundtrips == 2 and self.raw_predictions == 8
        report = dict(status='passed', official_train_entrypoint=True, input_size=[640, 640],
                      microbatch=2, accumulation=4, optimizer_updates=4, training_microsteps=16,
                      val_passes=2, val_images_per_pass=32, predictions_checked=64,
                      raw_predictions_checked=8, ema_validation_roundtrips=2,
                      ema_gamma=8000, coco_evaluation_scoped_to_subset=True,
                      strong_and_weak_augmentation=True, finite_losses=True,
                      finite_predictions=True, watched_parameter_tensors=len(self.watched),
                      gradient_max=self.gradients, parameter_updates_max=updates,
                      loss_scale_init=128, warmup_enabled=False, saved_checkpoints=False,
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
    cfg.optim_wrapper.loss_scale = dict(init_scale=128)
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
        image = max(by_image, key=lambda i: len(by_image[i] & unseen))
        assert by_image[image] & unseen
        selected.append(image)
        unseen -= by_image[image]
    for image in annotation['images']:
        if len(selected) >= 32:
            break
        if image['id'] not in selected:
            selected.append(image['id'])
    ids = [item['id'] for item in annotation['images']]
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.efs_adapt_probe')
    cfg.custom_hooks.append(dict(type='EFSAdaptAuditHook', report_path=str(output),
                                 image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / 'efs_adapt_smoke.py'
    cfg.dump(str(snapshot))
    command = [sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)]
    result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', json.dumps({k: v for k, v in report.items()
                                    if k not in ('gradient_max', 'parameter_updates_max')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'capacity', 'smoke'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.88)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    register_all_modules()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    if args.mode == 'preflight':
        preflight(cfg, args.report)
    elif args.mode == 'capacity':
        capacity(cfg, args.report, args.batch, args.allocator_fraction)
    else:
        smoke(cfg, args.report, args.allocator_fraction)


if __name__ == '__main__':
    main()
