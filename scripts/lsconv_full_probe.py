"""Official-entrypoint smoke and capacity checks for expanded LSConv."""
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

import cv2
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules
import torch

from custom_models.lsconv_full.operator import ConvBN, LSConv

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def watched_parameters(model):
    ids = {id(p) for layer in model.modules() if isinstance(layer, (LSConv, ConvBN))
           for p in layer.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def check_sharing(head):
    for towers in (head.cls_convs, head.reg_convs):
        for depth in range(head.stacked_convs):
            operators = [tower[depth][0] for tower in towers]
            for attr in ('cv1', 'cv2', 'cv3'):
                layers = [getattr(op.lkp, attr) for op in operators]
                assert len({id(layer.c) for layer in layers}) == 1
                assert len({id(layer.bn) for layer in layers}) == 3
            assert len({id(op.lkp.cv4) for op in operators}) == 1
            assert len({id(op.lkp.norm) for op in operators}) == 3
            assert len({id(op.bn) for op in operators}) == 3
            projections = [tower[depth][1] for tower in towers]
            assert len({id(layer.c) for layer in projections}) == 1
            assert len({id(layer.bn) for layer in projections}) == 3
    assert not ({id(p) for p in head.cls_convs.parameters()}
                & {id(p) for p in head.reg_convs.parameters()})


def preflight(cfg, output):
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640)
    assert cfg.train_dataloader.batch_size == 12 and cfg.optim_wrapper.accumulative_counts == 3
    assert cfg.custom_hooks[0].gamma == 6000 and cfg.custom_hooks[0].interval == 3
    assert cfg.param_scheduler[0].end == 3000
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    model = MODELS.build(deepcopy(cfg.model)).eval()
    check_sharing(model.bbox_head.head_module)
    assert sum(p.numel() for p in model.parameters()) == 3385305
    assert len(watched_parameters(model)) == 342
    assert len(model.backbone.lsconv_replacements) == 4
    assert len(model.neck.lsconv_replacements) == 7
    macs, calls = [0, 0], []
    handles = []
    def convolution(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    def aggregation(module, inputs, result):
        macs[1] += result.numel() * 9
        calls.append(list(result.shape))
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            handles.append(layer.register_forward_hook(convolution))
        if isinstance(layer, LSConv):
            handles.append(layer.register_forward_hook(aggregation))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert len(calls) == 23
    assert [list(t.shape) for t in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(t.shape) for t in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(loader.dataset.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected
    write_report(output, dict(status='passed', parameters=3385305,
                             conv_gmac=macs[0] / 1e9, aggregation_gmac=macs[1] / 1e9,
                             counted_gmac=sum(macs) / 1e9,
                             runtime_lsconv_calls=23, prediction_points=8400,
                             watched_parameter_tensors=342, scratch=True,
                             microbatch=12, accumulation=3, effective_batch=36,
                             ema_gamma=6000, scale_norms_independent=True,
                             head_convolutions_shared=True, lsconv_shapes=calls))


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
    check_sharing(model.bbox_head.head_module)
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
class FullLSConvAuditHook(Hook):
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
        assert all(int(runner.optim_wrapper.optimizer.state[p]['step']) == 6 for p in self.watched.values())
        assert len(self.losses) == 18 and len(self.metrics) == 2 and self.predictions == 64
        assert self.ema_roundtrips == 2 and self.raw_predictions == 8
        assert runner.optim_wrapper.loss_scaler.get_scale() == 32
        report = dict(status='passed', official_train_entrypoint=True, input_size=[640, 640],
                      microbatch=2, accumulation=3, optimizer_updates=6, training_microsteps=18,
                      val_passes=2, val_images_per_pass=32, predictions_checked=64,
                      raw_predictions_checked=8, ema_validation_roundtrips=2, ema_gamma=6000,
                      coco_evaluation_scoped_to_subset=True, strong_and_weak_augmentation=True,
                      finite_losses=True, finite_predictions=True, amp_skipped_updates=0,
                      watched_parameter_tensors=len(self.watched), gradient_max=self.gradients,
                      parameter_updates_max=updates, loss_scale_init=32, warmup_enabled=False,
                      saved_checkpoints=False, losses=self.losses,
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
    cfg.train_dataloader.dataset.indices = list(range(18))
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
    cfg.custom_imports.imports.append('scripts.lsconv_full_probe')
    cfg.custom_hooks.append(dict(type='FullLSConvAuditHook', report_path=str(output),
                                 image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / 'lsconv_full_smoke.py'
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
    try:
        if args.mode == 'preflight':
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
