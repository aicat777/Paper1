"""Check a queued candidate's exact config without changing its training batch.

The optional --batch is only for an isolated, smaller shared-GPU check. Formal
queues omit it and require microbatch 36 with no gradient accumulation. Dense
repeated annotations exercise allocation, not detection accuracy or speed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import time

import cv2
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from scripts.candidate_probe import METHODS, preflight, report_to, watched_parameters


def capacity_data(cfg, batch, gt_per_image=1200):
    """Use different dense train images, with explicit tiny-to-large box cases."""
    root = Path(cfg.data_root)
    annotation = json.loads((root / cfg.train_dataloader.dataset.ann_file).read_text())
    by_image = {row['id']: [] for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].append(row)
    metadata = {row['id']: row for row in annotation['images']}
    selected = sorted(by_image, key=lambda key: len(by_image[key]), reverse=True)[:batch]
    labels = {row['id']: cfg.classes.index(row['name']) for row in annotation['categories']}
    width, height = cfg.img_scale
    images, boxes = [], []
    for index, image_id in enumerate(selected):
        meta = metadata[image_id]
        image = cv2.imread(str(root / meta['file_name']))
        if image is None:
            raise RuntimeError('Cannot load capacity image: ' + meta['file_name'])
        image = cv2.resize(image, (width, height))
        images.append(torch.from_numpy(image.transpose(2, 0, 1).copy()))
        rows = []
        # Put explicit cases first so none are lost when a real image has more
        # than the target number of annotations. A 4 px box triggers rescue.
        for side in (4, 16, 48, 128, 320, 576):
            x, y = (width - side) / 2, (height - side) / 2
            rows.append([index, 0, x, y, x + side, y + side])
        sx, sy = width / meta['width'], height / meta['height']
        for row in by_image[image_id]:
            x, y, w, h = row['bbox']
            rows.append([index, labels[row['category_id']], x * sx, y * sy,
                         (x + w) * sx, (y + h) * sy])
        rows = torch.tensor(rows, dtype=torch.float32)
        boxes.append(rows[torch.arange(gt_per_image) % len(rows)])
    assert len(images) == batch and len(set(selected)) == batch
    return dict(inputs=torch.stack(images),
                data_samples=dict(bboxes_labels=torch.cat(boxes))), selected


def capacity(cfg, method, output, fraction, batch=None, config_path=None):
    if not 0 < fraction < 1:
        raise ValueError('Allocator fraction must leave GPU memory headroom')
    assert cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or cfg.train_dataloader.batch_size
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    model.cuda().train()
    # Only the probe uses scale 32 and no warmup, to verify real updates. The
    # source config and the queued formal training config remain untouched.
    optim_cfg = deepcopy(cfg.optim_wrapper)
    optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model, method)
    initial = {name: value.detach().cpu().clone() for name, value in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert watched and len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients = {}
    for name, parameter in watched.items():
        def capture(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        parameter.register_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []
    for _ in range(3):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {name: float((value.detach().cpu() - initial[name]).abs().max())
               for name, value in watched.items()}
    inactive = [name for name in watched if not gradients.get(name, 0) > 0 or not updates[name] > 0]
    assert not inactive, inactive
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report = dict(status='passed', method=method, microbatch=batch, accumulation=1,
                  optimizer_updates=3, training_microsteps=3, input_size=list(cfg.img_scale),
                  gt_per_image=1200, different_image_ids=image_ids, repeated_annotations=True,
                  synthetic_multiscale_boxes=True, allocator_fraction=fraction,
                  formal_batch_capacity_tested=batch == cfg.train_dataloader.batch_size,
                  loss_scale_init=32, warmup_enabled=False, amp_skipped_updates=0,
                  saved_checkpoints=False, watched_parameter_tensors=len(watched), losses=losses,
                  gradient_max=gradients, parameter_updates_max=updates,
                  cuda_device=torch.cuda.get_device_name(), cuda_total_gib=total / 2**30,
                  initial_free_gib=free / 2**30,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=time.perf_counter() - started)
    if config_path:
        report.update(config=str(Path(config_path).resolve()),
                      config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest())
    report_to(output, report)
    print('CAPACITY_PASSED', json.dumps({key: value for key, value in report.items()
          if key not in ('gradient_max', 'parameter_updates_max', 'losses')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--mode', choices=('preflight', 'capacity'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.88)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    register_all_modules()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    try:
        cfg = Config.fromfile(args.config)
        if args.mode == 'preflight':
            preflight(cfg, args.method, args.report)
        else:
            capacity(cfg, args.method, args.report, args.allocator_fraction,
                     batch=args.batch, config_path=args.config)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=args.method, mode=args.mode,
                  error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
