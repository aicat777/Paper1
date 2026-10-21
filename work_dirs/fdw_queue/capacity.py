"""GPU1 capacity check before official training; no saved training weights."""
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules


def main(config_path, output):
    logging.disable(logging.CRITICAL)
    register_all_modules()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = Config.fromfile(config_path)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    torch.cuda.set_per_process_memory_fraction(.88)
    batch = cfg.train_dataloader.batch_size
    annotation = json.loads((Path(cfg.data_root) / cfg.train_dataloader.dataset.ann_file).read_text())
    annotations_by_image = {}
    for item in annotation['annotations']:
        if not item.get('iscrowd', 0):
            annotations_by_image.setdefault(item['image_id'], []).append(item)
    candidates = sorted(
        (item for item in annotation['images'] if item['id'] in annotations_by_image),
        key=lambda item: (-len(annotations_by_image[item['id']]), item['id']))
    if len(candidates) < batch:
        raise RuntimeError('The pressure batch needs distinct real images')
    selected = candidates[:batch]
    categories = {item['id']: idx for idx, item in enumerate(annotation['categories'])}
    image_tensors, box_tensors = [], []
    # GAP -> BatchNorm in the router requires variation across samples. Repeating
    # one image makes its variance zero and invalidates the gradient-update check.
    # Keep 1200 GTs per image so the assignment-memory pressure remains unchanged.
    for batch_index, meta in enumerate(selected):
        image = cv2.imread(str(Path(cfg.data_root) / meta['file_name']))
        if image is None:
            raise RuntimeError('Failed to load pressure image: ' + meta['file_name'])
        image = cv2.resize(image, tuple(cfg.img_scale))
        image_tensors.append(torch.from_numpy(image.transpose(2, 0, 1).copy()))
        boxes = []
        for item in annotations_by_image[meta['id']]:
            x, y, w, h = item['bbox']
            sx, sy = cfg.img_scale[0] / meta['width'], cfg.img_scale[1] / meta['height']
            boxes.append([batch_index, categories[item['category_id']], x * sx, y * sy,
                          (x + w) * sx, (y + h) * sy])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        box_tensors.append(boxes[torch.arange(1200) % len(boxes)])
    inputs = torch.stack(image_tensors)
    boxes = torch.cat(box_tensors)
    data = dict(inputs=inputs, data_samples=dict(bboxes_labels=boxes))
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.cuda().train()
    # Short diagnostic must exercise real updates instead of AMP overflow skips.
    cfg.optim_wrapper.loss_scale = dict(init_scale=128)
    optimizer = build_optim_wrapper(model, cfg.optim_wrapper)
    watched = {n: p for n, p in model.named_parameters() if '.dft_weight' in n or '.conv1.conv.router.' in n}
    assert len(watched) == 24
    initial = {n: p.detach().cpu().clone() for n, p in watched.items()}
    optimizer_ids = {id(p) for group in optimizer.optimizer.param_groups for p in group['params']}
    assert all(id(p) in optimizer_ids for p in watched.values())
    gradients = {}
    def hook(name):
        def capture(grad):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        return capture
    for name, parameter in watched.items():
        parameter.register_hook(hook(name))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for _ in range(3):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in watched.items()}
    steps = {n: int(optimizer.optimizer.state[p].get('step', 0)) for n, p in watched.items()}
    inactive = [n for n in watched if gradients.get(n, 0) <= 0 or updates[n] <= 0 or steps[n] != 3]
    report = dict(status='failed' if inactive else 'passed', config=config_path,
                  batch=batch, input_size=list(cfg.img_scale),
                  gt_per_image=1200, repeated_annotations=True, optimizer_updates=3,
                  loss_scale_init=128, allocator_fraction=.88,
                  watched_parameter_tensors=len(watched), losses=losses,
                  distinct_images=len(selected), image_ids=[item['id'] for item in selected],
                  gradient_max=gradients, parameter_updates_max=updates,
                  optimizer_steps=steps, inactive_parameters=inactive,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=time.perf_counter() - started)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if inactive:
        raise RuntimeError('Pressure check parameters were inactive: ' + ', '.join(inactive))


if __name__ == '__main__':
    main(*sys.argv[1:])
