"""Read-only paired EMA/raw/BN diagnostics; no training or formal-state edits."""
import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from mmengine import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path('/home/pwt/Paper1')
OUT = ROOT / 'work_dirs/diagnostics/lsconv20_analysis'
register_all_modules()
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(.30)
cfg = Config.fromfile(str(ROOT / 'work_dirs/lsconv_20_queue/lsconv_20_640_bs36.py'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint')
args = parser.parse_args()
checkpoint = Path(args.checkpoint) if args.checkpoint else max(
    (ROOT / 'work_dirs/lsconv_20_rtmdet_tiny_dior_640_gpu1_bs36').glob('epoch_*.pth'),
    key=lambda path: int(path.stem.split('_')[1]))
saved = torch.load(checkpoint, map_location='cpu')
ema = saved['state_dict']
raw = {name[len('module.'):]: value for name, value in saved['ema_state_dict'].items()
       if name.startswith('module.')}
assert set(raw) == set(ema)
report = dict(scope=dict(formal_result=False, input_size=640, precision='FP32',
                        gpu='GPU0 shared', paired_same_images=True),
              checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              epoch=saved['meta']['epoch'], iteration=saved['meta']['iter'])
steps = sorted({int(state['step']) for state in saved['optimizer']['state'].values()})
report['optimizer'] = dict(steps=steps, scaler=saved['optimizer'].get('loss_scaler'),
                           expected_steps=saved['meta']['iter'],
                           skipped_steps=saved['meta']['iter'] - min(steps))
report['all_state_finite'] = all(not value.is_floating_point() or torch.isfinite(value).all().item()
                               for state in [raw, ema] for value in state.values())
assert report['all_state_finite']
annotation = json.loads((Path(cfg.data_root) / cfg.val_dataloader.dataset.ann_file).read_text())
by_image = {row['id']: set() for row in annotation['images']}
for row in annotation['annotations']:
    if not row.get('iscrowd', 0):
        by_image[row['image_id']].add(row['category_id'])
remaining, selected = {row['id'] for row in annotation['categories']}, []
while remaining:
    image = max(by_image, key=lambda key: len(by_image[key] & remaining))
    selected.append(image); remaining -= by_image[image]
ids = [row['id'] for row in annotation['images']]
for index in torch.randperm(len(ids), generator=torch.Generator().manual_seed(42)).tolist():
    if ids[index] not in selected:
        selected.append(ids[index])
    if len(selected) == 256:
        break
report['scope'].update(images=len(selected), image_ids=selected, all_twenty_classes=True)
loader_cfg = deepcopy(cfg.val_dataloader)
loader_cfg.update(batch_size=8, num_workers=0, persistent_workers=False)
loader_cfg.dataset.indices = [ids.index(image) for image in selected]
loader = Runner.build_dataloader(loader_cfg, seed=42)
report['paired_metrics'] = {}
hybrid = dict(ema)
bn_keys = [name for name in raw if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
for name in bn_keys:
    hybrid[name] = raw[name]
states = [('ema', ema), ('raw', raw), ('ema_weights_raw_bn_buffers', hybrid)]
model = MODELS.build(deepcopy(cfg.model)).cuda().eval()
report['norm_counts'] = {}
for prefix, module in [('backbone', model.backbone), ('neck', model.neck),
                       ('head', model.bbox_head.head_module)]:
    norms = [layer for layer in module.modules() if isinstance(layer, nn.BatchNorm2d)]
    report['norm_counts'][prefix] = dict(BN=len(norms), GN=sum(isinstance(layer, nn.GroupNorm) for layer in module.modules()),
        bn_momentum=sorted({layer.momentum for layer in norms}), bn_eps=sorted({layer.eps for layer in norms}))
head_baseline_params = 341064
report['parameters'] = dict(total=sum(p.numel() for p in model.parameters()),
    head=sum(p.numel() for p in model.bbox_head.parameters()), baseline_head=head_baseline_params)
for label, state in states:
    model.load_state_dict(state, strict=True)
    model.eval()
    evaluator_cfg = deepcopy(cfg.val_evaluator)
    for metric in evaluator_cfg:
        if metric['type'] == 'mmdet.CocoMetric':
            metric['classwise'] = False
    evaluator = Evaluator(evaluator_cfg)
    evaluator.dataset_meta = loader.dataset.metainfo
    for metric in evaluator.metrics:
        if hasattr(metric, 'img_ids'):
            metric.img_ids = selected
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            samples = model.test_step(batch)
            assert all(torch.isfinite(sample.pred_instances.bboxes).all()
                       and torch.isfinite(sample.pred_instances.scores).all() for sample in samples)
            evaluator.process(data_samples=samples, data_batch=batch)
    metrics = evaluator.evaluate(len(selected))
    report['paired_metrics'][label] = dict(metrics={key: float(value) for key, value in metrics.items()},
                                         seconds=time.perf_counter() - start)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PAIRED', label, report['paired_metrics'][label], flush=True)
    del evaluator

# BN-buffer differences are diagnostics, not a proposed change to evaluation.
report['bn_buffer_difference'] = {}
for prefix in ['backbone.', 'neck.', 'bbox_head.']:
    ratios, shifts = [], []
    for name in bn_keys:
        if name.startswith(prefix) and name.endswith('running_var'):
            ratios.extend((ema[name] / raw[name].clamp_min(1e-8)).tolist())
        elif name.startswith(prefix) and name.endswith('running_mean'):
            variance = raw[name[:-len('running_mean')] + 'running_var'].clamp_min(1e-8)
            shifts.extend(((ema[name] - raw[name]).abs() / variance.sqrt()).tolist())
    report['bn_buffer_difference'][prefix] = dict(
        ema_over_raw_variance_median=statistics.median(ratios),
        normalized_mean_shift_median=statistics.median(shifts))
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print('REPORT', str(OUT / 'report.json'), flush=True)
