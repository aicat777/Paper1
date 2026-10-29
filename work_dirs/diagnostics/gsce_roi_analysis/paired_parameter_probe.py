"""Compare ROI and detector gradients on isolated, completed models.

No training-process access, GPU allocation, optimizer step or checkpoint writes.
The backbone is frozen for this probe; measured parameters are in the neck
and classification tower, through which both tasks train shared features.
"""
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
register_all_modules()
logging.disable(logging.CRITICAL)
random.seed(431)
np.random.seed(431)
torch.manual_seed(431)
METHODS = {
    'original_roi': 'instance_semantic',
    'g4_roi': 'scdown_p3_g4_semantic',
}
started = time.monotonic()


def load(method):
    directory = ROOT / 'work_dirs' / (
        METHODS[method] + '_rtmdet_tiny_dior_640_gpu1_bs36')
    cfg = Config.fromfile(str(directory / (METHODS[method] + '_640_bs36.py')))
    saved = torch.load(directory / 'epoch_300.pth', map_location='cpu', weights_only=False)
    # EMAHook stores raw parameters under ema_state_dict.module.* after swapping
    # the state_dict to EMA values when saving a checkpoint.
    raw = {k.removeprefix('module.'): v for k, v in saved['ema_state_dict'].items()
           if k.startswith('module.')}
    assert raw and set(raw) == set(saved['state_dict'])
    model = MODELS.build(deepcopy(cfg.model))
    model.load_state_dict(raw, strict=True)
    model.cpu().train()
    meta = {k: saved['meta'][k] for k in ('epoch', 'iter', 'seed', 'time')}
    assert meta['epoch'] == 300
    return cfg, model, meta


def compare(first, second):
    pairs = [(a, b) for a, b in zip(first, second) if a is not None and b is not None]
    if not pairs:
        return dict(parameters=0)
    aa = sum(a.double().square().sum() for a, _ in pairs)
    bb = sum(b.double().square().sum() for _, b in pairs)
    ab = sum((a.double() * b.double()).sum() for a, b in pairs)
    return dict(parameters=len(pairs), cosine=float(ab / (aa * bb).sqrt().clamp_min(1e-30)),
                auxiliary_over_main_norm=float((bb / aa.clamp_min(1e-30)).sqrt()),
                main_norm=float(aa.sqrt()), auxiliary_norm=float(bb.sqrt()))


cfg, model, _ = load('original_roi')
loader_cfg = deepcopy(cfg.train_dataloader)
loader_cfg.update(batch_size=36, num_workers=0, persistent_workers=False,
                  sampler=dict(type='DefaultSampler', shuffle=False))
loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 108, replace=False).tolist()
random.seed(431)
np.random.seed(431)
torch.manual_seed(431)
loader = Runner.build_dataloader(loader_cfg, seed=431)
batches = []
for index, batch in enumerate(loader):
    processed = model.data_preprocessor(batch, training=True)
    if index:
        batches.append(processed)
    if len(batches) == 2:
        break
del loader, model
gc.collect()

result = dict(
    scope='Two completed raw epoch300 models; identical 72 strong-augmented train '
          'images, isolated CPU FP32, real B36 batch BN. Backbone parameters '
          'are frozen; neck and classification-tower gradients are measured. '
          'No optimizer updates and no GPU use.',
    auxiliary_weight_included=0.1,
    methods={},
    limitations=[
        'Local gradient directions do not causally decompose validation AP.',
        'Only two batches at convergence are measured, not training history.',
        'Backbone parameter gradients and AdamW preconditioning are not measured.',
        'Strong augmentation here is a diagnostic input; formal last20 epochs use weak augmentation.',
    ])
for method in METHODS:
    _, model, meta = load(method)
    head = model.bbox_head
    selected = [(name, p) for name, p in model.named_parameters()
                if name.startswith('neck.') or
                name.startswith('bbox_head.head_module.cls_convs.')]
    names, params = zip(*selected)
    records = []
    for index, batch in enumerate(batches):
        with torch.no_grad():
            backbone = model.backbone(batch['inputs'])
        neck = model.neck(backbone)
        cls, reg, features = head.head_module(neck, return_cls_features=True)
        rows = batch['data_samples']['bboxes_labels']
        losses = head.loss_by_feat(cls, reg, rows, batch['data_samples']['img_metas'])
        aux = head.instance_semantic(features, rows)
        gradients = {}
        for key, loss in [('cls', losses['loss_cls']), ('bbox', losses['loss_bbox']),
                          ('roi', aux)]:
            gradients[key] = torch.autograd.grad(loss, params, retain_graph=True,
                                                 allow_unused=True)
        main = tuple(None if a is None and b is None else
                     (torch.zeros_like(p) if a is None else a) +
                     (torch.zeros_like(p) if b is None else b)
                     for a, b, p in zip(gradients['cls'], gradients['bbox'], params))
        groups = {
            'all_shared': lambda n: True,
            'neck': lambda n: n.startswith('neck.'),
            'classification_tower': lambda n: n.startswith('bbox_head.'),
        }
        measurements = {}
        for label, condition in groups.items():
            indices = [i for i, name in enumerate(names) if condition(name)]
            measurements[label] = {
                key + '_vs_roi': compare([values[i] for i in indices],
                                        [gradients['roi'][i] for i in indices])
                for key, values in [('main', main), ('cls', gradients['cls']),
                                    ('bbox', gradients['bbox'])]}
        valid_rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(-1)]
        with torch.no_grad():
            rois = torch.cat((valid_rows[:, :1], valid_rows[:, 2:6]), 1)
            pooled = head.instance_semantic.pool_instances(features, rois)
            logits = head.instance_semantic.classifier(pooled)
            target = valid_rows[:, 1].long()
            roi_accuracy = float((logits.argmax(-1) == target).float().mean())
        record = dict(batch=index, images=len(batch['inputs']), valid_gt=len(valid_rows),
                      losses={**{k: float(v.detach()) for k, v in losses.items()},
                              'loss_instance_semantic': float(aux.detach())},
                      roi_accuracy=roi_accuracy, gradients=measurements,
                      elapsed_seconds=time.monotonic() - started)
        records.append(record)
        print(json.dumps(dict(method=method, **record)), flush=True)
        del backbone, neck, cls, reg, features, losses, aux, loss, gradients, main, pooled, logits
        gc.collect()
    result['methods'][method] = dict(checkpoint_meta=meta, records=records)
    del model, head, selected, params
    gc.collect()
    (OUT / 'paired_parameter_probe.json').write_text(json.dumps(result, indent=2) + '\n')
print('DONE', flush=True)
