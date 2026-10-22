"""Read-only CPU audit: one frozen ROI model, identical main/auxiliary features."""
from collections import defaultdict
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from custom_models.roi_supervision_common.supervision import valid_rows_and_matches
from custom_models.roi_supervision_common.assigner import ROISemanticIndexedAssigner

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
register_all_modules()
logging.disable(logging.CRITICAL)
random.seed(431)
np.random.seed(431)
torch.manual_seed(431)
method = os.environ.get('SEMANTIC_PROBE_METHOD', 'roi')
assert method in ('roi', 'matched', 'random')
suffix = '' if method == 'roi' else '_' + method
directory = ROOT / f'work_dirs/instance_semantic{suffix}_rtmdet_tiny_dior_640_gpu1_bs36'
cfg = Config.fromfile(str(directory / f'instance_semantic{suffix}_640_bs36.py'))
cfg.model.train_cfg.assigner.type = 'ROISemanticIndexedAssigner'
checkpoint = directory / 'epoch_300.pth'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
raw = {k.removeprefix('module.'): v for k, v in saved['ema_state_dict'].items() if k.startswith('module.')}
assert raw and set(raw) == set(saved['state_dict'])
model = MODELS.build(deepcopy(cfg.model))
model.load_state_dict(raw, strict=True)
model.cpu().train()
del saved, raw
loader_cfg = deepcopy(cfg.train_dataloader)
loader_cfg.update(batch_size=36, num_workers=0, persistent_workers=False,
                  sampler=dict(type='DefaultSampler', shuffle=False))
loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 144, replace=False).tolist()
loader = Runner.build_dataloader(loader_cfg, seed=431)
head = model.bbox_head
capture = {}
handle = head.assigner.register_forward_hook(lambda mod, args, result: capture.update(result))
records = []
instance_rows = []
started = time.monotonic()


def norm(xs):
    return sum((x.float().square().sum() for x in xs)).sqrt()


def cosine(xs, ys):
    dot = sum((x.float() * y.float()).sum() for x, y in zip(xs, ys))
    return float(dot / (norm(xs) * norm(ys)).clamp_min(1e-30))


for index, batch in enumerate(loader):
    batch = model.data_preprocessor(batch, training=True)
    if index == 0:
        continue
    with torch.no_grad():
        pyramid = model.extract_feat(batch['inputs'])
        _, reg, features = head.head_module(pyramid, return_cls_features=True)
    features = tuple(x.detach().requires_grad_() for x in features)
    cls = tuple(layer(x) for layer, x in zip(head.head_module.rtm_cls, features))
    gt = batch['data_samples']['bboxes_labels']
    losses = head.loss_by_feat(cls, reg, gt, batch['data_samples']['img_metas'])
    padded = gt_instances_preprocess(gt, 36)
    rows, matched = valid_rows_and_matches(gt, padded, capture['assigned_gt_indices'])
    rois = torch.cat((rows[:, :1], rows[:, 2:6]), 1)
    pooled = head.instance_semantic.pool_instances(features, rois)
    logits = head.instance_semantic.classifier(pooled)
    ce = F.cross_entropy(logits, rows[:, 1].long(), reduction='none')
    scaled = ce * (head.instance_semantic.loss_weight / len(rows))
    official = head.instance_semantic(features, gt)
    torch.testing.assert_close(scaled.sum(), official, rtol=2e-6, atol=1e-7)
    gm = torch.autograd.grad(scaled[matched].sum(), features, retain_graph=True)
    gu = torch.autograd.grad(scaled[~matched].sum(), features, retain_graph=True)
    gq = torch.autograd.grad(losses['loss_cls'], features, retain_graph=False)
    total = tuple(a + b for a, b in zip(gm, gu))
    assert all(bool(torch.isfinite(g).all()) for g in (*gm, *gu, *gq))
    with torch.no_grad():
        probability = logits.softmax(-1)
        error = probability.clone()
        error[torch.arange(len(rows)), rows[:, 1].long()] -= 1
        descriptor_gradient = error @ head.instance_semantic.classifier.weight
        descriptor_norm = descriptor_gradient.norm(dim=1)
        sizes = rows[:, 4:6] - rows[:, 2:4]
        classes = list(cfg.classes)
        for i in range(len(rows)):
            instance_rows.append(dict(batch=index, row_index=i, matched=bool(matched[i]),
                category=classes[int(rows[i, 1])], width=float(sizes[i, 0]),
                height=float(sizes[i, 1]), ce=float(ce[i]),
                descriptor_gradient_l2_unweighted=float(descriptor_norm[i]),
                correct=bool(logits[i].argmax() == rows[i, 1])))
        record = dict(batch=index, images=36, valid_gt=len(rows), matched=int(matched.sum()),
            unmatched=int((~matched).sum()), unmatched_ce_fraction=float(ce[~matched].sum()/ce.sum()),
            unmatched_descriptor_gradient_norm_sum_fraction=float(descriptor_norm[~matched].sum()/descriptor_norm.sum()),
            matched_mean_ce=float(ce[matched].mean()), unmatched_mean_ce=float(ce[~matched].mean()),
            feature_gradient_l2_matched=float(norm(gm)), feature_gradient_l2_unmatched=float(norm(gu)),
            feature_gradient_l2_all=float(norm(total)),
            feature_gradient_unmatched_to_all_norm_ratio=float(norm(gu)/norm(total)),
            feature_gradient_matched_unmatched_cosine=cosine(gm, gu),
            feature_gradient_main_matched_cosine=cosine(gq, gm),
            feature_gradient_main_unmatched_cosine=cosine(gq, gu),
            elapsed_seconds=time.monotonic()-started)
        records.append(record)
        print('BATCH', json.dumps(record), flush=True)
    del pyramid, features, cls, reg, losses, pooled, logits, ce, scaled, official, gm, gu, gq, total, batch
    capture.clear()
handle.remove()


def summarize(rows):
    n = len(rows)
    ce = sum(x['ce'] for x in rows)
    grads = sum(x['descriptor_gradient_l2_unweighted'] for x in rows)
    return dict(n=n, ce_sum=ce, ce_mean=ce/n,
        descriptor_gradient_norm_sum=grads, descriptor_gradient_norm_mean=grads/n,
        accuracy=sum(x['correct'] for x in rows)/n,
        mean_short_side=float(np.mean([min(x['width'],x['height']) for x in rows])),
        median_short_side=float(np.median([min(x['width'],x['height']) for x in rows])),
        below_stride8_fraction=sum(min(x['width'],x['height'])<8 for x in rows)/n)


groups = {key:summarize([x for x in instance_rows if x['matched']==flag])
          for key,flag in [('matched',True),('unmatched',False)]}
u, m = groups['unmatched'], groups['matched']
summary = dict(groups=groups, unmatched_gt_fraction=u['n']/(u['n']+m['n']),
    unmatched_ce_fraction=u['ce_sum']/(u['ce_sum']+m['ce_sum']),
    unmatched_descriptor_gradient_norm_sum_fraction=u['descriptor_gradient_norm_sum']/(u['descriptor_gradient_norm_sum']+m['descriptor_gradient_norm_sum']),
    unmatched_to_matched_mean_ce=u['ce_mean']/m['ce_mean'],
    unmatched_to_matched_mean_descriptor_gradient_norm=u['descriptor_gradient_norm_mean']/m['descriptor_gradient_norm_mean'])
result = dict(checkpoint=str(checkpoint), weights='raw training weights from ema_state_dict.module.*',
    device='cpu', precision='FP32', training_mode=True, batch_size=36,
    sample='108 fixed real train images with original strong augmentation after one 36-image cache warmup',
    optimizer_updates=0, running_training_untouched=True,
    note='BN batch statistics are computed in this isolated model copy. No checkpoint is saved.',
    limitations=['Converged snapshot and a small augmented sample do not measure full-training causal contributions.',
        'Descriptor gradients precede ROI spatial spreading; feature-map group gradient norms include overlap and cancellation.',
        'CE and gradient fractions are not detection AP or additive shares of AP improvement.'],
    summary=summary, batches=records, instances=instance_rows)
(OUT/f'semantic_signal_probe{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')
print('SUMMARY', json.dumps(summary), flush=True)
