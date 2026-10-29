"""CPU-only endpoint audit of actual selective KL and shared-head gradients.

Compare completed raw epoch300 ROI and transfer checkpoints on identical
strong/weak augmented B36 inputs. No optimizer updates or GPU allocation.
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
cfg = Config.fromfile(str(ROOT / 'work_dirs/instance_semantic_transfer_queue/'
                          'instance_semantic_transfer_640_bs36.py'))
started = time.monotonic()


def comparison(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    aa = sum((x.double().square().sum() for x, _ in pairs), torch.tensor(0.))
    bb = sum((y.double().square().sum() for _, y in pairs), torch.tensor(0.))
    ab = sum(((x.double() * y.double()).sum() for x, y in pairs), torch.tensor(0.))
    return dict(cosine=float(ab / (aa * bb).sqrt().clamp_min(1e-30)),
                transfer_over_reference_norm=float((bb / aa.clamp_min(1e-30)).sqrt()),
                reference_norm=float(aa.sqrt()), transfer_norm=float(bb.sqrt()),
                dot=float(ab))


def load(method):
    directory = ROOT / 'work_dirs' / (method + '_rtmdet_tiny_dior_640_gpu1_bs36')
    saved = torch.load(directory / 'epoch_300.pth', map_location='cpu', weights_only=False)
    raw = {k[7:]: v for k, v in saved['ema_state_dict'].items() if k.startswith('module.')}
    assert raw and set(raw) == set(saved['state_dict'])
    assert all(bool(v.isfinite().all()) for v in raw.values())
    model = MODELS.build(deepcopy(cfg.model))
    model.load_state_dict(raw, strict=True)
    model.cpu().train()
    meta = {k: saved['meta'][k] for k in ('epoch', 'iter', 'seed', 'time')}
    updates = sorted({float(v['step']) for v in saved['optimizer']['state'].values()
                      if 'step' in v})
    meta['optimizer_steps'] = updates
    assert meta['epoch'] == 300
    return model, meta


random.seed(431)
np.random.seed(431)
torch.manual_seed(431)
preprocessor = MODELS.build(deepcopy(cfg.model.data_preprocessor))
batches = []
for phase, count in [('strong', 2), ('weak', 1)]:
    loader_cfg = deepcopy(cfg.train_dataloader)
    loader_cfg.update(batch_size=36, num_workers=0, persistent_workers=False,
                      sampler=dict(type='DefaultSampler', shuffle=False))
    loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 144, replace=False).tolist()
    if phase == 'weak':
        loader_cfg.dataset.pipeline = deepcopy(cfg.train_pipeline_stage2)
    random.seed(431)
    np.random.seed(431)
    torch.manual_seed(431)
    loader = Runner.build_dataloader(loader_cfg, seed=431)
    for index, batch in enumerate(loader):
        if index == 0:
            continue
        batches.append((phase, preprocessor(batch, training=True)))
        if index == count:
            break
    del loader
del preprocessor
gc.collect()
result = dict(
    status='running',
    scope='Completed raw epoch300 ROI and transfer weights, identical two strong '
          'B36 batches and one weak B36 batch; CPU FP32 batch BN, no optimizer '
          'updates. Backbone/neck run under no_grad; measured gradients are at '
          'neck outputs, final classification features and shared classification '
          'tower parameters. Original ROI weights use identical indexed loss '
          'plumbing and a counterfactual transfer term only for measurement.',
    methods={}, limitations=[
        'Endpoint, training-image probes do not establish the entire optimization history.',
        'Infinitesimal gradient directions are not finite AdamW updates or AP effects.',
        'Backbone/neck parameter gradients are outside this measurement.',
        'Weak and strong pipelines use some of the same source images; these are '
        '108 input views, not 108 independent original images.'])
for method in ['instance_semantic', 'instance_semantic_transfer']:
    model, meta = load(method)
    head = model.bbox_head
    cls_params = tuple(p for name, p in head.head_module.named_parameters()
                       if name.startswith('cls_convs.'))
    records, points = [], []
    for index, (phase, batch) in enumerate(batches):
        with torch.no_grad():
            neck = model.extract_feat(batch['inputs'])
        neck = tuple(x.detach().requires_grad_() for x in neck)
        capture = {}
        hook = head.local_transfer.register_forward_pre_hook(
            lambda module, inputs: capture.update(inputs=inputs))
        losses = head.loss(neck, batch['data_samples'])
        hook.remove()
        features, scores, assigned, rows, lookup, teacher, classifier = capture['inputs']
        targets = (*neck, *features, *cls_params)
        gradients = {}
        for key, loss in [('main', losses['loss_cls'] + losses['loss_bbox']),
                          ('roi', losses['loss_instance_semantic']),
                          ('transfer', losses['loss_instance_transfer'])]:
            values = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
            gradients[key] = tuple(torch.zeros_like(p) if g is None else g
                                   for p, g in zip(targets, values))
        groups = dict(neck_outputs=slice(0, 3), classification_features=slice(3, 6),
                      classification_tower_parameters=slice(6, None))
        measured = {}
        for label, slots in groups.items():
            gm, gr, gt = (gradients[k][slots] for k in ('main', 'roi', 'transfer'))
            measured[label] = dict(main_vs_transfer=comparison(gm, gt),
                                   roi_vs_transfer=comparison(gr, gt),
                                   main_vs_roi=comparison(gm, gr))
        chosen = []
        offset = 0
        for level, (feature, score) in enumerate(zip(features, scores)):
            h, w = feature.shape[-2:]
            slots = assigned['assigned_gt_indices'][:, offset:offset + h * w]
            b, p = (slots >= 0).nonzero(as_tuple=True)
            ri = lookup[b, slots[b, p]]
            good = ri >= 0
            b, p, ri = b[good], p[good], ri[good]
            y, x = p // w, p % w
            label = rows[ri, 1].long()
            pred = score[b, :, y, x].detach()
            quality = assigned['assign_metrics'][b, offset + p]
            keep = ((quality >= .5) & (teacher[ri].detach().argmax(-1) == label) &
                    (pred.argmax(-1) != label))
            for j in keep.nonzero().flatten().tolist():
                chosen.append((level, int(b[j]), int(y[j]), int(x[j]),
                               int(ri[j]), float(quality[j])))
            offset += h * w
        counts = torch.bincount(torch.tensor([x[4] for x in chosen], dtype=torch.long),
                                minlength=len(rows))
        participating = int((counts > 0).sum())
        assert len(chosen) == int(losses['transfer_selected_points'])
        assert participating == int(losses['transfer_selected_gt'])
        for level, b, y, x, ri, quality in chosen:
            z = features[level][b, :, y, x].detach().float()
            wa = classifier.weight.detach()
            ba = classifier.bias.detach()
            local = F.linear(z, wa, ba)
            p, q = local.softmax(-1), teacher[ri].detach().softmax(-1)
            label = int(rows[ri, 1])
            gt = gradients['transfer'][3 + level][b, :, y, x]
            gm = gradients['main'][3 + level][b, :, y, x]
            gr = gradients['roi'][3 + level][b, :, y, x]
            scale = .1 / participating / int(counts[ri])
            closed = scale * ((p - q) @ wa)
            torch.testing.assert_close(gt, closed, rtol=2e-4, atol=2e-7)
            wd = head.head_module.rtm_cls[level].weight.detach()[:, :, 0, 0]
            bd = head.head_module.rtm_cls[level].bias.detach()
            actual = scores[level][b, :, y, x].detach()
            torch.testing.assert_close(F.linear(z, wd, bd), actual, rtol=2e-5, atol=2e-5)
            wrong = int(actual.argmax())
            margin_change = float(-((wd[label] - wd[wrong]) * gt).sum())
            cosine = float((gm * gt).sum() / (gm.norm() * gt.norm()).clamp_min(1e-30))
            ce_grad = scale * ((p - F.one_hot(torch.tensor(label), 20)) @ wa)
            widths = rows[ri, 4:6] - rows[ri, 2:4]
            roi_level = int(torch.floor(torch.log2(widths.prod().sqrt() / 56 + 1e-6)).clamp(0, 2))
            points.append(dict(
                batch=index, phase=phase, gt_row=ri, level=level + 3,
                roi_level=roi_level + 3, iou=quality, gt_class=label,
                auxiliary_point_correct=int(local.argmax()) == label,
                auxiliary_point_gt_probability=float(p[label]), roi_gt_probability=float(q[label]),
                roi_less_gt_confident_than_point=bool(q[label] < p[label]),
                kl=float((q * (q.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum()),
                cosine_with_main=cosine,
                transfer_over_main_norm=float(gt.norm() / gm.norm().clamp_min(1e-30)),
                qfl_increased_first_order=bool((gm * gt).sum() < 0),
                margin_worsened_first_order=margin_change < 0,
                point_aux_gt_score_lowered_first_order=bool((wa[label] * gt).sum() > 0),
                detector_gt_score_lowered_first_order=bool((wd[label] * gt).sum() > 0),
                hard_aux_ce_increases_qfl_first_order=bool((gm * ce_grad).sum() < 0),
                combined_gradient_reverses_main=bool((gm * (gm + gr + gt)).sum() < 0),
                original_roi_gradient_reverses_main=bool((gm * (gm + gr)).sum() < 0),
                G_over_M=len(rows) / participating,
                local_G_over_M_vs_all_GT_normalization=len(rows) / participating))
        record = dict(batch=index, phase=phase, valid_gt=len(rows), selected_gt=participating,
                      selected_points=len(chosen), roi_accuracy=float(
                          (teacher.detach().argmax(-1) == rows[:, 1].long()).float().mean()),
                      losses={k: float(v.detach()) for k, v in losses.items()},
                      gradients=measured, elapsed_seconds=time.monotonic() - started)
        records.append(record)
        print(json.dumps(dict(method=method, **record)), flush=True)
        del capture, features, scores, assigned, rows, lookup, teacher, classifier, neck
        del losses, targets, gradients, values, loss, gm, gr, gt
        gc.collect()
    result['methods'][method] = dict(checkpoint_meta=meta, records=records, points=points)
    (OUT / 'gradient_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    del model, head, cls_params
    gc.collect()
result['status'] = 'complete'
(OUT / 'gradient_probe.json').write_text(json.dumps(result, indent=2) + '\n')
print('DONE', flush=True)
