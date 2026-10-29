"""Read-only ROI/main-loss gradients on an isolated epoch-200 GSCE model."""
from copy import deepcopy
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
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
directory = ROOT / 'work_dirs/scdown_p3_g4_semantic_rtmdet_tiny_dior_640_gpu1_bs36'
cfg = Config.fromfile(str(directory / 'scdown_p3_g4_semantic_640_bs36.py'))
checkpoint = directory / 'epoch_200.pth'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
raw = {k.removeprefix('module.'): v for k, v in saved['ema_state_dict'].items()
       if k.startswith('module.')}
assert raw and set(raw) == set(saved['state_dict'])
model = MODELS.build(deepcopy(cfg.model))
model.load_state_dict(raw, strict=True)
model.cpu().train()
meta = {k: saved['meta'][k] for k in ('epoch', 'iter', 'seed', 'time')}
del saved, raw
loader_cfg = deepcopy(cfg.train_dataloader)
loader_cfg.update(batch_size=36, num_workers=0, persistent_workers=False,
                  sampler=dict(type='DefaultSampler', shuffle=False))
loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 144, replace=False).tolist()
loader = Runner.build_dataloader(loader_cfg, seed=431)
head = model.bbox_head
records = []
started = time.monotonic()


def norm(xs):
    return sum(x.double().square().sum() for x in xs).sqrt()


def comparison(a, b):
    na, nb = norm(a), norm(b)
    dot = sum((x.double() * y.double()).sum() for x, y in zip(a, b))
    return dict(cosine=float(dot / (na * nb).clamp_min(1e-30)),
                first_norm=float(na), second_norm=float(nb),
                second_over_first_norm=float(nb / na.clamp_min(1e-30)))


for index, batch in enumerate(loader):
    batch = model.data_preprocessor(batch, training=True)
    if index == 0:
        continue
    with torch.no_grad():
        neck = model.extract_feat(batch['inputs'])
    neck = tuple(x.detach().requires_grad_() for x in neck)
    cls, reg, features = head.head_module(neck, return_cls_features=True)
    gt = batch['data_samples']['bboxes_labels']
    losses = head.loss_by_feat(cls, reg, gt, batch['data_samples']['img_metas'])
    aux = head.instance_semantic(features, gt)
    gradients = {}
    for key, loss in [('cls', losses['loss_cls']), ('bbox', losses['loss_bbox']), ('roi', aux)]:
        grads = torch.autograd.grad(loss, neck, retain_graph=True, allow_unused=True)
        gradients[key] = tuple(torch.zeros_like(x) if g is None else g for x, g in zip(neck, grads))
        assert all(bool(g.isfinite().all()) for g in gradients[key])
    gradients['main'] = tuple(a + b for a, b in zip(gradients['cls'], gradients['bbox']))
    record = dict(batch=index, images=36,
                  valid_gt=int(((gt[:, 4:6] > gt[:, 2:4]).all(-1)).sum()),
                  losses={**{k: float(v.detach()) for k, v in losses.items()},
                          'loss_instance_semantic': float(aux.detach())},
                  comparisons={key + '_vs_roi': comparison(gradients[key], gradients['roi'])
                               for key in ('cls', 'bbox', 'main')},
                  by_level={str(level + 3): {
                      key + '_vs_roi': comparison((gradients[key][level],), (gradients['roi'][level],))
                      for key in ('cls', 'bbox', 'main')}
                      for level in range(3)},
                  elapsed_seconds=time.monotonic() - started)
    records.append(record)
    print(json.dumps(record), flush=True)
    del neck, cls, reg, features, gt, losses, aux, gradients, grads, batch

result = dict(scope='Frozen raw epoch200 GSCE+ROI; 108 fixed strong-augmented train images; '
                    'CPU FP32, isolated B36 batch-BN; gradients measured at the shared neck '
                    'outputs before classification/regression towers; no parameter updates.',
              checkpoint=str(checkpoint), checkpoint_meta=meta,
              auxiliary_weight_included=0.1, records=records,
              limitations=['A local gradient cosine is not a causal decomposition of AP.',
                           'Neck features are detached to isolate the shared detection-head input; '
                           'this does not measure parameter gradients in earlier backbone layers.',
                           'One checkpoint and three augmented batches do not establish the '
                           'direction of auxiliary gradients throughout training.'])
(OUT / 'gradient_probe.json').write_text(json.dumps(result, indent=2) + '\n')
print('DONE', flush=True)
