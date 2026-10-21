"""Read-only gradient audit of a frozen in-progress checkpoint, no updates."""
from collections import defaultdict
from copy import deepcopy
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
register_all_modules()
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
random.seed(431); np.random.seed(431); torch.manual_seed(431)
free, total = torch.cuda.mem_get_info()
assert free >= total * .30 + 2 * 2**30
torch.cuda.set_per_process_memory_fraction(.30)
torch.backends.cudnn.benchmark = False


def flat(maps):
    return torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
                      for x in maps], 1).float()


def cosine(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def gradients(loss, values):
    result = torch.autograd.grad(loss * 8192., values, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p).float() if g is None else g.float() / 8192.
            for g, p in zip(result, values)]


directory = ROOT / 'work_dirs/instance_semantic_rank_rtmdet_tiny_dior_640_gpu1_bs36'
cfg = Config.fromfile(str(ROOT / 'work_dirs/instance_semantic_rank_queue/instance_semantic_rank_640_bs36.py'))
checkpoint = directory / 'epoch_200.pth'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
raw = {k.removeprefix('module.'): v for k, v in saved['ema_state_dict'].items()
       if k.startswith('module.')}
assert raw and set(raw) == set(saved['state_dict'])
state = saved['optimizer'].get('state', saved['optimizer'].get('optimizer', {}).get('state', {}))
integrity = dict(checkpoint=str(checkpoint), epoch=saved['meta']['epoch'], iteration=saved['meta']['iter'],
    optimizer_steps=sorted({int(v['step']) for v in state.values() if 'step' in v}),
    raw_bn_counts=sorted({int(v) for k, v in raw.items() if k.endswith('num_batches_tracked')}),
    nonfinite_weights=[key+'.'+name for key in ('state_dict','ema_state_dict')
        for name,value in saved[key].items() if isinstance(value,torch.Tensor)
        and value.is_floating_point() and not torch.isfinite(value).all()],
    loss_scaler=saved['optimizer'].get('loss_scaler'))
assert not integrity['nonfinite_weights']
model = MODELS.build(deepcopy(cfg.model))
model.load_state_dict(raw, strict=True)
model.cuda().train()
del saved, state, raw
loader_cfg = deepcopy(cfg.train_dataloader)
loader_cfg.update(batch_size=36,num_workers=0,persistent_workers=False,
                  sampler=dict(type='DefaultSampler',shuffle=False))
loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 144, replace=False).tolist()
loader = Runner.build_dataloader(loader_cfg, seed=431)
head = model.bbox_head
captured = {}
handle = head.assigner.register_forward_hook(lambda mod, args, result: captured.update(result))
records, raw_vectors = [], defaultdict(list)
parameter_list = list(head.head_module.rtm_cls.parameters())
for index, batch in enumerate(loader):
    batch = model.data_preprocessor(batch, training=True)
    # The cache is warmed by the first batch; diagnostics use the next 108
    # real augmented main images at the actual training microbatch of 36.
    if index == 0:
        continue
    with torch.no_grad(), torch.autocast('cuda'):
        pyramid = model.extract_feat(batch['inputs'])
    with torch.autocast('cuda'):
        cls, reg, features = head.head_module(pyramid, return_cls_features=True)
        rows = batch['data_samples']['bboxes_labels']
        losses = head.loss_by_feat(cls, reg, rows, batch['data_samples']['img_metas'])
        semantic = head.instance_semantic(features, rows)
    q, rank = gradients(losses['loss_cls'], cls), gradients(losses['loss_instance_rank'], cls)
    q, rank = flat(q), flat(rank)
    qparam = torch.cat([x.flatten() for x in gradients(losses['loss_cls'], parameter_list)])
    rparam = torch.cat([x.flatten() for x in gradients(losses['loss_instance_rank'], parameter_list)])
    assert torch.isfinite(q).all() and torch.isfinite(rank).all()
    logits = flat([x.detach() for x in cls])
    scores = logits.sigmoid()
    assigned = captured
    ids = assigned['assigned_gt_indices']
    b, p = (ids >= 0).nonzero(as_tuple=True)
    labels = assigned['assigned_labels'][b, p]
    ious = assigned['assign_metrics'][b, p].float()
    probabilities = scores[b, p, labels]
    gq, gr = q[b, p, labels], rank[b, p, labels]
    gi = gt_instances_preprocess(rows, 36)
    pooled, targets, keys, counts = head.instance_rank.pool_logits(logits, ids, gi[:, :, :1])
    valid = (rows[:, 4:6] > rows[:, 2:4]).all(-1)
    priors = head.flatten_priors_train[p, :2]
    boxes = assigned['assigned_bboxes'][b, p]
    own_inside = ((priors > boxes[:, :2]) & (priors < boxes[:, 2:])).all(-1)
    record = dict(batch=index, main_images=36, gt=int(valid.sum()), matched_gt=len(targets),
        positive_priors=len(p), loss_qfl=float(losses['loss_cls']), loss_rank=float(losses['loss_instance_rank']),
        loss_semantic=float(semantic), pooled_class_accuracy=float((pooled.argmax(1)==targets).float().mean()),
        positive_class_accuracy=float((logits[b,p].argmax(1)==labels).float().mean()),
        classifier_weight_gradient_cosine=cosine(qparam,rparam),
        classifier_weight_rank_to_qfl_norm=float(rparam.norm()/qparam.norm().clamp_min(1e-30)),
        positive_trueclass_gradient_cosine=cosine(gq,gr),
        positive_trueclass_rank_to_qfl_norm=float(gr.norm()/gq.norm().clamp_min(1e-30)),
        mean_priors_per_matched_gt=float(counts.float().mean()),
        assigned_outside_own_gt=int((~own_inside).sum()),
        nonfinite_gradients=0)
    for name, values in dict(iou=ious, score=probabilities, gq=gq, gr=gr,
                              correctly_classified=logits[b,p].argmax(1)==labels).items():
        raw_vectors[name].append(values.detach().cpu())
    records.append(record)
    print('BATCH', json.dumps(record),flush=True)
    del losses, semantic, cls, reg, features, pyramid, batch, q, rank, qparam, rparam
    captured.clear()
handle.remove()
vectors = {k:torch.cat(v) for k,v in raw_vectors.items()}


def summarize(mask):
    q,r = vectors['gq'][mask],vectors['gr'][mask]
    iou,score = vectors['iou'][mask],vectors['score'][mask]
    conflict = (q > 0) & (r < 0)
    return dict(n=int(mask.sum()),mean_iou=float(iou.mean()),mean_score=float(score.mean()),
        correctly_classified_fraction=float(vectors['correctly_classified'][mask].float().mean()),
        qfl_requests_lower_score_fraction=float((q>0).float().mean()),
        qfl_rank_opposite_direction_fraction=float(conflict.float().mean()),
        rank_overturns_qfl_lowering_fraction=float((conflict & (q+r < 0)).float().mean()),
        mean_absolute_qfl_gradient=float(q.abs().mean()),mean_absolute_rank_gradient=float(r.abs().mean()),
        rank_to_qfl_trueclass_gradient_norm=float(r.norm()/q.norm().clamp_min(1e-30)),
        score_exceeds_iou_005_fraction=float((score>iou+.05).float().mean()))


summary = dict(all=summarize(torch.ones_like(vectors['iou'],dtype=torch.bool)))
for low,high in ((0,.1),(.1,.3),(.3,.5),(.5,.75),(.75,1.01)):
    mask=(vectors['iou']>=low)&(vectors['iou']<high)
    if mask.any():summary[f'iou_{low}_{high}']=summarize(mask)
result = dict(integrity=integrity,sample='108 real train images after original strong augmentation, B36; one discarded cache warmup batch',
    checkpoint_weight='raw training weights from ema_state_dict.module.*, not EMA evaluation weights',
    modifies_running_training=False,optimizer_updates=0,batches=records,gradients=summary,
    limitation='Local gradients are not an AP counterfactual or a proof of the whole-run cause.',
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
(OUT/'gradient_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',json.dumps(summary),flush=True)
