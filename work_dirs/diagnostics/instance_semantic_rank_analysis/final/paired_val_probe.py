"""Paired frozen EMA diagnostics on the same held-out val images; no updates.

Coverage below ignores duplicates/false positives and is NOT detection AP.
GT-restricted best scores are diagnostic oracle queries, not deployable output.
"""
from collections import defaultdict
from copy import deepcopy
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.structures.bbox import bbox_overlaps
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
register_all_modules()
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
free, total = torch.cuda.mem_get_info()
assert free > total * .30 + 2 * 2**30
torch.cuda.set_per_process_memory_fraction(.30)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
indices = sorted(np.random.default_rng(431).choice(5863, 512, replace=False).tolist())
methods = dict(roi=('instance_semantic', 'instance_semantic_640_bs36.py'),
               rank=('instance_semantic_rank', 'instance_semantic_rank_640_bs36.py'))
report = dict(sample='512 fixed random official val images, no training augmentation',
              dataset_indices=indices, weights='epoch_300.pth/state_dict: EMA used for official val',
              input=640, precision='FP32', optimizer_updates=0, batch=8, models={},
              limitations=['GT coverage is not one-to-one recall or AP; it ignores false positives.',
                'Raw geometry has no score/class filtering, and can be covered by unrelated proposals.',
                'Inside-GT diagnostics use GT to restrict candidate priors; they are not deployable predictions.'])


def flat(maps):
    return torch.cat([x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]) for x in maps], 1)


for name, (method, filename) in methods.items():
    started = time.perf_counter()
    directory = ROOT / ('work_dirs/' + method + '_rtmdet_tiny_dior_640_gpu1_bs36')
    cfg = Config.fromfile(str(directory / filename))
    saved = torch.load(directory / 'epoch_300.pth', map_location='cpu', weights_only=False)
    model = MODELS.build(deepcopy(cfg.model))
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda().eval()
    del saved
    loader_cfg = deepcopy(cfg.val_dataloader)
    loader_cfg.update(batch_size=8, num_workers=2, persistent_workers=False)
    loader_cfg.dataset.indices = indices
    loader_cfg.sampler = dict(type='DefaultSampler', shuffle=False)
    loader = Runner.build_dataloader(loader_cfg, seed=431)
    head = model.bbox_head
    rows, semantic_correct, semantic_ce, semantic_n = [], 0, 0., 0
    with torch.no_grad():
        for iteration, original_batch in enumerate(loader):
            batch = model.data_preprocessor(original_batch, training=False)
            samples = batch['data_samples']
            metas = [sample.metainfo for sample in samples]
            feats = model.extract_feat(batch['inputs'])
            cls, reg, cls_feats = head.head_module(feats, return_cls_features=True)
            logits, distances = flat(cls), flat(reg)
            priors = torch.cat(head.prior_generator.grid_priors(
                [x.shape[-2:] for x in cls], with_stride=True, device='cuda'))
            boxes = head.bbox_coder.decode(priors[None, :, :2], distances, priors[:, 2])
            post = head.predict_by_feat(cls, reg, batch_img_metas=metas, rescale=True)
            scores = logits.sigmoid()
            roi_rows = []
            for b, sample in enumerate(samples):
                gt = sample.gt_instances
                gb, labels = gt.bboxes, gt.labels
                if hasattr(gb, 'tensor'):
                    gb = gb.tensor
                if not len(gb):
                    continue
                meta = metas[b]
                sf = gb.new_tensor(meta['scale_factor']).repeat(2)
                pad = meta.get('pad_param', [0, 0, 0, 0])
                border = gb.new_tensor([pad[2], pad[0], pad[2], pad[0]])
                original_boxes = (boxes[b] - border) / sf
                original_boxes[:, 0::2].clamp_(0, meta['ori_shape'][1])
                original_boxes[:, 1::2].clamp_(0, meta['ori_shape'][0])
                prior_xy = (priors[:, :2] - border[:2]) / sf[:2]
                ious = bbox_overlaps(original_boxes, gb)
                raw_best = ious.amax(0)
                post_ious = bbox_overlaps(post[b].bboxes, gb)
                class_mask = post[b].labels[:, None] == labels[None]
                post_best = (post_ious * class_mask).amax(0) if len(post[b]) else gb.new_zeros(len(gb))
                inside = ((prior_xy[:, None] > gb[None, :, :2]) &
                          (prior_xy[:, None] < gb[None, :, 2:])).all(-1)
                input_boxes = gb * sf + border
                roi_rows.append(torch.cat((gb.new_full((len(gb), 1), b), labels[:, None].to(gb), input_boxes), 1))
                for j in range(len(gb)):
                    mask = inside[:, j]
                    info = dict(image_id=int(meta['img_id']), label=int(labels[j]),
                        area=float((gb[j, 2:] - gb[j, :2]).prod()),
                        raw_best_iou=float(raw_best[j]), post_correct_class_best_iou=float(post_best[j]),
                        inside_priors=int(mask.sum()))
                    if mask.any():
                        local_iou = ious[mask, j]
                        local_score = scores[b, mask, labels[j]]
                        k = local_score.argmax()
                        info.update(inside_best_iou=float(local_iou.max()),
                            inside_topscore_iou=float(local_iou[k]),
                            inside_topscore_class_correct=bool(logits[b, mask][k].argmax() == labels[j]),
                            inside_trueclass_maxscore=float(local_score[k]),
                            inside_iou50_exists=bool((local_iou >= .5).any()),
                            inside_iou75_exists=bool((local_iou >= .75).any()))
                    rows.append(info)
            if roi_rows:
                all_rois = torch.cat(roi_rows)
                rois = torch.cat((all_rois[:, :1], all_rois[:, 2:]), 1)
                roi_logits = head.instance_semantic.classifier(head.instance_semantic.pool_instances(cls_feats, rois))
                targets = all_rois[:, 1].long()
                semantic_correct += int((roi_logits.argmax(1) == targets).sum())
                semantic_ce += float(torch.nn.functional.cross_entropy(roi_logits, targets, reduction='sum'))
                semantic_n += len(targets)
            if iteration % 16 == 0:
                print(name, iteration, len(rows), flush=True)
    stats = {}
    groups = dict(all=rows, small=[x for x in rows if x['area'] < 32**2],
                  medium=[x for x in rows if 32**2 <= x['area'] < 96**2],
                  large=[x for x in rows if x['area'] >= 96**2])
    for group, values in groups.items():
        if not values:
            continue
        def mean(key, subset=values):
            return float(np.mean([x[key] for x in subset])) if subset else None
        local = [x for x in values if x['inside_priors']]
        stats[group] = dict(n=len(values), raw_mean_best_iou=mean('raw_best_iou'),
            raw_geometry_iou50_coverage=float(np.mean([x['raw_best_iou'] >= .5 for x in values])),
            raw_geometry_iou75_coverage=float(np.mean([x['raw_best_iou'] >= .75 for x in values])),
            post_correct_class_iou50_coverage=float(np.mean([x['post_correct_class_best_iou'] >= .5 for x in values])),
            post_correct_class_iou75_coverage=float(np.mean([x['post_correct_class_best_iou'] >= .75 for x in values])),
            inside_gt_count=len(local),inside_mean_best_iou=mean('inside_best_iou', local),
            inside_mean_topscore_iou=mean('inside_topscore_iou', local),
            inside_mean_topscore_class_accuracy=mean('inside_topscore_class_correct', local),
            inside_topscore_iou75_when_available=float(np.mean([x['inside_topscore_iou'] >= .75
                for x in local if x['inside_iou75_exists']])))
    report['models'][name] = dict(checkpoint=str(directory / 'epoch_300.pth'),
        summary=stats, roi_semantic_accuracy=semantic_correct / semantic_n,
        roi_semantic_ce=semantic_ce / semantic_n, n_semantic=semantic_n, objects=rows,
        elapsed_seconds=time.perf_counter() - started)
    print(name, json.dumps(stats), flush=True)
    (OUT/'paired_val.json').write_text(json.dumps(report, indent=2)+'\n')
    del model, head, loader
    torch.cuda.empty_cache()
