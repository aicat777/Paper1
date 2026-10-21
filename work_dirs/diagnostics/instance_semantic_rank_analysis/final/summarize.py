"""Summarize final recorded results and frozen diagnostics; no training."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config

ROOT=Path('/home/pwt/Paper1')
OUT=Path(__file__).resolve().parent
runs=dict(baseline='rtmdet_tiny_dior_640_gpu1_bs36',
          roi_semantic='instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36',
          roi_plus_rank='instance_semantic_rank_rtmdet_tiny_dior_640_gpu1_bs36')
keys=['pascal_voc/mAP','coco/bbox_mAP','coco/bbox_mAP_50','coco/bbox_mAP_75',
      'coco/bbox_mAP_s','coco/bbox_mAP_m','coco/bbox_mAP_l']
curves, train = {}, {}
for method, directory in runs.items():
    val, losses = {}, {}
    for path in sorted((ROOT/'work_dirs'/directory).glob('*/vis_data/scalars.json')):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if 'pascal_voc/mAP' in row:
                val[row['step']]=row
            elif 'loss_cls' in row:
                losses[row['step']]=row
    assert max(val)==300
    curves[method]=val
    train[method]=list(losses.values())

rank, roi = curves['roi_plus_rank'], curves['roi_semantic']
grad=json.loads((OUT/'gradient_audit.json').read_text())
paired=json.loads((OUT/'paired_val.json').read_text())
assert grad['integrity']['epoch']==300 and len(paired['models'])==2
same_cfg_a=Config.fromfile(str(ROOT/'work_dirs/instance_semantic_queue/instance_semantic_640_bs36.py'))
same_cfg_b=Config.fromfile(str(ROOT/'work_dirs/instance_semantic_rank_queue/instance_semantic_rank_640_bs36.py'))
protocol_keys=['train_dataloader','val_dataloader','val_evaluator','train_cfg','custom_hooks',
               'optim_wrapper','param_scheduler','randomness','default_hooks']
protocol={key:same_cfg_a[key]==same_cfg_b[key] for key in protocol_keys}
assert all(protocol.values())
manifest=json.loads((ROOT/'work_dirs/instance_semantic_rank_queue/manifest.json').read_text())
changed=[path for path,digest in manifest['source_sha256'].items()
         if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest]
assert not changed
class_delta={key:(rank[300][key]-roi[300][key])*100 for key in rank[300] if key.endswith('_precision')}
summary=dict(epoch=300,metric_weights='EMA, official full val: 5863 images',
    final_metrics_percent={name:{key:row[300][key]*100 for key in keys} for name,row in curves.items()},
    delta_rank_minus_roi_pp={key:(rank[300][key]-roi[300][key])*100 for key in keys},
    class_coco_ap_changes_rank_minus_roi_pp=class_delta,
    lower_coco_classes=sum(value<0 for value in class_delta.values()),
    voc_lower_epochs=[e for e in rank if rank[e]['pascal_voc/mAP']<roi[e]['pascal_voc/mAP']],
    ap75_lower_epochs=[e for e in rank if rank[e]['coco/bbox_mAP_75']<roi[e]['coco/bbox_mAP_75']],
    same_training_and_evaluation_protocol=protocol,source_changes_since_validation=changed,
    checkpoint_integrity=grad['integrity'],
    gradient_sample=dict(images=108,augmentation='original strong augmentation',microbatch=36,
        checkpoint='raw epoch300 training weights',total_gt=sum(b['gt'] for b in grad['batches']),
        matched_gt=sum(b['matched_gt'] for b in grad['batches']),
        positive_priors=grad['gradients']['all']['n'],summary=grad['gradients']),
    paired_ema_val=dict(images=512,objects=paired['models']['roi']['summary']['all']['n'],
        sample_is_not_full_detection_ap=True,models={name:{'summary':data['summary'],
            'roi_semantic_accuracy':data['roi_semantic_accuracy'],
            'roi_semantic_ce':data['roi_semantic_ce']} for name,data in paired['models'].items()}),
    mechanism=dict(
        original='All positive-area augmented GTs -> ROI feature -> separate auxiliary classifier -> CE.',
        added='Existing DSL positive priors -> mean actual inference logits per matched GT -> softmax CE.',
        gradient_formula='dL_rank/dz_i,c = 0.1/(G*n_g) * (softmax(mean_z_g)_c - 1[c=y_g])',
        exact_properties=['All priors of one GT receive the same category gradient, independent of their IoU.',
            'Permuting logits between good and poor boxes of the same GT leaves this loss unchanged.',
            'Common-logit-shift invariance does not imply unchanged true-class sigmoid gradients.',
            'Equal GT weighting plus division by n_g increases per-prior emphasis on few-positive GTs.',
            'Unmatched GTs have no new ranking loss; original ROI supervision is retained.']),
    interpretation='The implemented signal works but ignores quality in the actual quality-aware classifier. Local low-IoU conflicts, classification-only pooling, and shared-feature changes are supported; their exact contributions to full AP loss are not identified.',
    limitations=['One full run per variant: no seed-variance estimate.',
        'Gradient audit uses 108 strongly augmented train images, not the full training trajectory.',
        '512-image coverage/classification probes use GT and ignore false positives/duplicates; they are not AP.',
        'Geometry and score selection both differ; AP75 loss cannot be attributed entirely to calibration.'],
    reports=dict(gradient=str(OUT/'gradient_audit.json'),paired_val=str(OUT/'paired_val.json'),
        curves=str(OUT/'curves.json'),plot=str(OUT/'final_comparison.png')))
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
colors=['#667085','#1677b8','#d45042']
labels=['Baseline','ROI semantics','ROI + category rank']
fig,axes=plt.subplots(1,3,figsize=(13.8,3.5),layout='constrained')
for ax,key,title in zip(axes[:2],['pascal_voc/mAP','coco/bbox_mAP_75'],['VOC mAP@0.5','COCO AP75']):
    for (method,points),color,label in zip(curves.items(),colors,labels):
        es=[e for e in sorted(points) if e>=100]
        ax.plot(es,[points[e][key]*100 for e in es],label=label,color=color,lw=2)
    ax.set(xlabel='Epoch',ylabel='AP (%)',title=title)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8,frameon=False)
# Compare to the successful ROI method, avoiding an unrelated efficiency claim.
metrics=['AP','AP50','AP75','Small','Medium','Large']
values=[summary['delta_rank_minus_roi_pp'][key] for key in keys[1:]]
axes[2].bar(metrics,values,color=['#1677b8' if v>0 else '#d45042' for v in values])
axes[2].axhline(0,color='#444',lw=.8)
axes[2].set(title='Rank minus ROI semantics (epoch 300)',ylabel='Change (percentage points)')
axes[2].tick_params(axis='x',labelsize=8)
for x,v in enumerate(values):
    axes[2].text(x,v+(.06 if v>=0 else -.06),f'{v:+.1f}',ha='center',
                 va='bottom' if v>=0 else 'top',fontsize=9)
axes[2].set_ylim(-2.1,1.9)
fig.savefig(OUT/'final_comparison.png',dpi=160)
print(json.dumps({k:summary[k] for k in ['epoch','final_metrics_percent','lower_coco_classes']},indent=2))
