from copy import deepcopy
import json, logging, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from mmyolo.models.utils import gt_instances_preprocess
from custom_models.roi_supervision_common.assigner import ROISemanticIndexedAssigner
root=Path('/home/pwt/Paper1')
out=root/'work_dirs/diagnostics/roi_region_point_analysis'
torch.set_num_threads(2)
register_all_modules(); logging.disable(logging.CRITICAL)
random.seed(431); np.random.seed(431); torch.manual_seed(431)
directory=root/'work_dirs/instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36'
cfg=Config.fromfile(str(directory/'instance_semantic_640_bs36.py'))
cfg.model.train_cfg.assigner.type='ROISemanticIndexedAssigner'
saved=torch.load(directory/'epoch_300.pth',map_location='cpu',weights_only=False)
raw={k.removeprefix('module.'):v for k,v in saved['ema_state_dict'].items() if k.startswith('module.')}
assert raw and set(raw)==set(saved['state_dict'])
model=MODELS.build(deepcopy(cfg.model)); model.load_state_dict(raw,strict=True); model.cpu().train()
del saved,raw
loader_cfg=deepcopy(cfg.train_dataloader)
loader_cfg.update(batch_size=36,num_workers=0,persistent_workers=False,sampler=dict(type='DefaultSampler',shuffle=False))
loader_cfg.dataset.indices=np.random.default_rng(431).choice(5862,144,replace=False).tolist()
loader=Runner.build_dataloader(loader_cfg,seed=431)
head=model.bbox_head
capture={}
handle=head.assigner.register_forward_hook(lambda mod,args,result:capture.update(result))
records=[]; gt_records=[]; started=time.monotonic()
with torch.no_grad():
 for index,batch in enumerate(loader):
  batch=model.data_preprocessor(batch,training=True)
  if index==0: continue
  pyramid=model.extract_feat(batch['inputs'])
  cls,reg,features=head.head_module(pyramid,return_cls_features=True)
  gt=batch['data_samples']['bboxes_labels']
  head.loss_by_feat(cls,reg,gt,batch['data_samples']['img_metas'])
  padded=gt_instances_preprocess(gt,36)
  valid=(padded[...,3:5]>padded[...,1:3]).all(-1)
  rb,rg=valid.nonzero(as_tuple=True)
  rows=torch.cat((rb[:,None].to(padded),padded[rb,rg]),1)
  rois=torch.cat((rows[:,:1],rows[:,2:6]),1)
  roi_logits=head.instance_semantic.classifier(head.instance_semantic.pool_instances(features,rois))
  lookup=torch.full(valid.shape,-1,dtype=torch.long)
  lookup[rb,rg]=torch.arange(len(rows))
  assigned=capture['assigned_gt_indices']
  b,p=(assigned>=0).nonzero(as_tuple=True)
  g=assigned[b,p]; ri=lookup[b,g]
  keep=ri>=0; b,p,g,ri=b[keep],p[keep],g[keep],ri[keep]
  flat=torch.cat([x.permute(0,2,3,1).reshape(36,-1,96) for x in features],1)
  dense_cls=torch.cat([x.permute(0,2,3,1).reshape(36,-1,20) for x in cls],1)
  local_logits=head.instance_semantic.classifier(flat[b,p])
  detector_logits=dense_cls[b,p]
  y=padded[b,g,0].long()
  roi_point_logits=roi_logits[ri]
  ce_roi=F.cross_entropy(roi_point_logits,y,reduction='none')
  ce_local=F.cross_entropy(local_logits,y,reduction='none')
  rc=roi_point_logits.argmax(-1)==y
  lc=local_logits.argmax(-1)==y
  dc=detector_logits.argmax(-1)==y
  ious=capture['assign_metrics'][b,p]
  for j in range(len(b)):
   records.append(dict(batch=index,gt_row=int(ri[j]),iou=float(ious[j]),roi_correct=bool(rc[j]),aux_point_correct=bool(lc[j]),detector_point_correct=bool(dc[j]),roi_ce=float(ce_roi[j]),aux_point_ce=float(ce_local[j])))
  for row in ri.unique():
   mask=ri==row
   gt_records.append(dict(batch=index,row=int(row),points=int(mask.sum()),roi_correct=bool(rc[mask][0]),aux_point_accuracy=float(lc[mask].float().mean()),detector_point_accuracy=float(dc[mask].float().mean()),roi_ce=float(ce_roi[mask][0]),aux_point_ce=float(ce_local[mask].mean())))
  print(json.dumps({'batch':index,'valid_gt':len(rows),'matched_gt':len(ri.unique()),'matched_points':len(b),'roi_accuracy_per_point':float(rc.float().mean()),'aux_point_accuracy':float(lc.float().mean()),'detector_point_accuracy':float(dc.float().mean()),'elapsed_seconds':time.monotonic()-started}),flush=True)
  del batch,pyramid,features,flat,cls,reg,roi_logits,roi_point_logits,local_logits,detector_logits,dense_cls
  capture.clear()
handle.remove()
def summarize(xs):
 n=len(xs)
 if not n: return {'n':0}
 return dict(n=n,roi_accuracy=sum(x['roi_correct'] for x in xs)/n,aux_point_accuracy=sum(x['aux_point_correct'] for x in xs)/n,detector_point_accuracy=sum(x['detector_point_correct'] for x in xs)/n,roi_ce=sum(x['roi_ce'] for x in xs)/n,aux_point_ce=sum(x['aux_point_ce'] for x in xs)/n,roi_correct_aux_wrong=sum(x['roi_correct'] and not x['aux_point_correct'] for x in xs),roi_wrong_aux_correct=sum(not x['roi_correct'] and x['aux_point_correct'] for x in xs),roi_correct_detector_wrong=sum(x['roi_correct'] and not x['detector_point_correct'] for x in xs),roi_wrong_detector_correct=sum(not x['roi_correct'] and x['detector_point_correct'] for x in xs))
by_gt={k:sum(x[k] for x in gt_records)/len(gt_records) for k in ['roi_correct','aux_point_accuracy','detector_point_accuracy','roi_ce','aux_point_ce']}
result=dict(scope='Read-only frozen raw epoch300 ROI model; 108 fixed strong-augmented train images, isolated CPU FP32, B36 batch BN, same protocol as existing ROI signal probe; no parameter updates or checkpoint writes',summary=summarize(records),gt_balanced_summary=dict(n=len(gt_records),**by_gt),by_iou={f'{lo}_{hi}':summarize([x for x in records if lo<=x['iou']<hi]) for lo,hi in [(0,.1),(.1,.3),(.3,.5),(.5,.7),(.7,1.01)]},limitations=['Auxiliary classifier was trained on region averages, not point vectors; its point-readout gap is not itself evidence that dense detection is deficient.','Point classification accuracy is not AP; locations were selected by GT matching.','One converged checkpoint on augmented training samples does not establish an online teacher advantage across training or generalization.'],gt_records=gt_records,points=records,elapsed_seconds=time.monotonic()-started)
(out/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',json.dumps({k:v for k,v in result.items() if k not in ['gt_records','points']},ensure_ascii=False),flush=True)
