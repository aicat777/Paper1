from copy import deepcopy
import json,logging,random,time
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
root=Path('/home/pwt/Paper1'); out=root/'work_dirs/diagnostics/roi_region_point_analysis'
torch.set_num_threads(2); register_all_modules(); logging.disable(logging.CRITICAL)
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
assert head.head_module.pred_kernel_size==1
capture={}; handle=head.assigner.register_forward_hook(lambda mod,args,result:capture.update(result))
records=[]; counts=[]; start=time.monotonic()
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
  lookup=torch.full(valid.shape,-1,dtype=torch.long); lookup[rb,rg]=torch.arange(len(rows))
  assigned=capture['assigned_gt_indices']; b,p=(assigned>=0).nonzero(as_tuple=True)
  g=assigned[b,p]; ri=lookup[b,g]; keep=ri>=0
  b,p,g,ri=b[keep],p[keep],g[keep],ri[keep]
  flat=torch.cat([x.permute(0,2,3,1).reshape(36,-1,96) for x in features],1)
  dense=torch.cat([x.permute(0,2,3,1).reshape(36,-1,20) for x in cls],1)
  y=padded[b,g,0].long(); quality=capture['assign_metrics'][b,p]
  region=roi_logits[ri]; detector=dense[b,p]
  choose=(quality>=.5)&(region.argmax(-1)==y)&(detector.argmax(-1)!=y)
  bounds=torch.tensor(np.cumsum([x.shape[-2]*x.shape[-1] for x in features])[:-1])
  levels=torch.bucketize(p[choose],bounds,right=True)
  f=flat[b[choose],p[choose]].detach()
  target=region[choose].softmax(-1).detach()
  wa=head.instance_semantic.classifier.weight.detach()
  ba=head.instance_semantic.classifier.bias.detach()
  detector_logits=detector[choose].detach(); labels=y[choose]; q=quality[choose]
  weights=torch.stack([m.weight.detach()[:,:,0,0] for m in head.head_module.rtm_cls])[levels]
  biases=torch.stack([m.bias.detach() for m in head.head_module.rtm_cls])[levels]
  torch.testing.assert_close(torch.bmm(weights,f[:,:,None]).squeeze(-1)+biases,detector_logits,rtol=2e-5,atol=2e-5)
  with torch.enable_grad():
   point=f.clone().requires_grad_()
   local=F.linear(point,wa,ba)
   kd=F.kl_div(F.log_softmax(local,dim=-1),target,reduction='none').sum(-1)
   grad_kd=torch.autograd.grad(kd.sum(),point)[0]
   scores=detector_logits.clone().requires_grad_()
   qloss=head.loss_cls(scores,(labels,q),reduction_override='none')
   assert qloss.shape==(len(f),)
   gl=torch.autograd.grad(qloss.sum(),scores)[0]
  grad_qfl=torch.bmm(gl[:,None,:],weights).squeeze(1)
  error_class=detector_logits.argmax(-1)
  margin_dir=weights[torch.arange(len(f)),labels]-weights[torch.arange(len(f)),error_class]
  margin_change=-(margin_dir*grad_kd).sum(-1)
  qfl_change=-(grad_qfl*grad_kd).sum(-1)
  cos=(grad_qfl*grad_kd).sum(-1)/(grad_qfl.norm(dim=-1)*grad_kd.norm(dim=-1)).clamp_min(1e-30)
  for j in range(len(f)):
   records.append(dict(batch=index,gt_row=int(ri[choose][j]),prior=int(p[choose][j]),level=int(levels[j]),iou=float(q[j]),roi_correct=True,detector_correct=False,aux_point_correct=bool(local.detach()[j].argmax()==labels[j]),kd=float(kd.detach()[j]),cosine_with_qfl=float(cos[j]),class_margin_first_order_change=float(margin_change[j]),qfl_first_order_change=float(qfl_change[j])))
  counts.append(dict(batch=index,matched_points=len(p),selected=len(f),margin_improved=int((margin_change>0).sum()),qfl_decreased=int((qfl_change<0).sum())))
  print(json.dumps(counts[-1]),flush=True)
  del batch,pyramid,features,flat,cls,reg,dense
  capture.clear()
handle.remove()
from statistics import mean,median
n=len(records)
summary=dict(n=n,class_margin_improved=sum(x['class_margin_first_order_change']>0 for x in records),class_margin_worsened=sum(x['class_margin_first_order_change']<0 for x in records),qfl_decreased=sum(x['qfl_first_order_change']<0 for x in records),qfl_increased=sum(x['qfl_first_order_change']>0 for x in records),mean_cosine_with_qfl=mean(x['cosine_with_qfl'] for x in records),median_cosine_with_qfl=median(x['cosine_with_qfl'] for x in records))
result=dict(scope='Read-only feature-gradient audit on the same frozen raw epoch300 ROI model and 108 augmented train images. No feature/parameter optimizer step, no CUDA, no checkpoint writes.',selection='IoU>=0.5, ROI argmax correct, actual detector argmax incorrect',transfer='KL(ROI softmax || point auxiliary softmax), T=1; ROI target and auxiliary classifier parameters detached',derivatives='Exact gradient to the selected 96-dimensional classification feature; dot products against actual detector class-margin and QFL feature gradients. Positive class-margin change and negative QFL change are favorable under an infinitesimal feature step.',summary=summary,batches=counts,points=records,elapsed_seconds=time.monotonic()-start,limitations=['This tests the proposed initial gradient on a converged model, not actual full training or detection AP.','Direction at one feature does not establish the combined parameter update direction across shared convolutions or images.','Filtered augmented training sample and GT-matched locations; no background/duplicate/missed-detection AP analysis.'])
(out/'transfer_gradient_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='points'},ensure_ascii=False),flush=True)
