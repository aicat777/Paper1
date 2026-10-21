"""Audit historical baseline GT grouping against exact official DSL indices."""
from copy import deepcopy
import json
import logging
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from mmyolo.models.utils import gt_instances_preprocess
from custom_models.instance_semantic_rank.assigner import IndexedDynamicSoftLabelAssigner

root=Path('/home/pwt/Paper1')
out=Path(__file__).resolve().parent
old=json.loads((root/'work_dirs/diagnostics/baseline_val_mechanism_20261015.json').read_text())
cases=[r for r in old['objects'] if r['n']>13]
annotation=json.loads((root/'datasets/DIOR/coco_annotations/instances_val.json').read_text())
ids=[r['id'] for r in annotation['images']]
selected=sorted(set(r['image_id'] for r in cases))
register_all_modules();logging.disable(logging.CRITICAL);torch.set_num_threads(2)
free,total=torch.cuda.mem_get_info();assert free>total*.2+2*2**30
torch.cuda.set_per_process_memory_fraction(.2)
cfg=Config.fromfile(str(root/'work_dirs/rtmdet_tiny_dior_640_gpu1_bs36/baseline_640_bs36.py'))
model=MODELS.build(deepcopy(cfg.model))
saved=torch.load(root/'work_dirs/rtmdet_tiny_dior_640_gpu1_bs36/epoch_300.pth',map_location='cpu',weights_only=False)
model.load_state_dict(saved['state_dict'],strict=True);del saved
model.cuda().eval()
loader_cfg=deepcopy(cfg.val_dataloader)
loader_cfg.update(batch_size=8,num_workers=0,persistent_workers=False)
loader_cfg.dataset.indices=[ids.index(i) for i in selected]
loader=Runner.build_dataloader(loader_cfg,seed=42)
acfg=deepcopy(cfg.model.train_cfg.assigner);acfg.pop('type')
indexed=IndexedDynamicSoftLabelAssigner(**acfg)
rows=[]
with torch.no_grad():
 for data in loader:
  data=model.data_preprocessor(data,training=False)
  cls,reg=model.bbox_head(model.extract_feat(data['inputs']))
  flat=lambda maps:torch.cat([x.permute(0,2,3,1).reshape(x.shape[0],-1,x.shape[1]) for x in maps],1)
  logits,distances=flat(cls),flat(reg)
  priors=torch.cat(model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls],with_stride=True,device='cuda'))
  predicted=model.bbox_head.bbox_coder.decode(priors[None,:,:2],distances,priors[:,2])
  targets=[]
  for b,sample in enumerate(data['data_samples']):
   box=sample.gt_instances.bboxes
   if hasattr(box,'tensor'):box=box.tensor
   sf=box.new_tensor(sample.metainfo['scale_factor']).repeat(2)
   p=sample.metainfo.get('pad_param',[0,0,0,0]);border=box.new_tensor([p[2],p[0],p[2],p[0]])
   targets.append(torch.cat((box.new_full((len(box),1),b),sample.gt_instances.labels[:,None].to(box),box*sf+border),1))
  gt=gt_instances_preprocess(torch.cat(targets),len(targets))
  args=(predicted,logits,priors,gt[:,:,:1],gt[:,:,1:],(gt[:,:,1:].sum(-1,keepdim=True)>0).float())
  official=model.bbox_head.assigner(*args);actual=indexed(*args)
  for key in official:torch.testing.assert_close(official[key],actual[key],atol=0,rtol=0)
  for b,sample in enumerate(data['data_samples']):
   boxes=sample.gt_instances.bboxes
   if hasattr(boxes,'tensor'):boxes=boxes.tensor
   labels=sample.gt_instances.labels
   areas=(boxes[:,2:]-boxes[:,:2]).prod(1)
   sides=(boxes[:,2:]-boxes[:,:2]).amin(1)*sample.metainfo['scale_factor'][0]
   for case in [r for r in cases if r['image_id']==sample.metainfo['img_id']]:
    candidates=((labels==case['label'])&((areas-case['area_original']).abs()<.1)&((sides-case['min_side_input']).abs()<.1)).nonzero().flatten()
    result=dict(historical=case,matching_gt_indices=candidates.tolist(),official_topk=indexed.topk)
    if len(candidates)==1:
     g=int(candidates[0]);result['exact_index_count']=int((actual['assigned_gt_indices'][b]==g).sum())
     # Historical statistics may have associated overlapping assigned boxes
     # with this GT; report reference distributions without assuming the bug.
     result['all_gt_counts_this_image']=[int((actual['assigned_gt_indices'][b]==i).sum()) for i in range(len(boxes))]
    rows.append(result)
report=dict(status='passed',scope='8 historical rows whose reported n exceeded configured topk13; same EMA checkpoint and val images',
 official_assignment_outputs_exact=True,optimizer_updates=0,rows=rows,
 historical_matches_reproduced=sum(r.get('exact_index_count')==r['historical']['n'] for r in rows),
 limitation='Historical generator source is absent from the diagnostic directory; discrepancy alone does not identify its implementation error.')
(out/'historical_assignment_statistics_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print([(r['historical']['image_id'],r['historical']['n'],r.get('exact_index_count')) for r in rows])
