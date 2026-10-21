"""Read-only LSConv checkpoint, initialization, and paired validation diagnosis."""
from copy import deepcopy
import json,logging,time
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT=Path('/home/pwt/Paper1')
OUT=ROOT/'work_dirs/diagnostics/lsconv_full_analysis'
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)
torch.manual_seed(42)
register_all_modules()
torch.cuda.set_per_process_memory_fraction(.30)
report={}
# Small deterministic paired subset: diagnostic differences, not formal results.
cfg=Config.fromfile(str(ROOT/'work_dirs/lsconv_full_queue/lsconv_full_640_bs12x3.py'))
annotation=json.loads((Path(cfg.data_root)/cfg.val_dataloader.dataset.ann_file).read_text())
by_image={row['id']:set() for row in annotation['images']}
for row in annotation['annotations']:
 if not row.get('iscrowd',0):by_image[row['image_id']].add(row['category_id'])
remaining={row['id'] for row in annotation['categories']};selected=[]
while remaining:
 image=max(by_image,key=lambda x:len(by_image[x]&remaining))
 selected.append(image);remaining-=by_image[image]
ids=[row['id'] for row in annotation['images']]
for idx in torch.randperm(len(ids),generator=torch.Generator().manual_seed(42)).tolist():
 if ids[idx] not in selected:selected.append(ids[idx])
 if len(selected)==256:break
loader_cfg=deepcopy(cfg.val_dataloader)
loader_cfg.update(batch_size=8,num_workers=0,persistent_workers=False)
loader_cfg.dataset.indices=[ids.index(i) for i in selected]
loader=Runner.build_dataloader(loader_cfg,seed=42)
report['scope']=dict(images=len(selected),image_ids=selected,paired_same_images=True,
                     input_size=640,precision='FP32',gpu='GPU0 shared',formal_result=False)
work=ROOT/'work_dirs/lsconv_full_rtmdet_tiny_dior_640_gpu1_bs12x3'
checkpoint=max(work.glob('epoch_*.pth'),key=lambda p:int(p.stem.split('_')[1]))
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
ema=saved['state_dict']
raw={k[len('module.'):]:v for k,v in saved['ema_state_dict'].items() if k.startswith('module.')}
assert set(raw)==set(ema)
report['checkpoint']=str(checkpoint);report['epoch']=saved['meta']['epoch']
report['optimizer']=dict(iterations=saved['meta']['iter'],
 steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),
 scaler=str(saved['optimizer'].get('loss_scaler')),
 ema_steps=int(saved['ema_state_dict']['steps']))
report['paired_metrics']={}
for label,state in [('ema',ema),('raw',raw)]:
 model=MODELS.build(deepcopy(cfg.model));model.load_state_dict(state,strict=True);model.cuda().eval()
 metric_cfg=deepcopy(cfg.val_evaluator)
 for m in metric_cfg:
  if m['type']=='mmdet.CocoMetric':m['classwise']=False
 evaluator=Evaluator(metric_cfg)
 evaluator.dataset_meta=loader.dataset.metainfo
 for metric in evaluator.metrics:
  if hasattr(metric,'img_ids'):metric.img_ids=selected
 start=time.perf_counter()
 with torch.inference_mode():
  for batch in loader:
   samples=model.test_step(batch)
   assert all(torch.isfinite(s.pred_instances.bboxes).all() and torch.isfinite(s.pred_instances.scores).all() for s in samples)
   evaluator.process(data_samples=samples,data_batch=batch)
 metrics=evaluator.evaluate(len(selected))
 metrics={k:float(v) for k,v in metrics.items()}
 report['paired_metrics'][label]=dict(metrics=metrics,seconds=time.perf_counter()-start)
 print(label,report['paired_metrics'][label],flush=True)
 del model,evaluator;torch.cuda.empty_cache()
# Inspect initial train/eval activation scales, without updating any saved model.
report['init_stats']={}
for label,path in [('baseline','configs/rtmdet_tiny_dior_300e.py'),('full','configs/lsconv_full/rtmdet_tiny_dior_640.py')]:
 current=Config.fromfile(str(ROOT/path));torch.manual_seed(42)
 model=MODELS.build(deepcopy(current.model));model.init_weights();model.cuda()
 stats={};handles=[]
 for name,module in model.named_modules():
  watch=name=='neck' or name in ['bbox_head.head_module.cls_convs.0.0','bbox_head.head_module.cls_convs.0.1',
   'bbox_head.head_module.reg_convs.0.0','bbox_head.head_module.reg_convs.0.1']
  if watch:
   def capture(module,inputs,output,name=name):
    values=output if isinstance(output,(list,tuple)) else [output]
    stats[name]=[dict(rms=float(v.float().square().mean().sqrt()),absmax=float(v.abs().max())) for v in values]
   handles.append(module.register_forward_hook(capture))
 image=torch.randn(2,3,256,256,device='cuda')
 report['init_stats'][label]={}
 for train in [True,False]:
  model.train(train)
  with torch.no_grad(): cls,reg=model._forward(image)
  report['init_stats'][label]['train' if train else 'eval_after_one_batch']=dict(layers=deepcopy(stats),reg_rms=[float(v.float().square().mean().sqrt()) for v in reg],cls_std=[float(v.float().std()) for v in cls])
 for handle in handles:handle.remove()
 del model;torch.cuda.empty_cache()
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('REPORT',str(OUT/'report.json'),flush=True)
