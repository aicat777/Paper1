"""Validate the four-site SCDown + seven-site G4 + original all-GT ROI composition."""
import argparse
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from custom_models.instance_semantic.roi import InstanceSemanticLoss
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown_p3_g4_semantic'
SITES = ('backbone.stage2.0', 'backbone.stage3.0', 'backbone.stage4.0', 'neck.downsample_layers.1')

GROUP_SITES = ('backbone.stage2.1.blocks.0.conv1', 'backbone.stage3.1.blocks.0.conv1',
               'backbone.stage4.2.blocks.0.conv1', 'neck.top_down_layers.0.0.blocks.0.conv1',
               'neck.top_down_layers.1.blocks.0.conv1', 'neck.bottom_up_layers.0.blocks.0.conv1',
               'neck.bottom_up_layers.1.blocks.0.conv1')

def watched_parameters(model):
    prefixes = tuple(site + '.' for site in SITES + GROUP_SITES) + (
        'bbox_head.head_module.cls_convs.', 'bbox_head.head_module.rtm_cls.',
        'bbox_head.instance_semantic.')
    return {name:p for name,p in model.named_parameters() if name.startswith(prefixes)}


def preflight(cfg, output, path):
    scdown = Config.fromfile(str(ROOT/'configs/scdown_p3_g4/rtmdet_tiny_dior_640.py'))
    roi = Config.fromfile(str(ROOT/'configs/instance_semantic/rtmdet_tiny_dior_640.py'))
    for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks', 'train_pipeline',
                'train_pipeline_stage2', 'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == scdown[key] == roi[key], key
    expected = deepcopy(scdown.model); expected.bbox_head = deepcopy(roi.model.bbox_head)
    assert cfg.model == expected
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.img_scale == (640,640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts',1) == 1
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    models=[]
    for source in (scdown,cfg):
        torch.manual_seed(42)
        model=MODELS.build(deepcopy(source.model)); model.init_weights(); model.eval()
        models.append(model)
    reference,model=models
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0,msg=name)
    assert type(model.bbox_head.instance_semantic) is InstanceSemanticLoss
    assert tuple(n for n,m in model.named_modules() if isinstance(m,SCDown)) == SITES
    grouped = {n.removesuffix('.conv') for n,m in model.named_modules()
               if isinstance(m,torch.nn.Conv2d) and m.groups==4}
    assert grouped==set(GROUP_SITES)
    assert sum(p.numel() for p in model.parameters()) == 3105704
    assert len(watched_parameters(model)) == 67
    macs,handles=[0],[]

    def count(layer,inputs,result):
        per_output=(layer.in_channels//layer.groups)*math.prod(layer.kernel_size) \
            if isinstance(layer,torch.nn.Conv2d) else layer.in_features
        macs[0]+=result.numel()*per_output

    for layer in model.modules():
        if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        actual=model._forward(torch.zeros(1,3,640,640))
        expected_outputs=reference._forward(torch.zeros(1,3,640,640))
    for handle in handles: handle.remove()
    for first,second in zip(actual,expected_outputs):
        for a,b in zip(first,second): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert [list(v.shape) for v in actual[0]] == [[1,20,s,s] for s in (80,40,20)]
    assert [list(v.shape) for v in actual[1]] == [[1,4,s,s] for s in (80,40,20)]
    assert math.isclose(macs[0]/1e9,6.78524544,abs_tol=1e-10)
    for loader,expected_count in ((cfg.train_dataloader,5862),(cfg.val_dataloader,5863)):
        annotation=json.loads((Path(cfg.data_root)/loader.dataset.ann_file).read_text())
        assert len(annotation['images'])==expected_count
    report_to(output,dict(status='passed',method=METHOD,parameters=3105704,
        inference_parameters=3103764,inference_conv_linear_gmac=macs[0]/1e9,
        auxiliary_parameters=1940,all_detector_initial_parameters_match_scdown=True,
        inference_predictions_match_scdown_exactly=True,original_roi_module_unchanged=True,
        scdown_sites=list(SITES),grouped_sites=list(GROUP_SITES),groups=4,prediction_points=8400,watched_parameter_tensors=67,
        train_images=5862,val_images=5863,input_size=[640,640],formal_batch=36,accumulation=1,
        scratch=True,unchanged_training_protocol=True,semantic_weight=.1,all_valid_gt_denominator=True,
        formal_batch_capacity_tested=False,**config_identity(path)))


@HOOKS.register_module()
class SCDownP3G4SemanticAuditHook(CandidateAuditHook):
    def before_run(self,runner):
        require_memory(self.fraction)

    def before_train(self,runner):
        self.model=runner.model
        self.watched=watched_parameters(self.model)
        assert len(self.watched)==67
        self.initial={n:p.detach().cpu().clone() for n,p in self.watched.items()}
        parameters=[p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters)==len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name,p in self.watched.items():
            def capture(grad,name=name):
                assert torch.isfinite(grad).all(),name
                self.gradients[name]=max(self.gradients.get(name,0),float(grad.abs().max()))
            p.register_hook(capture)
        self.assignment_calls,self.head_calls,self.coverage=0,0,[]

        def assignment(*args): self.assignment_calls+=1

        def head_forward(*args):
            if self.model.training: self.head_calls+=1

        def semantic(module,inputs):
            features,rows=inputs
            valid=rows[(rows[:,4:6]>rows[:,2:4]).all(-1)]
            assert len(features)==3
            self.coverage.append(dict(raw_gt=len(rows),valid_gt=len(valid),classifier_gt=None))

        def classifier(module,inputs):
            assert len(inputs[0])==self.coverage[-1]['valid_gt']
            assert inputs[0].requires_grad
            self.coverage[-1]['classifier_gt']=len(inputs[0])

        head=self.model.bbox_head
        self.handles=[head.assigner.register_forward_hook(assignment),
                      head.head_module.register_forward_hook(head_forward),
                      head.instance_semantic.register_forward_pre_hook(semantic),
                      head.instance_semantic.classifier.register_forward_pre_hook(classifier)]
        self.started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self,runner,batch_idx,data_batch=None):
        self.bn_before={n:int(m.num_batches_tracked) for n,m in self.model.named_modules()
                        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        for n,m in self.model.named_modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                assert int(m.num_batches_tracked)==self.bn_before[n]+1,n

    def after_train(self,runner):
        super().after_train(runner)
        for handle in self.handles: handle.remove()
        assert self.assignment_calls==self.head_calls==len(self.coverage)==16
        assert all(row['classifier_gt']==row['valid_gt'] for row in self.coverage)
        report=json.loads(Path(self.report_path).read_text())
        report.update(assignment_calls=self.assignment_calls,training_head_calls=self.head_calls,
            original_all_gt_roi_coverage=self.coverage,one_bn_update_per_train_step=True,
            original_roi_module_unchanged=True,scdown_sites=list(SITES),grouped_sites=list(GROUP_SITES),groups=4)
        report_to(self.report_path,report)


def smoke(cfg,output,fraction):
    directory=Path(output).parent/'smoke'
    if directory.exists(): raise RuntimeError('Existing short-test output would be overwritten')
    directory.mkdir(parents=True)
    cfg.work_dir=str(directory)
    cfg.train_cfg.update(max_epochs=2,val_interval=1,dynamic_intervals=[])
    cfg.param_scheduler=[]; cfg.default_hooks.checkpoint=None; cfg.default_hooks.logger.interval=1
    cfg.optim_wrapper.loss_scale=dict(init_scale=32); cfg.custom_hooks[1].switch_epoch=1
    cfg.train_dataloader.update(batch_size=2,num_workers=0,persistent_workers=False)
    cfg.train_dataloader.dataset.indices=list(range(16))
    cfg.val_dataloader.update(batch_size=4,num_workers=0,persistent_workers=False)
    annotation=json.loads((Path(cfg.data_root)/cfg.val_dataloader.dataset.ann_file).read_text())
    categories={row['id'] for row in annotation['categories']}
    by_image={row['id']:set() for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd',0): by_image[row['image_id']].add(row['category_id'])
    selected=[]
    while categories:
        image=max(by_image,key=lambda key:len(by_image[key]&categories))
        assert by_image[image]&categories
        selected.append(image); categories-=by_image[image]
    ids=[row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected)==32: break
        if image not in selected: selected.append(image)
    cfg.val_dataloader.dataset.indices=[ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.scdown_p3_g4_semantic_probe')
    cfg.custom_hooks.append(dict(type='SCDownP3G4SemanticAuditHook',method=METHOD,
        report_path=str(output),image_ids=selected,allocator_fraction=fraction))
    snapshot=directory/(METHOD+'_smoke.py'); cfg.dump(str(snapshot))
    result=subprocess.run([sys.executable,'-u',str(ROOT/'mmyolo/tools/train.py'),str(snapshot)],
                          cwd=ROOT,stdin=subprocess.DEVNULL)
    if result.returncode: raise RuntimeError(f'Official short training failed: {result.returncode}')
    report=json.loads(Path(output).read_text())
    assert report['status']=='passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED',METHOD,flush=True)


def capacity(cfg,output,fraction,batch,path):
    free,total=require_memory(fraction)
    assert cfg.train_dataloader.batch_size==36 and cfg.optim_wrapper.get('accumulative_counts',1)==1
    torch.backends.cudnn.benchmark=cfg.env_cfg.cudnn_benchmark
    batch=batch or 36
    data,image_ids=capacity_data(cfg,batch)
    model=MODELS.build(deepcopy(cfg.model)); model.init_weights(); model.cuda().train()
    optim_cfg=deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale=dict(init_scale=32)
    optimizer=build_optim_wrapper(model,optim_cfg)
    watched=watched_parameters(model)
    initial={n:p.detach().cpu().clone() for n,p in watched.items()}
    parameters=[p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len(watched)==67 and len(parameters)==len({id(p) for p in parameters})
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients={}
    for name,p in watched.items():
        def capture(grad,name=name):
            assert torch.isfinite(grad).all(),name
            gradients[name]=max(gradients.get(name,0),float(grad.abs().max()))
        p.register_hook(capture)
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter(); losses=[]
    for _ in range(3):
        values=model.train_step(data,optimizer)
        assert all(torch.isfinite(v).all() for v in values.values())
        losses.append({k:float(v) for k,v in values.items()}); torch.cuda.synchronize()
    updates={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in watched.items()}
    assert all(gradients.get(n,0)>0 and updates[n]>0 for n in watched)
    assert all(int(optimizer.optimizer.state[p]['step'])==3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale()==32
    report_to(output,dict(status='passed',method=METHOD,microbatch=batch,accumulation=1,
        optimizer_updates=3,gt_per_image=1200,different_image_ids=image_ids,
        repeated_annotations=True,synthetic_multiscale_boxes=True,formal_batch=36,
        formal_batch_capacity_tested=batch==36,allocator_fraction=fraction,
        loss_scale_init=32,warmup_enabled=False,amp_skipped_updates=0,saved_checkpoints=False,
        watched_parameter_tensors=len(watched),gradient_max=gradients,parameter_updates_max=updates,
        losses=losses,cuda_device=torch.cuda.get_device_name(),cuda_total_gib=total/2**30,
        initial_free_gib=free/2**30,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,cudnn_benchmark=torch.backends.cudnn.benchmark,
        elapsed_seconds=time.perf_counter()-started,**config_identity(path)))
    print('CAPACITY_PASSED',batch,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('preflight','smoke','capacity'),required=True)
    parser.add_argument('--config',default=str(ROOT/'configs'/METHOD/'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report',required=True)
    parser.add_argument('--allocator-fraction',type=float,default=.28)
    parser.add_argument('--batch',type=int)
    args=parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg=Config.fromfile(args.config)
    try:
        if args.mode=='preflight': preflight(cfg,args.report,args.config)
        elif args.mode=='smoke': smoke(cfg,args.report,args.allocator_fraction)
        else: capacity(cfg,args.report,args.allocator_fraction,args.batch,args.config)
    except Exception as error:
        report_to(args.report,dict(status='failed',method=METHOD,mode=args.mode,
            error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__': main()
