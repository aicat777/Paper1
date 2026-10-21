"""Read-only trained-router analysis on 64 deterministic real DIOR val images."""
from pathlib import Path
import json
import logging
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from custom_models.fdw.operator import FDWConv2d

ROOT=Path('/home/pwt/Paper1')
register_all_modules()
logging.disable(logging.CRITICAL)
torch.set_num_threads(4)
torch.manual_seed(42)
cfg=Config.fromfile(str(ROOT/'configs/fdw/rtmdet_tiny_dior_640.py'))
cfg.val_dataloader.update(batch_size=4,num_workers=0,persistent_workers=False,pin_memory=False)
# Spread across the fixed official list rather than only consecutive images.
cfg.val_dataloader.dataset.indices=torch.linspace(0,5862,64).round().int().tolist()
loader=Runner.build_dataloader(cfg.val_dataloader,seed=42)
checkpoint=torch.load(ROOT/'work_dirs/fdw_rtmdet_tiny_dior_640_gpu1_bs36/epoch_300.pth',map_location='cpu')
model=MODELS.build(cfg.model)
model.load_state_dict(checkpoint['state_dict'],strict=True)
model.eval()
observed={}
handles=[]
for name,module in model.named_modules():
    if isinstance(module,FDWConv2d):
        observed[name]=[]
        handles.append(module.router.register_forward_hook(
            lambda m,a,out,name=name: observed[name].append(out.detach().cpu())))
with torch.inference_mode():
    for i,data in enumerate(loader):
        processed=model.data_preprocessor(data,training=False)
        model.backbone(processed['inputs'])
        if i%4==0: print('batches',i+1,flush=True)
for handle in handles: handle.remove()
report=dict(status='passed',checkpoint='epoch_300.pth/state_dict (EMA)',images=64,
            image_indices=cfg.val_dataloader.dataset.indices,training_changed=False,layers={})
for name,values in observed.items():
    module=model.get_submodule(name)
    attention=torch.cat(values)
    # Effective gate multiplier after the alpha compensation: init approximately1.
    multiplier=attention*module.kernel_num
    centered=attention-attention.mean(dim=0,keepdim=True)
    per_sample_centered=attention-attention.mean(dim=1,keepdim=True)
    mean_attention=attention.mean(0,keepdim=True)
    with torch.no_grad():
        adaptive=module.aggregate_weight(attention)
        static=module.aggregate_weight(mean_attention)
        delta=adaptive-static
        relative_per_sample=delta.flatten(1).norm(dim=1)/static.flatten(1).norm(dim=1)
        reference=module.aggregate_weight(torch.ones_like(mean_attention)/module.kernel_num)
    optimizer_names=checkpoint['meta'].keys()
    entry=dict(channels=module.in_channels,kernel_num=module.kernel_num,
        effective_gate_mean=float(multiplier.mean()),effective_gate_min=float(multiplier.min()),
        effective_gate_max=float(multiplier.max()),
        effective_gate_mean_across_image_std=float(multiplier.std(dim=0).mean()),
        effective_gate_mean_within_image_std=float(multiplier.std(dim=1).mean()),
        effective_gate_mean_by_group=multiplier.mean(0).tolist(),
        effective_gate_std_across_images_by_group=multiplier.std(0).tolist(),
        kernel_relative_change_vs_mean_mean=float(relative_per_sample.mean()),
        kernel_relative_change_vs_mean_max=float(relative_per_sample.max()),
        mean_kernel_relative_change_vs_unit_gate=float((static-reference).norm()/reference.norm()),
        router_kernel_fc_weight_rms=float(module.router.kernel_fc.weight.square().mean().sqrt()),
        gates=multiplier.tolist())
    report['layers'][name]=entry
    print(name,json.dumps({k:v for k,v in entry.items() if k not in ('gates','effective_gate_mean_by_group','effective_gate_std_across_images_by_group')}),flush=True)
path=ROOT/'work_dirs/diagnostics/fdw_final_mechanism.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print('report',path,flush=True)
