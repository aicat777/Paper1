"""Temporary architecture audit; never launches or changes a formal experiment."""
from copy import deepcopy
from datetime import datetime
import gc
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from mmengine.config import Config
from mmengine.hooks import EMAHook
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from mmdet.models.layers.csp_layer import CSPNeXtBlock
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from custom_models.lsconv.operator import ConvBN, LSConv

ROOT = Path('/home/pwt/Paper1')
REPORT = ROOT / 'work_dirs/diagnostics/lsconv_scope_audit.json'
register_all_modules()
logging.disable(logging.CRITICAL)
torch.set_num_threads(2)


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def verify_sharing(model, projected):
    head = model.bbox_head.head_module
    for towers in (head.cls_convs, head.reg_convs):
        for depth in range(2):
            operators = [towers[level][depth][0] for level in range(3)]
            for attr in ('cv1', 'cv2', 'cv3'):
                assert len({id(getattr(op.lkp, attr).c.weight) for op in operators}) == 1
                assert len({id(getattr(op.lkp, attr).bn) for op in operators}) == 3
            assert len({id(op.lkp.cv4.weight) for op in operators}) == 1
            assert len({id(op.lkp.norm) for op in operators}) == 3
            assert len({id(op.bn) for op in operators}) == 3
            if projected:
                projections = [towers[level][depth][1] for level in range(3)]
                assert len({id(op.c.weight) for op in projections}) == 1
                assert len({id(op.bn) for op in projections}) == 3
    cls = head.cls_convs[0][0][0].lkp.cv1.c.weight
    reg = head.reg_convs[0][0][0].lkp.cv1.c.weight
    assert cls is not reg


def build(mode):
    seed()
    cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
    model = MODELS.build(deepcopy(cfg.model))
    init_stats = {}
    if mode != 'baseline':
        projected = mode == 'with_channel_projections'
        predictors = [id(p) for group in (model.bbox_head.head_module.rtm_cls,
                                         model.bbox_head.head_module.rtm_reg)
                      for p in group.parameters()]
        count = 0
        for _, block in list(model.named_modules()):
            if isinstance(block, CSPNeXtBlock):
                old = block.conv1
                conv = old.conv
                assert conv.in_channels == conv.out_channels and conv.stride == (1, 1)
                block.conv1 = nn.Sequential(LSConv(conv.in_channels), deepcopy(old.activate))
                count += 1
        assert count == 8
        for level in range(3):
            old = model.neck.out_layers[level]
            cin, cout = old.conv.in_channels, old.conv.out_channels
            assert cout == 96
            layers = []
            if cin != cout or projected:
                layers.append(ConvBN(cin, cout))
            layers.extend((LSConv(cout), deepcopy(old.activate)))
            model.neck.out_layers[level] = nn.Sequential(*layers)
        head = model.bbox_head.head_module
        for towers in (head.cls_convs, head.reg_convs):
            for depth in range(2):
                shared_operator = shared_projection = None
                for level in range(3):
                    old = towers[level][depth]
                    op = LSConv(96)
                    projection = ConvBN(96, 96) if projected else None
                    if shared_operator is None:
                        shared_operator, shared_projection = op, projection
                    else:
                        for attr in ('cv1', 'cv2', 'cv3'):
                            getattr(op.lkp, attr).c = getattr(shared_operator.lkp, attr).c
                        op.lkp.cv4 = shared_operator.lkp.cv4
                        if projected:
                            projection.c = shared_projection.c
                    layers = [op]
                    if projected:
                        layers.append(projection)
                    layers.append(deepcopy(old.activate))
                    towers[level][depth] = nn.Sequential(*layers)
        assert predictors == [id(p) for group in (head.rtm_cls, head.rtm_reg)
                              for p in group.parameters()]
        verify_sharing(model, projected)
        op = head.cls_convs[0][0][0]
        init_stats['head_lkp_before_std'] = {
            attr: float(getattr(op.lkp, attr).c.weight.std()) for attr in ('cv1', 'cv2', 'cv3')}
    model.init_weights()
    if mode != 'baseline':
        verify_sharing(model, projected)
        op = model.bbox_head.head_module.cls_convs[0][0][0]
        init_stats['head_lkp_after_std'] = {
            attr: float(getattr(op.lkp, attr).c.weight.std()) for attr in ('cv1', 'cv2', 'cv3')}
        assert all(torch.allclose(conv.bias, torch.full_like(conv.bias, -4.5951198501))
                   for conv in model.bbox_head.head_module.rtm_cls)
    return model, cfg, init_stats


def count(model):
    model.eval()
    macs = [0, 0]
    def conv_hook(module, args, output):
        macs[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
    def ls_hook(module, args, output):
        macs[1] += output.numel() * 9
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        if isinstance(module, LSConv):
            handles.append(module.register_forward_hook(ls_hook))
    with torch.no_grad():
        output = model._forward(torch.randn(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    shapes = [[list(t.shape) for t in group] for group in output]
    assert shapes == [[[1, 20, s, s] for s in (80, 40, 20)],
                      [[1, 4, s, s] for s in (80, 40, 20)]]
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                conv_gmac=float(macs[0] / 1e9), aggregation_gmac=float(macs[1] / 1e9),
                total_counted_gmac=float(sum(macs) / 1e9), output_shapes=shapes)


def loader(config, pipeline=None):
    config = deepcopy(config)
    config.update(batch_size=2, num_workers=0, persistent_workers=False, pin_memory=False)
    config.sampler.shuffle = False
    if pipeline is not None:
        config.dataset.pipeline = deepcopy(pipeline)
    return Runner.build_dataloader(config, seed=42)


def batches(cfg):
    seed()
    strong = iter(loader(cfg.train_dataloader))
    result = [next(strong) for _ in range(6)]
    weak = iter(loader(cfg.train_dataloader, cfg.train_pipeline_stage2))
    result.extend(next(weak) for _ in range(2))
    val = iter(loader(cfg.val_dataloader))
    return result, [next(val) for _ in range(2)]


def latency(model):
    model.eval()
    result = {}
    for batch in (1, 2):
        inputs = torch.randn(batch, 3, 640, 640, device='cuda')
        for amp in (False, True):
            with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                for _ in range(5):
                    model._forward(inputs)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    output = model._forward(inputs)
                end.record()
                torch.cuda.synchronize()
                assert all(torch.isfinite(t).all() for group in output for t in group)
                result[f'batch{batch}_{"amp" if amp else "fp32"}_ms'] = start.elapsed_time(end) / 20
    return result


def short_train(model, cfg, mode, training, validation):
    model.train()
    # Check unscaled FP32 gradients separately from GradScaler's startup range.
    initial_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    torch.cuda.reset_peak_memory_stats()
    prepared = model.data_preprocessor(deepcopy(training[-1]), training=True)
    fp32_losses = model(**prepared, mode='loss')
    fp32_total = sum(value.sum() for value in fp32_losses.values())
    assert torch.isfinite(fp32_total)
    fp32_total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    fp32_check = dict(status='passed', loss=float(fp32_total),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(initial_buffers[name])
    model.zero_grad(set_to_none=True)
    del prepared, fp32_losses, fp32_total, initial_buffers
    watched_ids = set()
    if mode != 'baseline':
        for module in model.modules():
            if isinstance(module, (LSConv, ConvBN)):
                watched_ids.update(id(p) for p in module.parameters())
    watched = {name: p for name, p in model.named_parameters() if id(p) in watched_ids}
    before = {name: p.detach().cpu().clone() for name, p in watched.items()}
    gradients = {}
    overflow_names = set()
    handles = []
    for name, p in watched.items():
        def capture(grad, name=name):
            if torch.isfinite(grad).all():
                gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
            else:
                overflow_names.add(name)
        handles.append(p.register_hook(capture))
    optim_cfg = deepcopy(cfg.optim_wrapper)
    optim_cfg.optimizer.lr = .0005625
    optim_cfg.loss_scale = dict(init_scale=128)
    optimizer = build_optim_wrapper(model, optim_cfg)
    optimizer_params = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in optimizer_params}) == len(optimizer_params)
    assert all(id(p) in {id(q) for q in optimizer_params} for p in watched.values())
    hook = EMAHook(ema_type='ExpMomentumEMA', momentum=.0002, gamma=2000,
                   update_buffers=True)
    runner = SimpleNamespace(model=model, iter=0, epoch=0)
    hook.before_run(runner)
    if mode != 'baseline':
        verify_sharing(hook.ema_model.module, mode == 'with_channel_projections')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    amp_backoffs = []
    anchor = optimizer_params[0]
    i = 0
    actual_updates = 0
    while actual_updates < 8 and i < 32:
        data = training[i % len(training)]
        scale_before = optimizer.loss_scaler.get_scale()
        logs = model.train_step(deepcopy(data), optimizer)
        assert all(torch.isfinite(t).all() for t in logs.values())
        losses.append({k: float(v) for k, v in logs.items()})
        scale_after = optimizer.loss_scaler.get_scale()
        if scale_after < scale_before:
            amp_backoffs.append(dict(iteration=i, before=scale_before, after=scale_after))
        actual_updates = int(optimizer.optimizer.state[anchor].get('step', 0))
        runner.iter = i
        hook.after_train_iter(runner, i)
        i += 1
    assert actual_updates == 8, (actual_updates, i, amp_backoffs)
    # Check final unscaled gradients after successful updates, not overflow attempts.
    assert all(torch.isfinite(p.grad).all() for p in optimizer_params if p.grad is not None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    updates = {name: float((p.detach().cpu() - before[name]).abs().max()) for name, p in watched.items()}
    steps = {name: int(optimizer.optimizer.state[p].get('step', 0)) for name, p in watched.items()}
    inactive = [name for name in watched if not gradients.get(name, 0) or not updates[name]]
    assert not inactive, inactive
    assert all(value == 8 for value in steps.values()), steps
    for handle in handles:
        handle.remove()
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    hook.before_val_epoch(runner)
    predictions = 0
    with torch.no_grad():
        for data in validation:
            values = model.test_step(deepcopy(data))
            for value in values:
                pred = value.pred_instances
                assert torch.isfinite(pred.bboxes).all() and torch.isfinite(pred.scores).all()
                assert ((pred.labels >= 0) & (pred.labels < 20)).all()
                predictions += 1
    hook.after_val_epoch(runner)
    assert all(torch.equal(value, raw[name]) for name, value in model.state_dict().items())
    if mode != 'baseline':
        verify_sharing(model, mode == 'with_channel_projections')
    checkpoint = dict(state_dict={name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()})
    hook.before_save_checkpoint(runner, checkpoint)
    restored, _, _ = build(mode)
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    assert all(torch.equal(value, checkpoint['state_dict'][name].cpu())
               for name, value in restored.state_dict().items())
    if mode != 'baseline':
        verify_sharing(restored, mode == 'with_channel_projections')
    result = dict(status='passed', batch=2, training_updates=actual_updates,
                  training_iterations=i, source_batches='six strong and two weak augmentation batches, cycled if AMP backs off',
                  watched_parameter_tensors=len(watched), fp32_backward=fp32_check,
                  amp_backoffs=amp_backoffs, scaled_overflow_parameters=sorted(overflow_names),
                  optimizer_unique=True, gradients_finite_nonzero=True, updates_nonzero=True,
                  losses=losses, gradient_max=gradients, parameter_updates_max=updates,
                  optimizer_steps=steps, ema_swap_restore=True, checkpoint_roundtrip=True,
                  validation_predictions=predictions,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  elapsed_seconds=elapsed, loss_scale_init=128, saved_checkpoints=False)
    del hook, runner, optimizer, raw, restored, checkpoint, watched, before, optimizer_params
    gc.collect()
    return result


def main():
    free, total = torch.cuda.mem_get_info()
    assert free / 2**30 > 10
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
    training, validation = batches(cfg)
    report = dict(status='running', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  gpu=torch.cuda.get_device_name(0), allocator_fraction=.12,
                  formal_experiments_changed=False, formal_experiment_queued=False,
                  latency_conditions='Shared GPU0; unfused PyTorch raw forward; no preprocessing/NMS; AMP and FP32',
                  cases={})
    for mode in ('baseline', 'direct_fifteen', 'with_channel_projections'):
        model, cfg, stats = build(mode)
        case = count(model)
        expected = {'baseline': 4878708, 'direct_fifteen': 3336729,
                    'with_channel_projections': 3385305}[mode]
        assert case['parameters'] == expected, case
        case['initialization'] = stats
        model.cuda()
        case['latency'] = latency(model)
        case['gpu_short_train'] = short_train(model, cfg, mode, training, validation)
        report['cases'][mode] = case
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(mode=mode, parameters=case['parameters'],
                              counted_gmac=case['total_counted_gmac'],
                              latency=case['latency'],
                              gpu_checks={k: v for k, v in case['gpu_short_train'].items()
                                          if k not in ('losses', 'gradient_max', 'parameter_updates_max', 'optimizer_steps')})), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'passed'
    REPORT.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
