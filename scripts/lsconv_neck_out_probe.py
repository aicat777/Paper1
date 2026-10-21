"""Isolated checks for LSConv-before-projection neck outputs."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from custom_models.lsconv.operator import LSConv as BackboneLSConv
from custom_models.lsconv_neck_out.operator import LSConv

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'lsconv_neck_out'


def watched_parameters(model):
    ids = {id(p) for m in model.backbone.modules() if isinstance(m, BackboneLSConv)
           for p in m.parameters()}
    ids.update(id(p) for p in model.neck.out_layers.parameters())
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def preflight(cfg, output, config_path):
    baseline = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == baseline[key], key
    expected = deepcopy(baseline.model)
    expected.neck.type = 'LSConvOutputPAFPN'
    assert cfg.model == expected
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    model.eval()
    macs, sites, projections, handles = [0, 0], [], [], []

    def count_conv(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    def count_ska(path):
        def capture(module, inputs, result):
            macs[1] += result.numel() * 9
            sites.append(dict(path=path, shape=list(result.shape)))
        return capture

    def count_projection(module, inputs, result):
        projections.append(dict(input=list(inputs[0].shape), output=list(result.shape),
                                kernel=list(module.conv.kernel_size)))

    for path, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count_conv))
        if isinstance(layer, (BackboneLSConv, LSConv)):
            handles.append(layer.register_forward_hook(count_ska(path)))
    for layer in model.neck.out_layers:
        handles.append(layer.projection.register_forward_hook(count_projection))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert [site['shape'] for site in sites] == [[1, c, s, s] for c, s in
        ((48, 80), (96, 40), (192, 20), (96, 80), (192, 40), (384, 20))]
    assert [site['path'] for site in sites[3:]] == [f'neck.out_layers.{i}.mixer' for i in range(3)]
    assert [p['input'] for p in projections] == [site['shape'] for site in sites[3:]]
    assert [p['output'] for p in projections] == [[1, 96, s, s] for s in (80, 40, 20)]
    priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls], with_stride=True)
    assert sum(len(x) for x in priors) == 8400
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 4277142
    assert math.isclose(macs[0] / 1e9, 7.09363584, abs_tol=1e-10)
    assert math.isclose(macs[1] / 1e9, .0145152, abs_tol=1e-10)
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected
    report_to(output, dict(status='passed', method=METHOD, parameters=parameters,
        conv_gmac=macs[0] / 1e9, aggregation_gmac=macs[1] / 1e9, counted_gmac=sum(macs) / 1e9,
        excludes_norm_activation_residual_and_postprocessing=True, lsconv_sites=sites,
        projections=projections, runtime_lsconv_calls=len(sites), prediction_points=8400,
        watched_parameter_tensors=len(watched_parameters(model)), scratch=True,
        input_size=[640, 640], formal_batch=36, accumulation=1, train_images=5862, val_images=5863,
        formal_batch_capacity_tested=False, aggregation_before_value_projection=True,
        unchanged_fpn_pan_and_head=True, **config_identity(config_path)))


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Allocator limit must leave GPU memory headroom')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    return free, total


def config_identity(path):
    if path is None:
        return {}
    path = Path(path).resolve()
    return dict(config=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@HOOKS.register_module()
class LSOutputAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert self.watched and len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, value in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            value.register_hook(capture)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()


def smoke(cfg, output, fraction):
    directory = Path(output).parent / 'smoke'
    if directory.exists():
        raise RuntimeError('Existing short-test outputs would be overwritten')
    directory.mkdir(parents=True)
    cfg.work_dir = str(directory)
    cfg.train_cfg.update(max_epochs=2, val_interval=1, dynamic_intervals=[])
    cfg.param_scheduler = []
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger.interval = 1
    cfg.optim_wrapper.loss_scale = dict(init_scale=32)
    cfg.custom_hooks[1].switch_epoch = 1
    cfg.train_dataloader.update(batch_size=2, num_workers=0, persistent_workers=False)
    cfg.train_dataloader.dataset.indices = list(range(16))
    cfg.val_dataloader.update(batch_size=4, num_workers=0, persistent_workers=False)
    annotation = json.loads((Path(cfg.data_root) / cfg.val_dataloader.dataset.ann_file).read_text())
    categories = {row['id'] for row in annotation['categories']}
    by_image = {row['id']: set() for row in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].add(row['category_id'])
    selected = []
    while categories:
        image = max(by_image, key=lambda key: len(by_image[key] & categories))
        assert by_image[image] & categories
        selected.append(image)
        categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.lsconv_neck_out_probe')
    cfg.custom_hooks.append(dict(type='LSOutputAuditHook', method=METHOD,
        report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (METHOD + '_smoke.py')
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', METHOD, flush=True)


def capacity(cfg, output, fraction, batch=None, config_path=None):
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or cfg.train_dataloader.batch_size
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda(); model.train()
    optim_cfg = deepcopy(cfg.optim_wrapper)
    optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model)
    initial = {name: p.detach().cpu().clone() for name, p in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert watched and len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients = {}
    for name, value in watched.items():
        def capture(gradient, name=name):
            assert torch.isfinite(gradient).all(), name
            gradients[name] = max(gradients.get(name, 0), float(gradient.abs().max()))
        value.register_hook(capture)
    started, losses = time.perf_counter(), []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in result.values()), result
        losses.append({key: float(value) for key, value in result.items()})
        torch.cuda.synchronize()
    updates = {name: float((p.detach().cpu() - initial[name]).abs().max()) for name, p in watched.items()}
    assert all(gradients.get(name, 0) > 0 and updates[name] > 0 for name in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=METHOD, microbatch=batch, accumulation=1,
        optimizer_updates=3, gt_per_image=1200, different_image_ids=image_ids,
        repeated_annotations=True, synthetic_multiscale_boxes=True,
        formal_batch=36, formal_batch_capacity_tested=batch == 36,
        allocator_fraction=fraction, loss_scale_init=32, warmup_enabled=False,
        amp_skipped_updates=0, saved_checkpoints=False, losses=losses,
        watched_parameter_tensors=len(watched), gradient_max=gradients, parameter_updates_max=updates,
        cuda_device=torch.cuda.get_device_name(), cuda_total_gib=total / 2**30,
        initial_free_gib=free / 2**30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        elapsed_seconds=time.perf_counter() - started, **config_identity(config_path)))
    print('CAPACITY_PASSED', batch, flush=True)


class CoordinateSKA(nn.Module):
    """Independent coordinate shifts for author-class numerical comparison."""
    def forward(self, x, weights):
        _, channels, height, width = x.shape
        padded = F.pad(x, (1, 1, 1, 1))
        groups = torch.arange(channels, device=x.device) % weights.shape[1]
        result = torch.zeros_like(x, dtype=torch.float32)
        for row in range(3):
            for col in range(3):
                value = padded[:, :, row:row + height, col:col + width].float()
                kernel = weights[:, groups, row * 3 + col].to(x.dtype).float()
                result = result + value * kernel
        return result.to(x.dtype)


def fidelity(reference, output, fraction):
    require_memory(fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    path = Path(reference)
    source = path.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    assert source_hash == 'f6fbe61695d3a257df8bcf04beebdea9929bae804ab86ed9975448dce948a419'
    names = ('Conv2d_BN', 'LKP', 'LSConv')
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
    assert len(classes) == 3
    environment = dict(torch=torch, nn=nn, SKA=CoordinateSKA)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), environment)
    cases = []
    for width in (96, 192, 384):
        for amp in (False, True):
            for training in (False, True):
                torch.manual_seed(73)
                ours = LSConv(width)
                torch.manual_seed(73)
                original = environment['LSConv'](width)
                assert set(ours.state_dict()) == set(original.state_dict())
                for name, value in ours.state_dict().items():
                    torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0, msg=name)
                ours.cuda().train(training)
                original.cuda().train(training)
                dtype = torch.float16 if amp else torch.float32
                x = torch.randn(2, width, 7, 9, device='cuda', dtype=dtype, requires_grad=True)
                rx = x.detach().clone().requires_grad_()
                with torch.autocast('cuda', enabled=amp):
                    actual, expected = ours(x), original(rx)
                atol, rtol = (.005, .03) if amp else (5e-5, 5e-4)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                seed = torch.randn_like(actual) / math.sqrt(actual.numel())
                names, parameters = zip(*ours.named_parameters())
                refs = dict(original.named_parameters())
                gradients = torch.autograd.grad(actual, (x, *parameters), seed)
                ref_gradients = torch.autograd.grad(expected, (rx, *(refs[n] for n in names)), seed)
                for name, first, second in zip(('input', *names), gradients, ref_gradients):
                    torch.testing.assert_close(first, second, atol=atol, rtol=rtol, msg=name)
                cases.append(dict(width=width, amp=amp, training=training, constructor_state_exact=True,
                    input_size=[7, 9], forward_max_error=float((actual.float() - expected.float()).abs().max()),
                    gradient_max_error=max(float((a.float() - b.float()).abs().max())
                                           for a, b in zip(gradients, ref_gradients))))
    report_to(output, dict(status='passed', method=METHOD, reference=str(path), reference_sha256=source_hash,
        upstream_commit='cbe737c92b7c43ecf02d08545a07f03f1010177c',
        original_classes_executed=['Conv2d_BN', 'LKP', 'LSConv'],
        reference_ska='independent coordinate shifts', cases=cases))
    print('FIDELITY_PASSED', len(cases), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/lsconv_neck_out/rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    args = parser.parse_args()
    register_all_modules()
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    assert cfg.model.neck.type == 'LSConvOutputPAFPN'
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            if not args.reference:
                raise ValueError('Fidelity requires the pinned author lsnet.py')
            fidelity(args.reference, args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.config)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=METHOD, mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
