"""Audit two independent CSP-only LS variants; no full training or queue edits."""
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

# Reuse unchanged audit/data utilities; candidate model code stays independent.
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from custom_models.lsblock_csp.block import LSFeatureBlock
from custom_models.lsblock_csp.operator import LSConv as BlockLSConv
from custom_models.lsconv_csp.operator import LSConv as PlainLSConv

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('lsconv_csp', 'lsblock_csp')


def watched_parameters(model):
    ids = {id(p) for m in model.modules() if isinstance(m, (PlainLSConv, LSFeatureBlock))
           for p in m.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Invalid GPU allocator fraction')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Not enough free memory for the limit plus a 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)


def preflight(cfg, method, output):
    baseline = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_pipeline', 'test_dataloader', 'test_evaluator', 'custom_hooks',
                'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == baseline[key], key
    assert cfg.model.bbox_head == baseline.model.bbox_head
    assert cfg.model.train_cfg == baseline.model.train_cfg
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.eval()
    core_type = PlainLSConv if method == 'lsconv_csp' else BlockLSConv
    cores = [(name, m) for name, m in model.named_modules() if isinstance(m, core_type)]
    assert len(cores) == 8
    assert sum(name.startswith('backbone.') for name, _ in cores) == 4
    assert sum(name.startswith('neck.') for name, _ in cores) == 4
    assert all('.blocks.' in name for name, _ in cores)
    assert type(model.bbox_head.head_module).__name__ == 'RTMDetSepBNHeadModule'
    for layer, width in zip(model.neck.out_layers, (96, 192, 384)):
        assert layer.conv.kernel_size == (3, 3)
        assert (layer.conv.in_channels, layer.conv.out_channels) == (width, 96)
    macs, shapes, handles = [0, 0], [], []
    def count_conv(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    def count_aggregation(module, inputs, result):
        macs[1] += result.numel() * 9
        shapes.append(list(result.shape))
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count_conv))
        if isinstance(layer, core_type):
            handles.append(layer.register_forward_hook(count_aggregation))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert len(shapes) == 8
    for loader, expected in [(cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)]:
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected
    blocks = [m for m in model.modules() if isinstance(m, LSFeatureBlock)]
    assert len(blocks) == (8 if method == 'lsblock_csp' else 0)
    for block in blocks:
        assert torch.count_nonzero(block.ffn.m.pw2.bn.weight) == 0
    report_to(output, dict(status='passed', method=method,
              parameters=sum(p.numel() for p in model.parameters()),
              conv_gmac=macs[0] / 1e9, aggregation_gmac=macs[1] / 1e9,
              counted_gmac=sum(macs) / 1e9, runtime_lsconv_calls=len(shapes),
              lsconv_shapes=shapes, replacement_paths=[name for name, _ in cores],
              watched_parameter_tensors=len(watched_parameters(model)), prediction_points=8400,
              scratch=True, microbatch=36, accumulation=1, input_size=[640, 640],
              train_images=5862, val_images=5863, baseline_head_and_neck_outputs=True,
              ffn_terminal_bn_zero=bool(blocks), formal_batch_capacity_tested=False))


class CoordinateSKA(nn.Module):
    """Independent explicit shifts, not unfold or any of the Triton kernels."""
    def forward(self, x, weights):
        b, channels, height, width = x.shape
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
    # Compare the custom aggregation rather than TF32/cudnn algorithm noise.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    reference = Path(reference)
    source = reference.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != 'f6fbe61695d3a257df8bcf04beebdea9929bae804ab86ed9975448dce948a419':
        raise ValueError('Reference does not match the pinned LSNet source')
    tree = ast.parse(source)
    classes = ('Conv2d_BN', 'Residual', 'FFN', 'LKP', 'LSConv', 'Block')
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in classes]
    assert len(nodes) == len(classes)
    environment = dict(torch=torch, nn=nn, SKA=CoordinateSKA)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), environment)
    cases = []
    for method in METHODS:
        for width in (24, 48, 96, 192):
            for amp in (False, True):
                for training in (False, True):
                    print('FIDELITY_CASE', method, width, 'amp', amp, 'training', training, flush=True)
                    torch.manual_seed(73)
                    ours = PlainLSConv(width) if method == 'lsconv_csp' else LSFeatureBlock(width)
                    torch.manual_seed(73)
                    original = (environment['LSConv'](width) if method == 'lsconv_csp'
                                else environment['Block'](width, 16, stage=0, depth=1))
                    assert set(ours.state_dict()) == set(original.state_dict())
                    for name, value in ours.state_dict().items():
                        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0, msg=name)
                    # Exercise the full FFN, not just its zero-scale initial identity.
                    if method == 'lsblock_csp':
                        with torch.no_grad():
                            ours.ffn.m.pw2.bn.weight.fill_(.3)
                            original.ffn.m.pw2.bn.weight.fill_(.3)
                    ours.cuda().train(training); original.cuda().train(training)
                    dtype = torch.float16 if amp else torch.float32
                    x = torch.randn(2, width, 7, 9, device='cuda', dtype=dtype, requires_grad=True)
                    rx = x.detach().clone().requires_grad_()
                    with torch.autocast('cuda', enabled=amp):
                        actual, expected = ours(x), original(rx)
                    atol, rtol = (.005, .03) if amp else (5e-5, 5e-4)
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    seed = torch.randn_like(actual) / math.sqrt(actual.numel())
                    names, params = zip(*ours.named_parameters())
                    ref = dict(original.named_parameters())
                    a = torch.autograd.grad(actual, (x, *params), seed)
                    e = torch.autograd.grad(expected, (rx, *(ref[name] for name in names)), seed)
                    for name, first, second in zip(('input', *names), a, e):
                        torch.testing.assert_close(first, second, atol=atol, rtol=rtol, msg=name)
                    cases.append(dict(method=method, width=width, amp=amp, training=training,
                        constructor_state_exact=True, input_size=[7, 9],
                        forward_max_error=float((actual.float() - expected.float()).abs().max()),
                        gradient_max_error=max(float((first.float() - second.float()).abs().max())
                                               for first, second in zip(a, e))))
                    del ours, original, actual, expected, a, e
    report_to(output, dict(status='passed', reference=str(reference),
              reference_sha256=hashlib.sha256(source.encode()).hexdigest(),
              upstream_commit='cbe737c92b7c43ecf02d08545a07f03f1010177c',
              original_classes_executed=list(classes), reference_ska='independent explicit shifts',
              nonzero_ffn_exercised=True, cases=cases))
    print('FIDELITY_PASSED', len(cases), flush=True)


@HOOKS.register_module()
class LSCSPAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {name: value.detach().cpu().clone() for name, value in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, value in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            value.register_hook(capture)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()


def smoke(cfg, method, output, fraction):
    directory = Path(output).parent / 'smoke'
    if directory.exists():
        raise RuntimeError('Existing short-test output would be overwritten')
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
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.lsconv_csp_probe')
    cfg.custom_hooks.append(dict(type='LSCSPAuditHook', method=method, report_path=str(output),
                                 image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (method + '_smoke.py')
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', method, flush=True)


def capacity(cfg, method, output, fraction, batch=None):
    require_memory(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or cfg.train_dataloader.batch_size
    data, ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model)); model.init_weights(); model.cuda().train()
    settings = deepcopy(cfg.optim_wrapper)
    settings.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, settings)
    watched = watched_parameters(model)
    initial = {n: p.detach().cpu().clone() for n, p in watched.items()}
    gradients = {}
    for name, p in watched.items():
        def capture(grad, name=name):
            assert torch.isfinite(grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(grad.abs().max()))
        p.register_hook(capture)
    start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    losses = []
    for _ in range(3):
        values = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in values.values())
        losses.append({k: float(v) for k, v in values.items()})
        torch.cuda.synchronize()
    updates = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in watched.items()}
    assert all(gradients.get(n, 0) > 0 and updates[n] > 0 for n in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=method, microbatch=batch, accumulation=1,
              optimizer_updates=3, gt_per_image=1200, different_image_ids=ids,
              repeated_annotations=True, synthetic_multiscale_boxes=True,
              allocator_fraction=fraction, amp_skipped_updates=0, saved_checkpoints=False,
              formal_batch_capacity_tested=batch == 36, loss_scale_init=32,
              watched_parameter_tensors=len(watched), gradient_max=gradients,
              parameter_updates_max=updates, losses=losses,
              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
              elapsed_seconds=time.perf_counter() - start))
    print('CAPACITY_PASSED', method, batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--allocator-fraction', type=float, default=.25)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(str(ROOT / 'configs' / args.method / 'rtmdet_tiny_dior_640.py'))
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.method, args.report)
        elif args.mode == 'smoke':
            smoke(cfg, args.method, args.report, args.allocator_fraction)
        elif args.mode == 'capacity':
            capacity(cfg, args.method, args.report, args.allocator_fraction, args.batch)
        else:
            if not args.reference:
                raise ValueError('Fidelity requires the pinned official lsnet.py')
            fidelity(args.reference, args.report, args.allocator_fraction)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=args.method, mode=args.mode,
                  error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
