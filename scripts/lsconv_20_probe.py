"""Independent preflight, reference, official-entry smoke and capacity audits."""
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

from custom_models.lsconv_20.operator import ConvBN, LSConv
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/lsconv_20/rtmdet_tiny_dior_640.py'


def watched_parameters(model):
    ids = {id(p) for module in model.modules() if isinstance(module, (LSConv, ConvBN))
           for p in module.parameters()}
    return {name: p for name, p in model.named_parameters() if id(p) in ids}


def check_sharing(head):
    for towers in (head.cls_convs, head.reg_convs):
        for depth in range(head.stacked_convs):
            operators = [tower[depth][0] for tower in towers]
            for name in ('cv1', 'cv2', 'cv3'):
                layers = [getattr(op.lkp, name) for op in operators]
                assert len({id(layer.c) for layer in layers}) == 1
                assert len({id(layer.bn) for layer in layers}) == 3
            assert len({id(op.lkp.cv4) for op in operators}) == 1
            assert len({id(op.lkp.norm) for op in operators}) == 3
            assert len({id(op.bn) for op in operators}) == 3
            projections = [tower[depth][1] for tower in towers]
            assert len({id(layer.c) for layer in projections}) == 1
            assert len({id(layer.bn) for layer in projections}) == 3
    assert not ({id(p) for p in head.cls_convs.parameters()}
                & {id(p) for p in head.reg_convs.parameters()})


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Allocator fraction must leave memory headroom')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus a 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    return free, total


def preflight(cfg, output, config_path=None):
    baseline = Config.fromfile(str(ROOT / 'configs/lsconv/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_pipeline', 'test_dataloader', 'test_evaluator', 'custom_hooks',
                'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == baseline[key], key
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    model = MODELS.build(deepcopy(cfg.model))
    before = {name: {key: value.clone() for key, value in layer.state_dict().items()}
              for name, layer in model.named_modules() if isinstance(layer, LSConv)}
    model.init_weights(); model.eval()
    for name, expected in before.items():
        for key, value in model.get_submodule(name).state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0, msg=name + '.' + key)
    assert len(before) == 20
    assert len(model.backbone.lsconv_replacements) == 4
    assert len(model.neck.lsconv_replacements) == 4
    assert len(model.bbox_head.head_module.lsconv_replacements) == 12
    assert len(watched_parameters(model)) == 288
    check_sharing(model.bbox_head.head_module)
    for layer, width in zip(model.neck.out_layers, (96, 192, 384)):
        assert layer.conv.kernel_size == (3, 3)
        assert (layer.conv.in_channels, layer.conv.out_channels) == (width, 96)
        assert not any(isinstance(child, LSConv) for child in layer.modules())
    macs, calls, handles = [0, 0], [], []
    def convolution(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    def aggregation(module, inputs, result):
        macs[1] += result.numel() * 9
        calls.append(list(result.shape))
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(convolution))
        if isinstance(layer, LSConv):
            handles.append(layer.register_forward_hook(aggregation))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert len(calls) == 20
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert sum(p.numel() for p in model.parameters()) == 3855645
    assert sum(macs) == 5179818240
    for loader, expected in [(cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)]:
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected
    head = model.bbox_head.head_module
    report = dict(status='passed', method='lsconv_20', parameters=3855645,
              conv_gmac=macs[0] / 1e9, aggregation_gmac=macs[1] / 1e9,
              counted_gmac=sum(macs) / 1e9, gflops_two_ops_per_mac=2 * sum(macs) / 1e9,
              runtime_lsconv_calls=20, lsconv_shapes=calls, prediction_points=8400,
              original_neck_outputs=True, head_convolutions_shared=True,
              scale_norms_independent=True, head_bn=sum(isinstance(m, nn.BatchNorm2d) for m in head.modules()),
              head_gn=sum(isinstance(m, nn.GroupNorm) for m in head.modules()),
              author_constructor_states_preserved=True, scratch=True, microbatch=36,
              accumulation=1, input_size=[640, 640], train_images=5862, val_images=5863,
              watched_parameter_tensors=288, formal_batch_capacity_tested=False)
    if config_path:
        report.update(config=str(Path(config_path).resolve()),
                      config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest())
    report_to(output, report)


class CoordinateSKA(nn.Module):
    """Independent explicit neighborhood shifts; no unfold or Triton reference."""
    def forward(self, x, weights):
        _, channels, height, width = x.shape
        padded = F.pad(x, (1, 1, 1, 1))
        indices = torch.arange(channels, device=x.device) % weights.shape[1]
        result = torch.zeros_like(x, dtype=torch.float32)
        for row in range(3):
            for col in range(3):
                value = padded[:, :, row:row + height, col:col + width].float()
                kernel = weights[:, indices, row * 3 + col].to(x.dtype).float()
                result = result + value * kernel
        return result.to(x.dtype)


def fidelity(reference, output, fraction):
    require_memory(fraction)
    # This process alone disables TF32/autotuning to compare arithmetic order.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source = Path(reference).read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != 'f6fbe61695d3a257df8bcf04beebdea9929bae804ab86ed9975448dce948a419':
        raise ValueError('Reference differs from the pinned author source')
    classes = ('Conv2d_BN', 'LKP', 'LSConv')
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.ClassDef) and node.name in classes]
    assert len(nodes) == len(classes)
    environment = dict(torch=torch, nn=nn, SKA=CoordinateSKA)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), environment)
    cases = []
    shapes = [(24, 7, 9), (48, 7, 9), (96, 7, 9), (192, 7, 9)]
    combinations = [(c, h, w, amp, train) for c, h, w in shapes
                    for amp in (False, True) for train in (False, True)]
    combinations += [(24, 160, 160, False, False), (96, 80, 80, True, True),
                     (96, 40, 40, True, True), (96, 20, 20, True, True)]
    for width, height, side, amp, training in combinations:
        torch.manual_seed(73)
        ours = LSConv(width)
        torch.manual_seed(73)
        original = environment['LSConv'](width)
        assert set(ours.state_dict()) == set(original.state_dict())
        for name, value in ours.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0, msg=name)
        ours.cuda().train(training); original.cuda().train(training)
        dtype = torch.float16 if amp else torch.float32
        x = torch.randn(2, width, height, side, device='cuda', dtype=dtype, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        with torch.autocast('cuda', enabled=amp):
            actual, expected = ours(x), original(rx)
        atol, rtol = (.005, .03) if amp else (5e-5, 5e-4)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        seed = torch.randn_like(actual) / math.sqrt(actual.numel())
        names, parameters = zip(*ours.named_parameters())
        ref = dict(original.named_parameters())
        a = torch.autograd.grad(actual, (x, *parameters), seed)
        e = torch.autograd.grad(expected, (rx, *(ref[name] for name in names)), seed)
        for name, first, second in zip(('input', *names), a, e):
            torch.testing.assert_close(first, second, atol=atol, rtol=rtol, msg=name)
        cases.append(dict(width=width, size=[height, side], amp=amp, training=training,
            constructor_state_exact=True, forward_max_error=float((actual.float() - expected.float()).abs().max()),
            gradient_max_error=max(float((first.float() - second.float()).abs().max())
                                   for first, second in zip(a, e))))
        print('FIDELITY_CASE_PASSED', cases[-1], flush=True)
        del ours, original, actual, expected, a, e
    report_to(output, dict(status='passed', reference=str(reference), reference_sha256=digest,
              upstream_commit='cbe737c92b7c43ecf02d08545a07f03f1010177c',
              original_classes_executed=list(classes), reference_ska='independent explicit shifts', cases=cases))


@HOOKS.register_module()
class LSConv20AuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        check_sharing(self.model.bbox_head.head_module)
        self.watched = watched_parameters(self.model)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in parameters}
        for name, p in self.watched.items():
            def capture(gradient, name=name):
                assert torch.isfinite(gradient).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(gradient.abs().max()))
            p.register_hook(capture)
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def restore_check(self):
        super().restore_check()
        check_sharing(self.model.bbox_head.head_module)


def smoke(cfg, output, fraction):
    directory = Path(output).parent / 'smoke'
    if directory.exists():
        raise RuntimeError('Existing short-test directory would be overwritten')
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
    cfg.custom_imports.imports.append('scripts.lsconv_20_probe')
    cfg.custom_hooks.append(dict(type='LSConv20AuditHook', method='lsconv_20',
                                 report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / 'lsconv_20_smoke.py'
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode:
        raise RuntimeError(f'Official short training failed: {result.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', flush=True)


def capacity(cfg, output, fraction, batch=None, config_path=None, gt_per_image=1200):
    free, total = require_memory(fraction)
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or cfg.train_dataloader.batch_size
    data, image_ids = capacity_data(cfg, batch, gt_per_image=gt_per_image)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    check_sharing(model.bbox_head.head_module)
    settings = deepcopy(cfg.optim_wrapper)
    settings.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, settings)
    watched = watched_parameters(model)
    initial = {name: p.detach().cpu().clone() for name, p in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in watched.values()} <= {id(p) for p in parameters}
    gradients = {}
    for name, p in watched.items():
        def capture(gradient, name=name):
            assert torch.isfinite(gradient).all(), name
            gradients[name] = max(gradients.get(name, 0), float(gradient.abs().max()))
        p.register_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []
    for _ in range(3):
        values = model.train_step(data, optimizer)
        assert all(torch.isfinite(value).all() for value in values.values())
        losses.append({key: float(value) for key, value in values.items()})
        torch.cuda.synchronize()
    updates = {name: float((p.detach().cpu() - initial[name]).abs().max()) for name, p in watched.items()}
    inactive = [name for name in watched if not gradients.get(name, 0) > 0 or not updates[name] > 0]
    assert not inactive, inactive
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report = dict(status='passed', method='lsconv_20', microbatch=batch,
              accumulation=1, optimizer_updates=3, gt_per_image=gt_per_image,
              different_image_ids=image_ids,
              repeated_annotations=True, synthetic_multiscale_boxes=True, allocator_fraction=fraction,
              formal_batch_capacity_tested=batch == 36, saved_checkpoints=False,
              watched_parameter_tensors=len(watched), gradient_max=gradients,
              parameter_updates_max=updates, losses=losses, loss_scale_init=32,
              warmup_enabled=False, amp_skipped_updates=0,
              initial_free_gib=free / 2**30, total_gib=total / 2**30,
              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
              elapsed_seconds=time.perf_counter() - started)
    if config_path:
        report.update(config=str(Path(config_path).resolve()),
                      config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest())
    report_to(output, report)
    print('CAPACITY_PASSED', batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--allocator-fraction', type=float, default=.25)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--gt-per-image', type=int, default=1200)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, config_path=args.config)
        elif args.mode == 'fidelity':
            if not args.reference:
                raise ValueError('A pinned author source is required for fidelity')
            fidelity(args.reference, args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.report, args.allocator_fraction, args.batch,
                     config_path=args.config, gt_per_image=args.gt_per_image)
    except Exception as error:
        report_to(args.report, dict(status='failed', method='lsconv_20', mode=args.mode,
                  error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
