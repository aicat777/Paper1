"""Independent SCDown audits and short runs; never launch a full experiment."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
from mmcv.cnn import ConvModule, fuse_conv_bn
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown'
SITES = ('backbone.stage3.0', 'backbone.stage4.0', 'neck.downsample_layers.1')
AUTHOR_COMMIT = '453c6e38a51e9d1d5a2aa5fb7f1014a711913397'
AUTHOR_HASHES = dict(
    block='797a1fc75d9839147cbd3c432ad5d2fe16bc37a419a9f4d55c8449227cedb9fe',
    conv='9dd51834bd9ff432b8d72d6f55bb44bac301c97cbe2cc32c04a95c5a3450f59f')


def watched_parameters(model):
    return {name: p for name, p in model.named_parameters()
            if any(name.startswith(path + '.') for path in SITES)}


def require_memory(fraction):
    if not 0 < fraction < 1:
        raise ValueError('Allocator limit must leave GPU memory headroom')
    free, total = torch.cuda.mem_get_info()
    if free < total * fraction + 2 * 2**30:
        raise RuntimeError('Insufficient free GPU memory for allocator limit plus 2 GiB guard')
    torch.cuda.set_per_process_memory_fraction(fraction)
    return free, total


def config_identity(path):
    path = Path(path).resolve()
    return dict(config=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def preflight(cfg, output, path):
    original = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
    for key in ('train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'val_dataloader',
                'val_evaluator', 'test_dataloader', 'test_evaluator', 'train_cfg', 'custom_hooks',
                'default_hooks', 'randomness'):
        assert cfg[key] == original[key], key
    for key in ('bbox_head', 'train_cfg', 'test_cfg', 'data_preprocessor'):
        assert cfg.model[key] == original.model[key], key
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert math.isclose(cfg.optim_wrapper.optimizer.lr, .0005625, rel_tol=1e-12)
    assert cfg.param_scheduler[0].end == 1000
    models = []
    for candidate in (original, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(candidate.model))
        model.init_weights(); model.eval()
        models.append(model)
    reference, model = models
    for name, value in reference.state_dict().items():
        if not any(name.startswith(site + '.') for site in SITES):
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    macs, sites, handles = [0], [], []

    def count(module, inputs, result):
        # Count every execution, including the shared head convolutions.
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            handles.append(layer.register_forward_hook(count))
        if isinstance(layer, SCDown):
            handles.append(layer.register_forward_hook(lambda mod, inputs, result, name=name:
                sites.append(dict(path=name, input=list(inputs[0].shape), output=list(result.shape)))))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [site['path'] for site in sites] == list(SITES)
    assert [site['input'] for site in sites] == [[1, 96, 80, 80], [1, 192, 40, 40], [1, 192, 40, 40]]
    assert [site['output'] for site in sites] == [[1, 192, 40, 40], [1, 384, 20, 20], [1, 192, 20, 20]]
    assert [list(v.shape) for v in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(v.shape) for v in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 3854964 and len(watched_parameters(model)) == 18
    assert math.isclose(macs[0] / 1e9, 7.62390144, abs_tol=1e-10)
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected
    report_to(output, dict(status='passed', method=METHOD, parameters=parameters,
        inference_conv_linear_gmac=macs[0] / 1e9, parameter_reduction_percent=20.983916233560194,
        mac_reduction_percent=4.5545208450814005, scdown_sites=sites,
        first_pan_dense_and_neck_outputs_and_head_unchanged=True,
        unchanged_parameter_initialization=True, prediction_points=8400,
        scratch=True, input_size=[640, 640], formal_batch=36, accumulation=1,
        watched_parameter_tensors=18, train_images=5862, val_images=5863,
        formal_batch_capacity_tested=False, **config_identity(path)))


def load_author(reference):
    """Use the pinned author's class bodies, without importing Ultralytics."""
    reference = Path(reference)
    namespace = dict(torch=torch, nn=torch.nn)
    for filename, names in (('conv', ('autopad', 'Conv')), ('block', ('SCDown',))):
        path = reference / (filename + '.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == AUTHOR_HASHES[filename]
        parsed = ast.parse(path.read_text())
        definitions = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                       and node.name in names]
        assert len(definitions) == len(names)
        module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        exec(compile(module, str(path), 'exec'), namespace)
    return namespace['SCDown']


def fidelity(reference, output, fraction, device):
    if device == 'cuda':
        require_memory(fraction)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    author_class = load_author(reference)
    cases = []
    for ci, co, height, width in ((96, 192, 80, 80), (192, 384, 40, 40), (192, 192, 40, 40),
                                   (96, 192, 13, 19), (192, 384, 9, 15), (192, 192, 11, 17)):
        for training in (False, True):
            for amp in ((False, True) if device == 'cuda' else (False,)):
                torch.manual_seed(23)
                ours = SCDown(ci, co).to(device).train(training)
                author = author_class(ci, co, 3, 2).to(device).train(training)
                # The transfer retains RTMDet BN eps/momentum. Align these
                # explicit normalization settings for the numerical comparison.
                for source, target in zip((ours.cv1.bn, ours.cv2.bn), (author.cv1.bn, author.cv2.bn)):
                    target.eps, target.momentum = source.eps, source.momentum
                author.load_state_dict(ours.state_dict(), strict=True)
                x = torch.randn(2, ci, height, width, device=device, requires_grad=True)
                reference_x = x.detach().clone().requires_grad_()
                with torch.autocast(device, enabled=amp):
                    actual, expected = ours(x), author(reference_x)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                names, params = zip(*ours.named_parameters())
                reference_params = dict(author.named_parameters())
                seed = torch.randn_like(actual) / math.sqrt(actual.numel())
                gradients = torch.autograd.grad(actual, (x, *params), seed)
                reference_gradients = torch.autograd.grad(expected,
                    (reference_x, *(reference_params[name] for name in names)), seed)
                for name, a, b in zip(('input', *names), gradients, reference_gradients):
                    assert torch.isfinite(a).all(), name
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=name)
                for name, value in ours.state_dict().items():
                    torch.testing.assert_close(value, author.state_dict()[name], atol=0, rtol=0, msg=name)
                cases.append(dict(in_channels=ci, out_channels=co, input_hw=[height, width],
                    training=training, amp=amp, forward_max_error=float((actual.float() - expected.float()).abs().max()),
                    gradient_max_error=max(float((a.float() - b.float()).abs().max())
                                           for a, b in zip(gradients, reference_gradients))))
                del ours, author, actual, expected, gradients, reference_gradients
    report_to(output, dict(status='passed', method=METHOD, device=device, author_commit=AUTHOR_COMMIT,
        reference=str(reference), source_sha256=AUTHOR_HASHES, cases=cases,
        preserves_project_bn_configuration=True, no_activation_after_dw=True))
    print('FIDELITY_PASSED', len(cases), flush=True)


@HOOKS.register_module()
class SCDownAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(self.watched) == 18 and len(parameters) == len({id(p) for p in parameters})
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
        selected.append(image); categories -= by_image[image]
    ids = [row['id'] for row in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.scdown_probe')
    cfg.custom_hooks.append(dict(type='SCDownAuditHook', method=METHOD,
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


def capacity(cfg, output, fraction, batch, path):
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    batch = batch or 36
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    optim_cfg = deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model)
    initial = {name: p.detach().cpu().clone() for name, p in watched.items()}
    parameters = [p for group in optimizer.optimizer.param_groups for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
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
        elapsed_seconds=time.perf_counter() - started, **config_identity(path)))
    print('CAPACITY_PASSED', batch, flush=True)


def fuse_for_latency(model):
    """Fold scale-specific BN into separate copies of shared head kernels."""
    seen, copied = set(), 0
    for layer in list(model.modules()):
        if isinstance(layer, ConvModule):
            if id(layer.conv) in seen:
                layer.conv = deepcopy(layer.conv)
                copied += 1
            else:
                seen.add(id(layer.conv))
    fuse_conv_bn(model)
    return copied


def latency(cfg, output, fraction):
    require_memory(fraction)
    torch.backends.cudnn.benchmark = False
    original = Config.fromfile(str(ROOT / 'configs/rtmdet_tiny_dior_300e.py'))
    models = {}
    for name, candidate in (('baseline', original), ('scdown', cfg)):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(candidate.model))
        model.init_weights(); models[name] = model.cuda().eval()
    x = torch.randn(1, 3, 640, 640, device='cuda')
    results = {}
    with torch.inference_mode(), torch.autocast('cuda'):
        for fused in (False, True):
            if fused:
                for name, model in models.items():
                    before = model._forward(x)
                    assert fuse_for_latency(model) == 8
                    after = model._forward(x)
                    for aa, bb in zip(before, after):
                        for a, b in zip(aa, bb):
                            torch.testing.assert_close(a, b, atol=.003, rtol=.01)
            for model in models.values():
                for _ in range(20):
                    model._forward(x)
            torch.cuda.synchronize()
            measured = {name: [] for name in models}
            # Alternate AB / BA / AB so shared-GPU scheduling drift does not
            # systematically favor one model. Wall time includes dispatch.
            for round_index in range(3):
                names = ('baseline', 'scdown') if round_index % 2 == 0 else ('scdown', 'baseline')
                for name in names:
                    torch.cuda.synchronize(); started = time.perf_counter()
                    for _ in range(80):
                        models[name]._forward(x)
                    torch.cuda.synchronize()
                    measured[name].append((time.perf_counter() - started) / 80 * 1000)
            results['fused' if fused else 'unfused'] = {
                name: dict(round_mean_ms=values, median_round_mean_ms=statistics.median(values))
                for name, values in measured.items()}
    report_to(output, dict(status='passed', method=METHOD, models=results, input_size=[640, 640],
        batch=1, amp=True, warmup=20, rounds=3, iterations_per_round=80,
        timing='synchronized wall time, includes Python/CUDA dispatch, excludes preprocess and NMS',
        shared_gpu=True, cudnn_benchmark=False, random_weights=True,
        bn_fusion_numerical_check=True, shared_head_convs_copied_before_scale_bn_fusion=8,
        cuda_device=torch.cuda.get_device_name(),
        not_a_training_speed_or_exclusive_gpu_benchmark=True))
    print('LATENCY_PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity', 'latency'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/scdown/rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference-dir', default=str(ROOT / 'work_dirs/diagnostics/parameter_module_research/author_reference'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            fidelity(args.reference_dir, args.report, args.allocator_fraction, args.device)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        elif args.mode == 'capacity':
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.config)
        else:
            latency(cfg, args.report, args.allocator_fraction)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=METHOD, mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
