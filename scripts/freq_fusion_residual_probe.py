"""Isolated checks for post-CSP detail routing; never launch full training.

Reuse stable audit/data helpers without changing the running experiment's
scripts. Model code and config remain independent of the old FreqFusion.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
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
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from custom_models.freq_fusion_residual.operator import SplitFreqFusion

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'freq_fusion_residual'
REFERENCE_SHA256 = '886e2d0068d2c1ce8f054ad12b4337e990a43a64220b29c978c4eec8057c12d7'


def watched_parameters(model):
    prefixes = ('neck.freq_fusions.', 'neck.detail_projections.')
    return {name: value for name, value in model.named_parameters() if name.startswith(prefixes)}


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


def preflight(cfg, output, config_path=None):
    previous = Config.fromfile(str(ROOT / 'configs/freq_fusion/rtmdet_tiny_dior_640.py'))
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg', 'val_evaluator',
                'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                'optim_wrapper', 'param_scheduler', 'default_hooks', 'randomness'):
        assert cfg[key] == previous[key], key
    for key in ('backbone', 'bbox_head', 'train_cfg', 'test_cfg', 'data_preprocessor'):
        assert cfg.model[key] == previous.model[key], key
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.img_scale == (640, 640)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.eval()
    macs, sites, projections, handles = [0], [], [], []

    def count(module, inputs, result):
        macs[0] += result.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    def capture_site(module, inputs, result):
        sites.append(dict(high=list(inputs[0].shape), low=list(inputs[1].shape),
                          detail=list(result[1].shape), upsampled=list(result[2].shape)))

    def capture_projection(module, inputs, result):
        projections.append(dict(input=list(inputs[0].shape), output=list(result.shape),
                                bias=module.bias is not None, kernel=list(module.kernel_size)))

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count))
        if isinstance(layer, SplitFreqFusion):
            handles.append(layer.register_forward_hook(capture_site))
    for layer in model.neck.detail_projections:
        handles.append(layer.register_forward_hook(capture_projection))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert [list(x.shape) for x in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert [x['high'] for x in sites] == [[1, 192, 40, 40], [1, 96, 80, 80]]
    assert [x['low'] for x in sites] == [[1, 192, 20, 20], [1, 96, 40, 40]]
    assert [x['output'] for x in projections] == [[1, 96, 40, 40], [1, 96, 80, 80]]
    priors = model.bbox_head.prior_generator.grid_priors([x.shape[-2:] for x in cls], with_stride=True)
    assert sum(len(x) for x in priors) == 8400
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 5015448
    assert math.isclose(macs[0] / 1e9, 8.28595584, abs_tol=1e-10)
    for loader, expected in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotations = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotations['images']) == expected
    report_to(output, dict(status='passed', method=METHOD, parameters=parameters,
              inference_conv_linear_gmac=macs[0] / 1e9, prediction_points=8400,
              excludes_dynamic_filtering_resampling_norm_and_postprocessing=True,
              projection_parameters=sum(p.numel() for p in model.neck.detail_projections.parameters()),
              extra_projection_gmac=.0884736, fusion_sites=sites, projections=projections,
              watched_parameter_tensors=len(watched_parameters(model)),
              scratch=True, input_size=[640, 640], formal_batch=36, accumulation=1,
              formal_batch_capacity_tested=False, train_images=5862, val_images=5863,
              unchanged_backbone_pan_outputs_head_and_training_protocol=True,
              **config_identity(config_path)))


def fidelity(reference, output, fraction):
    require_memory(fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    path = Path(reference)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location('freq_residual_upstream', str(path))
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    cases = []
    for width, compressed, h, w in ((192, 48, 20, 20), (96, 24, 40, 40), (96, 24, 7, 11)):
        for amp in (False, True):
            torch.manual_seed(19)
            ours = SplitFreqFusion(width, width, compressed).cuda()
            for layer in (ours.content_encoder, ours.content_encoder2):
                nn.init.normal_(layer.weight, std=.06)
            # This public upstream flag exposes the same raw signed detail.
            # Our contribution is the external CSP routing, not a new filter.
            original = upstream.FreqFusion(width, width, compressed_channels=compressed,
                feature_resample=True, feature_resample_group=4, hr_residual=False).cuda()
            original.load_state_dict(ours.state_dict(), strict=True)
            dtype = torch.float16 if amp else torch.float32
            high = torch.randn(2, width, h * 2, w * 2, device='cuda', dtype=dtype, requires_grad=True)
            low = torch.randn(2, width, h, w, device='cuda', dtype=dtype, requires_grad=True)
            ref_high = high.detach().clone().requires_grad_()
            ref_low = low.detach().clone().requires_grad_()
            with torch.autocast('cuda', enabled=amp):
                actual = ours(high, low)
            expected_fp32 = original(ref_high.float(), ref_low.float())
            expected = (expected_fp32[0], expected_fp32[1].to(dtype), expected_fp32[2].to(dtype))
            for first, second in zip(actual, expected):
                torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-4)
            names, parameters = zip(*ours.named_parameters())
            refs = dict(original.named_parameters())
            seeds = [torch.randn_like(value) / math.sqrt(value.numel()) for value in actual]
            grads = torch.autograd.grad(actual, (high, low, *parameters), seeds)
            ref_grads = torch.autograd.grad(expected, (ref_high, ref_low, *(refs[n] for n in names)), seeds)
            for name, first, second in zip(('high_input', 'low_input', *names), grads, ref_grads):
                torch.testing.assert_close(first, second, atol=4e-5, rtol=4e-4, msg=name)
            cases.append(dict(width=width, guidance_width=compressed, lr_size=[h, w], amp=amp,
                forward_max_error=max(float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)),
                gradient_max_error=max(float((a.float() - b.float()).abs().max()) for a, b in zip(grads, ref_grads))))
            del ours, original, actual, expected, expected_fp32, grads, ref_grads
    # A large identity plus small detail exposes cancellation when trying to
    # reconstruct D from an already rounded (H + D). The new API avoids that.
    torch.manual_seed(23)
    ours = SplitFreqFusion(96, 96, 24).cuda()
    original = upstream.FreqFusion(96, 96, compressed_channels=24,
        feature_resample=True, feature_resample_group=4, hr_residual=False).cuda()
    original.load_state_dict(ours.state_dict(), strict=True)
    high = torch.full((1, 96, 14, 22), 1000., device='cuda', dtype=torch.float16)
    high[:, :, 6, 10] += .5
    low = torch.randn(1, 96, 7, 11, device='cuda', dtype=torch.float16)
    with torch.no_grad(), torch.autocast('cuda'):
        detail = ours(high, low)[1]
    with torch.no_grad():
        ref_detail = original(high.float(), low.float())[1]
    torch.testing.assert_close(detail, ref_detail.half(), atol=0, rtol=0)
    recovered = (high.float() + ref_detail).half() - high
    delta = (detail.float() - recovered.float())[:, :, 4:-4, 4:-4]
    lost_count = int(torch.count_nonzero(delta))
    assert lost_count > 0
    report_to(output, dict(status='passed', method=METHOD, reference=str(path),
        reference_sha256=REFERENCE_SHA256, upstream_commit='3fb0c70637a3c194fb74294d3ce4681958b26241',
        reference_hr_residual=False, complete_generators_and_enhanced_initial_fusion=True,
        cases=cases, detail_extracted_before_identity_addition_and_amp_rounding=True,
        cancellation_case=dict(samples_differing_from_rounded_subtraction=lost_count,
                               max_difference=float(delta.abs().max()))))
    print('FIDELITY_PASSED', len(cases), flush=True)


@HOOKS.register_module()
class FreqResidualAuditHook(CandidateAuditHook):
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
    cfg.custom_imports.imports.append('scripts.freq_fusion_residual_probe')
    cfg.custom_hooks.append(dict(type='FreqResidualAuditHook', method=METHOD,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/freq_fusion_residual/rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--reference')
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
            if not args.reference:
                raise ValueError('Fidelity requires the pinned author FreqFusion.py')
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
