"""Independent G4/DSConv checks; never edit or enqueue an existing experiment."""
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
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.scdown import SCDown
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.scdown_probe import config_identity, require_memory
from scripts.scdown_variants_probe import CONTROL_SITES, GROUP_SITES

ROOT = Path(__file__).resolve().parents[1]
SPECS = dict(scdown_g4=dict(parameters=3139572, gmac=6.92717184, watched=39),
             scdown_dsconv=dict(parameters=3015540, gmac=6.80920704, watched=60))


def watched_parameters(model, method):
    if method not in SPECS:
        raise ValueError(method)
    return {name: p for name, p in model.named_parameters()
            if any(name.startswith(site + '.') for site in (*CONTROL_SITES, *GROUP_SITES))}


def preflight(cfg, method, output, path):
    control_cfg = Config.fromfile(str(ROOT / 'configs/scdown/rtmdet_tiny_dior_640.py'))
    reference_dict, candidate_dict = control_cfg.to_dict(), cfg.to_dict()
    for config in (reference_dict, candidate_dict):
        for key in ('custom_imports', 'work_dir'):
            config.pop(key, None)
        for component in ('backbone', 'neck'):
            config['model'][component].pop('type')
    assert candidate_dict == reference_dict, 'Training/evaluation protocol changed'
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    models = []
    for config in (control_cfg, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(config.model))
        model.init_weights(); model.eval(); models.append(model)
    control, model = models
    for name, value in control.state_dict().items():
        if not any(name.startswith(site + '.') for site in GROUP_SITES):
            torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    assert tuple(n for n, m in model.named_modules() if isinstance(m, SCDown)) == CONTROL_SITES
    grouped = tuple(n.removesuffix('.conv') for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Conv2d) and m.groups == 4)
    assert grouped == (GROUP_SITES if method == 'scdown_g4' else ())
    for site in GROUP_SITES:
        layer, original = model.get_submodule(site), control.get_submodule(site)
        width = original.conv.in_channels
        if method == 'scdown_g4':
            assert isinstance(layer, ConvModule) and layer.conv.groups == 4
            assert layer.conv.in_channels == layer.conv.out_channels == width
            parts = (layer,)
        else:
            assert isinstance(layer, DepthwiseSeparableConvModule)
            dw, pw = layer.depthwise_conv.conv, layer.pointwise_conv.conv
            assert dw.kernel_size == (3, 3) and dw.stride == (1, 1) and dw.padding == (1, 1)
            assert dw.in_channels == dw.out_channels == dw.groups == width
            assert pw.kernel_size == (1, 1) and pw.groups == 1
            assert pw.in_channels == pw.out_channels == width
            parts = (layer.depthwise_conv, layer.pointwise_conv)
        for part in parts:
            assert part.norm_cfg == original.norm_cfg and part.act_cfg == original.act_cfg
            assert part.order == ('conv', 'norm', 'act')
    macs, sites, handles = [0], [], []

    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        else:
            macs[0] += output.numel() * module.in_features

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
        if name in GROUP_SITES:
            handles.append(layer.register_forward_hook(lambda mod, inputs, output, name=name:
                sites.append(dict(path=name, input=list(inputs[0].shape), output=list(output.shape)))))
    with torch.no_grad():
        cls, reg = model._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    assert tuple(site['path'] for site in sites) == GROUP_SITES
    expected = [(48, 80), (96, 40), (192, 20), (96, 40), (48, 80), (96, 40), (192, 20)]
    for site, (width, size) in zip(sites, expected):
        assert site['input'] == site['output'] == [1, width, size, size], site
    assert [list(v.shape) for v in cls] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(v.shape) for v in reg] == [[1, 4, s, s] for s in (80, 40, 20)]
    priors = model.bbox_head.prior_generator.grid_priors(
        [v.shape[-2:] for v in cls], device='cpu', with_stride=True)
    assert sum(len(p) for p in priors) == 8400
    spec = SPECS[method]
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == spec['parameters'] and len(watched_parameters(model, method)) == spec['watched']
    assert math.isclose(macs[0] / 1e9, spec['gmac'], abs_tol=1e-10)
    for loader, expected_count in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == expected_count
    report_to(output, dict(status='passed', method=method, parameters=parameters,
        inference_conv_linear_gmac=macs[0] / 1e9,
        parameter_reduction_vs_original_percent=(1 - parameters / 4878708) * 100,
        mac_reduction_vs_original_percent=(1 - macs[0] / 7987703040) * 100,
        actual_sites=sites, scdown_sites=CONTROL_SITES,
        unchanged_initialization_including_control_scdowns=True,
        protocol_equal_to_scdown_control=True, prediction_points=8400,
        scratch=True, input_size=[640, 640], formal_batch=36, accumulation=1,
        watched_parameter_tensors=spec['watched'], train_images=5862, val_images=5863,
        formal_batch_capacity_tested=False, **config_identity(path)))
    print('PREFLIGHT_PASSED', method, parameters, macs[0] / 1e9, flush=True)


@HOOKS.register_module()
class SCDownCompressionAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model, self.method)
        self.initial = {name: p.detach().cpu().clone() for name, p in self.watched.items()}
        parameters = [p for group in runner.optim_wrapper.optimizer.param_groups for p in group['params']]
        assert len(self.watched) == SPECS[self.method]['watched']
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
    cfg.custom_imports.imports.append('scripts.scdown_compression_probe')
    cfg.custom_hooks.append(dict(type='SCDownCompressionAuditHook', method=method,
        report_path=str(output), image_ids=selected, allocator_fraction=fraction))
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


def capacity(cfg, method, output, fraction, batch, path):
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    batch = batch or 36
    data, image_ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    optim_cfg = deepcopy(cfg.optim_wrapper); optim_cfg.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, optim_cfg)
    watched = watched_parameters(model, method)
    assert len(watched) == SPECS[method]['watched']
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
    report_to(output, dict(status='passed', method=method, microbatch=batch, accumulation=1,
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
    print('CAPACITY_PASSED', method, batch, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SPECS), required=True)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'capacity'), required=True)
    parser.add_argument('--config')
    parser.add_argument('--report', required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--allocator-fraction', type=float, default=.22)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    path = args.config or str(ROOT / f'configs/{args.method}/rtmdet_tiny_dior_640.py')
    cfg = Config.fromfile(path)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.method, args.report, path)
        elif args.mode == 'smoke':
            smoke(cfg, args.method, args.report, args.allocator_fraction)
        else:
            capacity(cfg, args.method, args.report, args.allocator_fraction, args.batch, path)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=args.method, mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
