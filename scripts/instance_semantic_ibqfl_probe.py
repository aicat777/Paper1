"""Isolated IBQFL and GSCE checks; never launch full training or modify queues."""
import argparse
from copy import deepcopy
import gc
import json
import logging
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from mmdet.models.losses import QualityFocalLoss
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic_ibqfl import InstanceBalancedQFLLoss
from custom_models.scdown import SCDown
from scripts.candidate_capacity import capacity_data
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'instance_semantic_ibqfl'
COMBINED = 'scdown_p3_g4_semantic_ibqfl'


def identity(cfg):
    combined = cfg.model.backbone.type == 'SCDownP3G4CSPNeXt'
    return (COMBINED, 'scdown_p3_g4_semantic', 3105704, 6.78524544) if combined else (
        METHOD, 'instance_semantic', 4880648, 7.98770304)


def watched_parameters(model):
    prefixes = ['bbox_head.head_module.rtm_cls.', 'bbox_head.head_module.cls_convs.',
                'bbox_head.instance_semantic.']
    for name, layer in model.named_modules():
        if isinstance(layer, SCDown):
            prefixes.append(name + '.')
        elif isinstance(layer, torch.nn.Conv2d) and layer.groups == 4 and layer.kernel_size == (3, 3):
            prefixes.append(name.rpartition('.')[0] + '.')
    return {n: p for n, p in model.named_parameters() if n.startswith(tuple(prefixes))}


def preflight(cfg, output, path):
    method, ref, total, expected_gmac = identity(cfg)
    original = Config.fromfile(str(ROOT / 'configs' / ref / 'rtmdet_tiny_dior_640.py'))
    for key in ('img_scale', 'train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator', 'custom_hooks',
                'train_pipeline', 'train_pipeline_stage2', 'optim_wrapper', 'param_scheduler',
                'default_hooks', 'randomness', 'env_cfg'):
        assert cfg[key] == original[key], key
    expected = deepcopy(original.model)
    expected.bbox_head.update(type='InstanceBalancedSemanticRTMDetHead', instance_qfl_weight=.1)
    expected.train_cfg.assigner.type = 'ROISemanticIndexedAssigner'
    assert cfg.model == expected
    assert cfg.load_from is None and not cfg.resume and cfg.model.backbone.init_cfg is None
    assert cfg.img_scale == (640, 640) and cfg.train_dataloader.batch_size == 36
    assert cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    models = []
    for candidate in (original, cfg):
        torch.manual_seed(42)
        model = MODELS.build(deepcopy(candidate.model))
        model.init_weights(); model.eval()
        models.append(model)
    reference, model = models
    assert set(reference.state_dict()) == set(model.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    macs, handles = [0], []

    def count(layer, inputs, result):
        per_output = (layer.in_channels // layer.groups) * math.prod(layer.kernel_size) \
            if isinstance(layer, torch.nn.Conv2d) else layer.in_features
        macs[0] += result.numel() * per_output

    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(layer.register_forward_hook(count))
    with torch.no_grad():
        actual = model._forward(torch.zeros(1, 3, 640, 640))
        wanted = reference._forward(torch.zeros(1, 3, 640, 640))
    for handle in handles:
        handle.remove()
    for a, b in zip(actual, wanted):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert [list(x.shape) for x in actual[0]] == [[1, 20, s, s] for s in (80, 40, 20)]
    assert [list(x.shape) for x in actual[1]] == [[1, 4, s, s] for s in (80, 40, 20)]
    assert sum(p.numel() for p in model.parameters()) == total
    assert not list(model.bbox_head.instance_qfl.parameters())
    assert math.isclose(macs[0] / 1e9, expected_gmac, abs_tol=1e-10)
    scdown = [n for n, m in model.named_modules() if isinstance(m, SCDown)]
    g4 = [n for n, m in model.named_modules()
          if isinstance(m, torch.nn.Conv2d) and m.groups == 4 and m.kernel_size == (3, 3)]
    assert len(scdown) == (4 if method == COMBINED else 0)
    assert len(g4) == (7 if method == COMBINED else 0)
    for loader, images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == images
    report_to(output, dict(status='passed', method=method, reference=ref, parameters=total,
        inference_parameters=total - 1940, new_parameters=0, inference_conv_linear_gmac=expected_gmac,
        state_keys_and_initialization_match_reference=True, inference_matches_reference_exactly=True,
        original_roi_preserved=True, official_main_losses_and_matching_unchanged=True,
        semantic_weight=.1, instance_qfl_weight=.1, all_valid_gt_denominator=True,
        train_images=5862, val_images=5863, input_size=[640, 640], formal_batch=36,
        accumulation=1, scratch=True, unchanged_training_protocol=True,
        scdown_sites=scdown, g4_sites=g4, watched_parameter_tensors=len(watched_parameters(model)),
        formal_batch_capacity_tested=False, **config_identity(path)))


def fidelity(output, fraction):
    require_memory(fraction)
    cases = []
    for half in (False, True):
        for beta in (1.5, 2.):
            torch.manual_seed(431)
            dtype = torch.float16 if half else torch.float32
            scores = [torch.randn(2, 20, h, w, device='cuda', dtype=dtype, requires_grad=True)
                      for h, w in [(2, 3), (1, 2), (1, 1)]]
            copies = [x.detach().clone().requires_grad_() for x in scores]
            valid = torch.tensor([[True, True, True], [True, False, False]], device='cuda')
            indices = torch.tensor([[0, 0, 1, -1, 0, -1, 1, 1, 0],
                                    [0, -1, -1, 0, -1, -1, -1, -1, -1]], device='cuda')
            labels = torch.where(indices >= 0, indices, 20)
            quality = torch.rand(2, 9, device='cuda')
            quality[0, 0] = 0
            assignment = dict(assigned_gt_indices=indices, assigned_labels=labels, assign_metrics=quality)
            qfl = QualityFocalLoss(beta=beta)
            with torch.autocast('cuda', dtype=torch.float16, enabled=half):
                actual, stats = InstanceBalancedQFLLoss()(scores, assignment, valid, qfl)
                grouped = {}
                offset = 0
                for score in copies:
                    h, w = score.shape[-2:]
                    for b in range(2):
                        for y in range(h):
                            for x in range(w):
                                p = offset + y * w + x
                                slot = int(indices[b, p])
                                if slot < 0 or not bool(valid[b, slot]):
                                    continue
                                z = score[b, :, y, x]
                                # Independent scalar QFL definition; positive
                                # IoU and negative-zero targets remain distinct.
                                prob = z.sigmoid()
                                components = F.binary_cross_entropy_with_logits(
                                    z, torch.zeros_like(z), reduction='none') * prob.pow(beta)
                                label, q = int(labels[b, p]), quality[b, p]
                                components[label] = F.binary_cross_entropy_with_logits(
                                    z[label], q, reduction='none') * (q - prob[label]).abs().pow(beta)
                                grouped.setdefault((b, slot), []).append(components.sum())
                    offset += h * w
                expected = .1 * sum(torch.stack(x).mean() for x in grouped.values()) / valid.sum()
            torch.testing.assert_close(actual, expected, atol=2e-5 if half else 2e-7, rtol=2e-4)
            ga, gb = torch.autograd.grad(actual, scores), torch.autograd.grad(expected, copies)
            error = max(float((a.float() - b.float()).abs().max()) for a, b in zip(ga, gb))
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, atol=2e-5 if half else 2e-7, rtol=2e-3 if half else 2e-5)
            cases.append(dict(amp=half, beta=beta, loss_max_error=float((actual - expected).abs()),
                              gradient_max_error=error, valid_gt=int(stats['ibqfl_all_gt'])))
    report_to(output, dict(status='passed', independent_formula=True,
                           cross_level_image_gt_grouping=True, includes_zero_iou_positive=True, cases=cases))


@HOOKS.register_module()
class InstanceBalancedQFLAuditHook(CandidateAuditHook):
    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        self.model = runner.model
        self.watched = watched_parameters(self.model)
        self.initial = {n: p.detach().cpu().clone() for n, p in self.watched.items()}
        params = [p for g in runner.optim_wrapper.optimizer.param_groups for p in g['params']]
        assert len(params) == len({id(p) for p in params})
        assert {id(p) for p in self.watched.values()} <= {id(p) for p in params}
        for name, p in self.watched.items():
            def capture(g, name=name):
                assert g.isfinite().all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(g.abs().max()))
            p.register_hook(capture)
        self.calls = dict(head=0, assignment=0, roi=0, balanced=0)

        def count(key):
            def hook(*args):
                if self.model.training:
                    self.calls[key] += 1
            return hook

        head = self.model.bbox_head
        self.handles = [head.head_module.register_forward_hook(count('head')),
                        head.assigner.register_forward_hook(count('assignment')),
                        head.instance_semantic.register_forward_hook(count('roi')),
                        head.instance_qfl.register_forward_hook(count('balanced'))]
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.bn_before = {n: int(m.num_batches_tracked) for n, m in self.model.named_modules()
                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        assert outputs['ibqfl_positive_points'] > 0 and outputs['loss_instance_qfl'] > 0
        assert outputs['ibqfl_matched_gt'] <= outputs['ibqfl_all_gt']
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                assert int(m.num_batches_tracked) == self.bn_before[n] + 1, n

    def after_train(self, runner):
        super().after_train(runner)
        for hook in self.handles:
            hook.remove()
        assert set(self.calls.values()) == {16}, self.calls
        report = json.loads(Path(self.report_path).read_text())
        report.update(training_forward_counts=self.calls, one_bn_update_per_train_step=True,
                      original_all_gt_roi_preserved=True, qfl_active_all_short_test_steps=True)
        report_to(self.report_path, report)


def smoke(cfg, output, fraction):
    method, _, _, _ = identity(cfg)
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
    remaining = {x['id'] for x in annotation['categories']}
    by_image = {x['id']: set() for x in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0):
            by_image[row['image_id']].add(row['category_id'])
    selected = []
    while remaining:
        image = max(by_image, key=lambda x: len(by_image[x] & remaining))
        assert by_image[image] & remaining
        selected.append(image); remaining -= by_image[image]
    ids = [x['id'] for x in annotation['images']]
    for image in ids:
        if len(selected) == 32:
            break
        if image not in selected:
            selected.append(image)
    cfg.val_dataloader.dataset.indices = [ids.index(x) for x in selected]
    cfg.custom_imports.imports.append('scripts.instance_semantic_ibqfl_probe')
    cfg.custom_hooks.append(dict(type='InstanceBalancedQFLAuditHook', method=method,
                                report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (method + '_smoke.py')
    cfg.dump(str(snapshot))
    done = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                          cwd=ROOT, stdin=subprocess.DEVNULL)
    if done.returncode:
        raise RuntimeError(f'Official short training failed: {done.returncode}')
    report = json.loads(Path(output).read_text())
    assert report['status'] == 'passed' and not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)
    print('SMOKE_PASSED', method, flush=True)


def capacity(cfg, output, fraction, batch, path):
    method, _, _, _ = identity(cfg)
    free, total = require_memory(fraction)
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    torch.backends.cudnn.benchmark = cfg.env_cfg.cudnn_benchmark
    batch = batch or 36
    data, ids = capacity_data(cfg, batch)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights(); model.cuda().train()
    opts = deepcopy(cfg.optim_wrapper)
    opts.loss_scale = dict(init_scale=32)
    optimizer = build_optim_wrapper(model, opts)
    watched = watched_parameters(model)
    initial = {n: p.detach().cpu().clone() for n, p in watched.items()}
    grads = {}
    for name, p in watched.items():
        def capture(g, name=name):
            assert g.isfinite().all(), name
            grads[name] = max(grads.get(name, 0), float(g.abs().max()))
        p.register_hook(capture)
    torch.cuda.reset_peak_memory_stats()
    started, losses = time.perf_counter(), []
    for _ in range(3):
        values = model.train_step(data, optimizer)
        assert all(x.isfinite().all() for x in values.values())
        assert values['loss_instance_qfl'] > 0
        losses.append({k: float(v) for k, v in values.items()})
        torch.cuda.synchronize()
    updates = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in watched.items()}
    assert all(grads.get(n, 0) > 0 and updates[n] > 0 for n in watched)
    assert all(int(optimizer.optimizer.state[p]['step']) == 3 for p in watched.values())
    assert optimizer.loss_scaler.get_scale() == 32
    report_to(output, dict(status='passed', method=method, microbatch=batch, formal_batch=36,
        accumulation=1, optimizer_updates=3, gt_per_image=1200, different_image_ids=ids,
        repeated_annotations=True, synthetic_multiscale_boxes=True, formal_batch_capacity_tested=batch == 36,
        allocator_fraction=fraction, loss_scale_init=32, warmup_enabled=False, amp_skipped_updates=0,
        saved_checkpoints=False, watched_parameter_tensors=len(watched), gradient_max=grads,
        parameter_updates_max=updates, losses=losses, cuda_device=torch.cuda.get_device_name(),
        cuda_total_gib=total / 2**30, initial_free_gib=free / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        elapsed_seconds=time.perf_counter() - started, **config_identity(path)))


def gradient_probe(cfg, output):
    method, ref, _, _ = identity(cfg)
    checkpoint = ROOT / 'work_dirs' / (ref + '_rtmdet_tiny_dior_640_gpu1_bs36') / 'epoch_300.pth'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    raw = {k[7:]: v for k, v in saved['ema_state_dict'].items() if k.startswith('module.')}
    assert raw and set(raw) == set(saved['state_dict'])
    model = MODELS.build(deepcopy(cfg.model))
    model.load_state_dict(raw, strict=True)
    model.cpu().train()
    del raw, saved
    random.seed(431); np.random.seed(431); torch.manual_seed(431)
    loader_cfg = deepcopy(cfg.train_dataloader)
    loader_cfg.update(batch_size=36, num_workers=0, persistent_workers=False,
                      sampler=dict(type='DefaultSampler', shuffle=False))
    loader_cfg.dataset.indices = np.random.default_rng(431).choice(5862, 108, replace=False).tolist()
    loader = Runner.build_dataloader(loader_cfg, seed=431)
    head = model.bbox_head
    params = tuple(p for n, p in head.head_module.named_parameters() if n.startswith('cls_convs.'))
    records = []

    def compare(a, b):
        aa = sum(x.double().square().sum() for x in a)
        bb = sum(x.double().square().sum() for x in b)
        ab = sum((x.double() * y.double()).sum() for x, y in zip(a, b))
        return dict(cosine=float(ab / (aa * bb).sqrt().clamp_min(1e-30)),
                    added_over_main_norm=float((bb / aa.clamp_min(1e-30)).sqrt()))

    for index, batch in enumerate(loader):
        batch = model.data_preprocessor(batch, training=True)
        if index == 0:
            continue
        with torch.no_grad():
            neck = model.extract_feat(batch['inputs'])
        neck = tuple(x.detach().requires_grad_() for x in neck)
        losses = head.loss(neck, batch['data_samples'])
        targets = (*neck, *params)
        gradients = {}
        for key, loss in [('main', losses['loss_cls'] + losses['loss_bbox']),
                          ('roi', losses['loss_instance_semantic']), ('extra', losses['loss_instance_qfl'])]:
            values = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
            gradients[key] = tuple(torch.zeros_like(p) if g is None else g
                                   for p, g in zip(targets, values))
        comparisons = {}
        for name, slots in [('neck_outputs', slice(0, 3)), ('classification_tower_parameters', slice(3, None))]:
            gm, gr, ge = (gradients[k][slots] for k in ('main', 'roi', 'extra'))
            comparisons[name] = dict(main_vs_extra=compare(gm, ge), roi_vs_extra=compare(gr, ge))
        records.append(dict(batch=index, losses={k: float(v.detach()) for k, v in losses.items()},
                            comparisons=comparisons))
        print(json.dumps(dict(method=method, **records[-1])), flush=True)
        del neck, losses, gradients, targets, batch, gm, gr, ge, loss, values
        gc.collect()
    report_to(output, dict(status='passed', method=method, raw_reference_checkpoint=str(checkpoint),
        scope='Two strong-augmented B36 batches, frozen raw epoch300 reference weights, CPU FP32 '
              'batch BN. Backbone/neck computed without gradients; probe measured shared neck '
              'output and classification-tower parameter gradients. No updates or checkpoints.',
        formal_training_still_scratch=True, records=records,
        limitations=['Endpoint gradient diagnostic is not full-training or AP evidence.',
                     'Pointwise nonnegative QFL scaling does not force alignment after shared-parameter aggregation.']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity', 'gradients'), required=True)
    parser.add_argument('--config', default=str(ROOT / f'configs/{METHOD}/rtmdet_tiny_dior_640.py'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--allocator-fraction', type=float, default=.28)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            fidelity(args.report, args.allocator_fraction)
        elif args.mode == 'smoke':
            smoke(cfg, args.report, args.allocator_fraction)
        elif args.mode == 'capacity':
            capacity(cfg, args.report, args.allocator_fraction, args.batch, args.config)
        else:
            gradient_probe(cfg, args.report)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=identity(cfg)[0], mode=args.mode,
                                   error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
