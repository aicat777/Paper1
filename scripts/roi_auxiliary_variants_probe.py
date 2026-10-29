"""Isolated audits of private, cross-level and contrastive ROI variants.

Never starts full training or modifies an existing queue. Short training uses
official tools/train.py and the existing model/optimizer/scaler/EMA auditor.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType

import torch
from torch.nn import functional as F
from torchvision.ops import roi_align
from mmdet.utils import reduce_mean
from mmengine.config import Config
from mmyolo.registry import HOOKS, MODELS
from mmyolo.utils import register_all_modules

from custom_models.instance_semantic.roi import InstanceSemanticLoss
from scripts.candidate_probe import CandidateAuditHook, report_to
from scripts.candidate_capacity import capacity as generic_capacity
from scripts.instance_semantic_gradscale_probe import watched_parameters
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'scdown_p3_g4_semantic_gradscale'
SPECS = dict(
    private=dict(head='PrivateROISemanticRTMDetHead', parameters=3142856,
                 extras=dict(semantic_private_hidden_channels=192)),
    levels=dict(head='AllLevelROISemanticRTMDetHead', parameters=3105704, extras={}),
    contrast=dict(head='ContrastiveROISemanticRTMDetHead', parameters=3127432,
                  extras=dict(semantic_contrast_weight=.1, semantic_contrast_temperature=.2,
                              semantic_contrast_channels=128, semantic_pair_chunk_size=128)))


def total(losses):
    return sum(value for name, value in losses.items() if 'loss' in name)


def build(cfg, device='cpu'):
    torch.manual_seed(42)
    model = MODELS.build(deepcopy(cfg.model))
    model.init_weights()
    return model.to(device)


def source_identity():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and (ROOT / 'custom_models') in path.parents:
                paths.add(path)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def preflight(cfg, method, output, path):
    reference_cfg = Config.fromfile(str(ROOT / 'configs' / REFERENCE / 'rtmdet_tiny_dior_640.py'))
    spec = SPECS[method]
    for key in ('img_scale', 'train_dataloader', 'val_dataloader', 'test_dataloader',
                'train_cfg', 'val_cfg', 'test_cfg', 'val_evaluator', 'test_evaluator',
                'train_pipeline', 'train_pipeline_stage2', 'test_pipeline', 'custom_hooks',
                'default_hooks', 'optim_wrapper', 'param_scheduler', 'randomness', 'env_cfg'):
        assert cfg[key] == reference_cfg[key], key
    expected = deepcopy(reference_cfg.model)
    expected.bbox_head.update(type=spec['head'], **spec['extras'])
    assert cfg.model == expected
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    reference = build(reference_cfg).eval()
    rng = torch.get_rng_state().clone()
    model = build(cfg).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0, msg=name)
    actual_state = model.state_dict()
    assert all(n.startswith('bbox_head.instance_semantic.')
               for n in set(actual_state) - set(reference.state_dict()))
    head = model.bbox_head.head_module
    for level in (1, 2):
        for index in range(len(head.cls_convs[0])):
            assert head.cls_convs[0][index].conv is head.cls_convs[level][index].conv
            assert head.reg_convs[0][index].conv is head.reg_convs[level][index].conv
            assert head.cls_convs[0][index].bn is not head.cls_convs[level][index].bn
    macs, handles, auxiliary_calls = [0], [], []
    def count(module, inputs, result):
        per_output = module.in_features if isinstance(module, torch.nn.Linear) else \
            (module.in_channels // module.groups) * math.prod(module.kernel_size)
        macs[0] += result.numel() * per_output
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(count))
    handles.append(model.bbox_head.instance_semantic.register_forward_hook(
        lambda *args: auxiliary_calls.append(1)))
    with torch.no_grad():
        x = torch.zeros(1, 3, 640, 640)
        actual, wanted = model._forward(x), reference._forward(x)
    for handle in handles: handle.remove()
    for a, b in zip(actual, wanted):
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not auxiliary_calls
    assert sum(p.numel() for p in model.parameters()) == spec['parameters']
    assert math.isclose(macs[0] / 1e9, 6.78524544, abs_tol=1e-10)
    for loader, count_images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == count_images
    report_to(output, dict(status='passed', method='scdown_p3_g4_semantic_' + method,
        reference=REFERENCE, parameters=spec['parameters'], inference_parameters=3103764,
        training_only_parameters=spec['parameters']-3103764,
        inference_conv_linear_gmac=macs[0] / 1e9, auxiliary_executions_in_inference=0,
        unchanged_detector_initialization=True, unchanged_rng_state=True,
        exact_inference_match=True, shared_tower_convs_and_independent_bn_preserved=True,
        all_gt=True, semantic_weight=.1, shared_feature_gradient_scale=.25,
        normalization='all valid GT; additionally divide by 3 for levels',
        roi_source='neck outputs' if method == 'private' else 'classification tower outputs',
        train_images=5862, val_images=5863, formal_batch=36, accumulation=1,
        input_size=[640, 640], scratch=True, training_protocol_unchanged=True,
        formal_batch_capacity_tested=False, source_code_sha256=source_identity(),
        **config_identity(path)))


def reference_forward(method):
    def forward(module, features, rows):
        with torch.autocast('cuda', enabled=False):
            rows = rows[(rows[:, 4:6] > rows[:, 2:4]).all(1)]
            boxes = torch.cat((rows[:, :1], rows[:, 2:]), 1)
            denominator = reduce_mean(rows.new_tensor(float(len(rows)))).clamp(min=1)
            if method == 'levels':
                losses = []
                for feature, stride in zip(features, module.strides):
                    adjusted = boxes.clone(); adjusted[:, 1:] = adjusted[:, 1:] / stride + .5
                    vector = roi_align(feature.float(), adjusted, 3, sampling_ratio=2,
                                       aligned=True).mean((-2, -1))
                    vector = vector.detach() + .25 * (vector - vector.detach())
                    losses.append(F.cross_entropy(module.classifier(vector), rows[:, 1].long(),
                                                   reduction='sum') * .1 / denominator)
                return sum(losses) / len(losses)
            vector = InstanceSemanticLoss.pool_instances(module, features, boxes)
            vector = vector.detach() + .25 * (vector - vector.detach())
            if method == 'private':
                return .1 * F.cross_entropy(module.classifier(module.encoder(vector)),
                                            rows[:, 1].long(), reduction='sum') / denominator
            labels = rows[:, 1].long()
            ce = .1 * F.cross_entropy(module.classifier(vector), labels, reduction='sum') / denominator
            embedding = F.normalize(module.projector(vector), dim=1, eps=1e-12)
            contrast = embedding.sum() * 0
            for i in range(len(rows)):
                candidates = [j for j in range(len(rows)) if j != i]
                positives = [j for j in candidates if labels[i] == labels[j]]
                if positives:
                    scores = torch.stack([(embedding[i] * embedding[j]).sum() / .2
                                          for j in candidates])
                    correct = torch.stack([(embedding[i] * embedding[j]).sum() / .2
                                           for j in positives])
                    contrast = contrast + torch.logsumexp(scores, 0) - correct.mean()
            return dict(loss_instance_semantic=ce, loss_instance_contrast=.1*contrast/denominator,
                        semantic_all_gt=rows.new_tensor(float(len(rows))),
                        semantic_contrast_anchors=rows.new_tensor(3.))
    return forward


def fidelity(cfg, method, output, fraction):
    require_memory(fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    for amp, multiplier in ((False, 1.), (False, 32.), (True, 32.), (True, 128.)):
        detector = build(cfg, 'cuda')
        module = detector.bbox_head.instance_semantic
        reference = deepcopy(module)
        reference.forward = MethodType(reference_forward(method), reference)
        del detector
        torch.manual_seed(17)
        dtype = torch.float16 if amp else torch.float32
        features = [torch.randn(2, 96, side, side, device='cuda', dtype=dtype,
                                requires_grad=True) for side in (80, 40, 20)]
        copies = [f.detach().clone().requires_grad_() for f in features]
        rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
            [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
            [0, 3, 8, 8, 600, 600], [1, 0, 4.25, 5.5, 5.25, 6.5]],
            device='cuda', dtype=torch.float32)
        with torch.autocast('cuda', enabled=amp):
            actual, wanted = module(features, rows), reference(copies, rows)
            if not isinstance(actual, dict):
                actual, wanted = dict(loss_instance_semantic=actual), dict(loss_instance_semantic=wanted)
            for key in actual:
                torch.testing.assert_close(actual[key], wanted[key], atol=4e-7, rtol=3e-6, msg=key)
        # Match BaseModel.train_step: only forward runs in optim_context;
        # AmpOptimWrapper.update_params/backward is OUTSIDE autocast.
        (total(actual) * multiplier).backward()
        (total(wanted) * multiplier).backward()
        parameters = list(module.named_parameters())
        ref_parameters = list(reference.named_parameters())
        actual_values = [('feature_'+str(i), f) for i, f in enumerate(features)] + parameters
        expected_values = copies + [p for n, p in ref_parameters]
        max_error, squared_error, squared_reference = 0., 0., 0.
        for (name, parameter), expected in zip(actual_values, expected_values):
            assert parameter.grad is not None and expected.grad is not None, name
            assert parameter.grad.isfinite().all() and expected.grad.isfinite().all(), name
            difference = (parameter.grad.float() - expected.grad.float()) / multiplier
            max_error = max(max_error, float(difference.abs().max()))
            squared_error += float(difference.double().square().sum())
            squared_reference += float((expected.grad.double() / multiplier).square().sum())
            torch.testing.assert_close(parameter.grad.float() / multiplier,
                                       expected.grad.float() / multiplier,
                                       atol=3e-7, rtol=3e-3 if amp else 3e-5, msg=name)
        relative = math.sqrt(squared_error / max(squared_reference, 1e-30))
        record = dict(amp=amp, outer_loss_multiplier=multiplier,
                      maximum_unscaled_gradient_error=max_error, gradient_relative_l2_error=relative)
        records.append(record); print(json.dumps(record), flush=True)
        assert relative < 5e-4
        del module, reference, actual, wanted, features, copies, rows, actual_values, expected_values
        gc.collect(); torch.cuda.empty_cache()
    report_to(output, dict(status='passed', method=method,
        scope='All auxiliary inputs/parameters; actual 96-channel 80/40/20 maps; full-model routes and main gradients checked separately',
        independent_reference='Original ROI geometry; per-level CE or scalar per-anchor contrast; ordinary autograd',
        cases=records, saved_checkpoints=False, source_code_sha256=source_identity()))


@HOOKS.register_module()
class ROIAuxiliaryVariantsAuditHook(CandidateAuditHook):
    def __init__(self, actual_method, **kwargs):
        super().__init__(method='instance_semantic', **kwargs)
        self.actual_method = actual_method

    def before_run(self, runner):
        require_memory(self.fraction)

    def before_train(self, runner):
        super().before_train(runner)
        additional = {n: p for n, p in watched_parameters(self.model).items() if n not in self.watched}
        self.watched.update(additional)
        self.initial.update({n: p.detach().cpu().clone() for n, p in additional.items()})
        for name, p in additional.items():
            def capture(g, name=name):
                assert torch.isfinite(g).all(), name
                self.gradients[name] = max(self.gradients.get(name, 0), float(g.abs().max()))
            p.register_hook(capture)
        self.calls = {key: 0 for key in ('backbone', 'neck', 'tower', 'assignment', 'roi')}
        self.coverage, self.classifier_rows, self.handles = [], [], []
        for key, module in [('backbone', self.model.backbone), ('neck', self.model.neck),
                            ('tower', self.model.bbox_head.head_module),
                            ('assignment', self.model.bbox_head.assigner)]:
            def count(*args, key=key):
                if self.model.training: self.calls[key] += 1
            self.handles.append(module.register_forward_hook(count))
        def record_roi(module, args):
            rows = args[1]; self.calls['roi'] += 1
            self.coverage.append(int((rows[:, 4:6] > rows[:, 2:4]).all(1).sum()))
        self.handles.append(self.model.bbox_head.instance_semantic.register_forward_pre_hook(record_roi))
        self.handles.append(self.model.bbox_head.instance_semantic.classifier.register_forward_pre_hook(
            lambda module, args: self.classifier_rows.append(len(args[0]))))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.bn_before = {n: int(m.num_batches_tracked) for n, m in self.model.named_modules()
                          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        for name, module in self.model.named_modules():
            if name in self.bn_before:
                assert int(module.num_batches_tracked) == self.bn_before[name]+1, name

    def after_train(self, runner):
        super().after_train(runner)
        for handle in self.handles: handle.remove()
        assert set(self.calls.values()) == {16}, self.calls
        factor = 3 if self.actual_method == 'levels' else 1
        assert self.classifier_rows == [factor*n for n in self.coverage]
        assert min(self.coverage) > 0
        if self.actual_method == 'contrast':
            assert any(x['loss_instance_contrast'] > 0 for x in self.losses)
        report = json.loads(Path(self.report_path).read_text())
        report.update(method='scdown_p3_g4_semantic_'+self.actual_method,
                      all_gt_coverage=self.coverage, classifier_rows=self.classifier_rows,
                      forward_counts=self.calls, one_bn_update_per_train_step=True,
                      source_code_sha256=source_identity())
        report_to(self.report_path, report)


def smoke(cfg, method, output, fraction):
    require_memory(fraction)
    directory = Path(output).parent / 'smoke'
    if directory.exists(): raise RuntimeError('Existing smoke outputs would be overwritten')
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
    categories = {r['id'] for r in annotation['categories']}
    by_image = {r['id']: set() for r in annotation['images']}
    for row in annotation['annotations']:
        if not row.get('iscrowd', 0): by_image[row['image_id']].add(row['category_id'])
    selected = []
    while categories:
        image = max(by_image, key=lambda k: len(by_image[k] & categories))
        assert by_image[image] & categories
        selected.append(image); categories -= by_image[image]
    image_ids = [r['id'] for r in annotation['images']]
    for image in image_ids:
        if len(selected) == 32: break
        if image not in selected: selected.append(image)
    cfg.val_dataloader.dataset.indices = [image_ids.index(image) for image in selected]
    cfg.custom_imports.imports.append('scripts.roi_auxiliary_variants_probe')
    cfg.custom_hooks.append(dict(type='ROIAuxiliaryVariantsAuditHook', actual_method=method,
        report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (method+'_smoke.py'); cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT/'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    if result.returncode: raise RuntimeError(f'Official short training failed: {result.returncode}')
    assert json.loads(Path(output).read_text())['status'] == 'passed'
    assert not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SPECS), required=True)
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--config')
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL); torch.set_num_threads(2)
    path = args.config or str(ROOT/'configs'/('scdown_p3_g4_semantic_'+args.method)/'rtmdet_tiny_dior_640.py')
    cfg = Config.fromfile(path)
    try:
        if args.mode == 'preflight': preflight(cfg, args.method, args.report, path)
        elif args.mode == 'fidelity': fidelity(cfg, args.method, args.report, args.allocator_fraction)
        elif args.mode == 'smoke': smoke(cfg, args.method, args.report, args.allocator_fraction)
        else:
            generic_capacity(cfg, 'instance_semantic', args.report, args.allocator_fraction,
                             batch=args.batch, config_path=path)
            report = json.loads(Path(args.report).read_text())
            report.update(method='scdown_p3_g4_semantic_'+args.method,
                          source_code_sha256=source_identity())
            report_to(args.report, report)
    except Exception as error:
        report_to(args.report, dict(status='failed', method=args.method, mode=args.mode,
            error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
