"""Validate the independent alpha=1 classification-only ROI config."""
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

import torch
from mmengine.config import Config
from mmyolo.registry import HOOKS
from mmyolo.utils import register_all_modules

from scripts.candidate_probe import report_to
from scripts.instance_semantic_neckhead_probe import build, ROINeckHeadAuditHook
from scripts.instance_semantic_clstower_probe import capacity as existing_capacity
from scripts.scdown_probe import config_identity, require_memory

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'scdown_p3_g4_semantic_clstower_alpha1'
PREVIOUS = 'scdown_p3_g4_semantic_clstower'


def identity():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and ROOT / 'custom_models' in path.parents:
                paths.add(path)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def preflight(cfg, output, path):
    previous = Config.fromfile(str(ROOT / 'configs' / PREVIOUS / 'rtmdet_tiny_dior_640.py'))
    for key in previous:
        if key not in ('model', 'work_dir', 'custom_imports'):
            assert cfg[key] == previous[key], key
    assert cfg.custom_imports == previous.custom_imports
    expected = deepcopy(previous.model)
    expected.bbox_head.semantic_feature_gradient_scale = 1.
    assert cfg.model == expected
    assert cfg.model.backbone.init_cfg is None and cfg.load_from is None and not cfg.resume
    assert cfg.train_dataloader.batch_size == 36 and cfg.optim_wrapper.get('accumulative_counts', 1) == 1
    assert cfg.train_cfg.max_epochs == 300 and cfg.train_cfg.val_interval == 10
    model, old = build(cfg).eval(), build(previous).eval()
    assert model.bbox_head.instance_semantic.feature_gradient_scale == 1.
    assert model.bbox_head.instance_semantic.loss_weight == .1
    assert sum(p.numel() for p in model.parameters()) == 3105704
    for name, value in old.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    selected = {id(p) for p in model.roi_gradient_parameters()}
    head = model.bbox_head.head_module
    wanted = {id(p) for p in head.cls_convs.parameters()} | {id(p) for p in model.bbox_head.instance_semantic.parameters()}
    assert selected == wanted and len(selected) == 16
    assert not selected & {id(p) for p in (*model.backbone.parameters(), *model.neck.parameters(),
        *head.reg_convs.parameters(), *head.rtm_cls.parameters(), *head.rtm_reg.parameters())}
    for level in (1, 2):
        for index in range(len(head.cls_convs[0])):
            assert head.cls_convs[0][index].conv is head.cls_convs[level][index].conv
            assert head.cls_convs[0][index].bn is not head.cls_convs[level][index].bn
    macs, aux_calls, handles = [0], [], []
    def count(module, inputs, result):
        size = module.in_features if isinstance(module, torch.nn.Linear) else (
            module.in_channels // module.groups) * math.prod(module.kernel_size)
        macs[0] += result.numel() * size
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            handles.append(module.register_forward_hook(count))
    handles.append(model.bbox_head.instance_semantic.register_forward_hook(lambda *args: aux_calls.append(1)))
    with torch.no_grad():
        images = torch.zeros(1, 3, 640, 640)
        for a, b in zip(model._forward(images), old._forward(images)):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
    for handle in handles:
        handle.remove()
    assert not aux_calls and math.isclose(macs[0] / 1e9, 6.78524544, abs_tol=1e-10)
    for loader, count_images in ((cfg.train_dataloader, 5862), (cfg.val_dataloader, 5863)):
        annotation = json.loads((Path(cfg.data_root) / loader.dataset.ann_file).read_text())
        assert len(annotation['images']) == count_images
    report_to(output, dict(status='passed', method=METHOD, reference=PREVIOUS,
        only_model_config_change='semantic_feature_gradient_scale: .25 -> 1',
        feature_gradient_scale=1., backbone_roi_gradient_scale=0., neck_roi_gradient_scale=0.,
        classification_roi_gradient_scale=1., auxiliary_classifier_weight=.1,
        auxiliary_parameter_tensors=16, parameters=3105704, inference_parameters=3103764,
        inference_conv_linear_gmac=macs[0] / 1e9, exact_initialization_and_inference=True,
        auxiliary_executions_in_inference=0, formal_batch=36, accumulation=1,
        scratch=True, training_protocol_unchanged=True, source_code_sha256=identity(), **config_identity(path)))


def fidelity(cfg, output, fraction):
    require_memory(fraction)
    reference_cfg = Config.fromfile(str(ROOT / 'configs/scdown_p3_g4_semantic_gradscale/rtmdet_tiny_dior_640.py'))
    reference_cfg.model.bbox_head.semantic_feature_gradient_scale = 1.
    cases = []
    for amp, multiplier in ((False, 1.), (False, 32.), (True, 32.), (True, 128.)):
        reference, model = build(reference_cfg, 'cuda').train(), build(cfg, 'cuda').train()
        torch.manual_seed(17)
        images = torch.randn(2, 3, 256, 256, device='cuda')
        rows = torch.tensor([[1, 1, 8, 8, 26, 28], [0, 1, 10, 12, 30, 36],
            [1, 2, 12, 16, 152, 168], [0, 1, 10, 12, 30, 36],
            [0, 3, 8, 8, 248, 248], [1, 4, 4.25, 5.5, 5.25, 6.5]], device='cuda', dtype=torch.float32)
        samples = dict(bboxes_labels=rows, img_metas=[dict(img_shape=(256, 256))] * 2)
        with torch.autocast('cuda', enabled=amp):
            expected = reference.loss(images, samples)
        parameters = tuple(reference.parameters())
        gm = torch.autograd.grad(multiplier * (expected['loss_cls'] + expected['loss_bbox']),
            parameters, allow_unused=True, retain_graph=True)
        ga = torch.autograd.grad(multiplier * expected['loss_instance_semantic'], parameters, allow_unused=True)
        with torch.autocast('cuda', enabled=amp):
            actual = model.loss(images, samples)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        (multiplier * sum(actual.values())).backward()
        maximum = 0.
        for (name, p), main, aux in zip(model.named_parameters(), gm, ga):
            selected = name.startswith(('bbox_head.head_module.cls_convs.', 'bbox_head.instance_semantic.'))
            wanted = main if not selected or aux is None else (aux if main is None else main + aux)
            if wanted is None:
                assert p.grad is None or not p.grad.any(), name
            else:
                torch.testing.assert_close(p.grad / multiplier, wanted / multiplier,
                                           atol=5e-6, rtol=3e-4, msg=name)
                maximum = max(maximum, float(((p.grad - wanted) / multiplier).abs().max()))
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=amp):
            aux_loss = model.loss(images, samples)['loss_instance_semantic']
        (multiplier * aux_loss).backward()
        allowed = {id(p) for p in model.roi_gradient_parameters()}
        assert all(p.grad is None or not p.grad.any() for p in model.parameters() if id(p) not in allowed)
        assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in model.roi_gradient_parameters())
        cases.append(dict(amp=amp, outer_loss_multiplier=multiplier,
            maximum_unscaled_gradient_error=maximum, auxiliary_outside_classification_exactly_zero=True))
        del reference, model, parameters, expected, actual, gm, ga, images, samples, rows, aux_loss
        gc.collect(); torch.cuda.empty_cache()
    report_to(output, dict(status='passed', method=METHOD, cases=cases,
        reference='Separate main/ROI autograd at alpha=1, retain ROI derivatives only on cls tower and auxiliary classifier',
        source_code_sha256=identity(), saved_checkpoints=False))


@HOOKS.register_module()
class ROIClassificationAlphaOneAuditHook(ROINeckHeadAuditHook):
    def before_train(self, runner):
        super().before_train(runner)
        assert self.model.bbox_head.instance_semantic.feature_gradient_scale == 1.
        assert self.model.bbox_head.instance_semantic.loss_weight == .1

    def after_train(self, runner):
        super().after_train(runner)
        report = json.loads(Path(self.report_path).read_text())
        report.update(method=METHOD,
            feature_gradient_scale=self.model.bbox_head.instance_semantic.feature_gradient_scale,
            neck_roi_gradient_scale=0., classification_roi_gradient_scale=1.,
            auxiliary_classifier_weight=self.model.bbox_head.instance_semantic.loss_weight,
            auxiliary_parameter_tensors=len(self.model.roi_gradient_parameters()), source_code_sha256=identity())
        report_to(self.report_path, report)


def smoke(cfg, output, fraction):
    require_memory(fraction)
    directory = Path(output).parent / 'smoke'
    assert not directory.exists(), directory
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
    cfg.custom_imports.imports.append('scripts.instance_semantic_clstower_alpha1_probe')
    cfg.custom_hooks.append(dict(type='ROIClassificationAlphaOneAuditHook', actual_method=METHOD,
        report_path=str(output), image_ids=selected, allocator_fraction=fraction))
    snapshot = directory / (METHOD + '_smoke.py')
    cfg.dump(str(snapshot))
    result = subprocess.run([sys.executable, '-u', str(ROOT / 'mmyolo/tools/train.py'), str(snapshot)],
                            cwd=ROOT, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.returncode
    assert json.loads(Path(output).read_text())['status'] == 'passed'
    assert not list(directory.rglob('*.pth'))
    shutil.rmtree(directory)


def capacity(cfg, output, fraction, batch, path):
    assert cfg.model.bbox_head.semantic_feature_gradient_scale == 1.
    existing_capacity(cfg, output, fraction, batch, path)
    report = json.loads(Path(output).read_text())
    # The common capacity runner measures this config but labels its old
    # experiment/alpha literally. Record the actual variant explicitly.
    report.update(method=METHOD, feature_gradient_scale=cfg.model.bbox_head.semantic_feature_gradient_scale,
                  classification_roi_gradient_scale=1., auxiliary_classifier_weight=.1,
                  source_code_sha256=identity())
    report_to(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs' / METHOD / 'rtmdet_tiny_dior_640.py'))
    parser.add_argument('--mode', choices=('preflight', 'fidelity', 'smoke', 'capacity'), required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allocator-fraction', type=float, default=.35)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    register_all_modules(); logging.disable(logging.CRITICAL)
    torch.set_num_threads(2); torch.manual_seed(42)
    cfg = Config.fromfile(args.config)
    try:
        if args.mode == 'preflight':
            preflight(cfg, args.report, args.config)
        elif args.mode == 'fidelity':
            fidelity(cfg, args.report, args.allocator_fraction)
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
