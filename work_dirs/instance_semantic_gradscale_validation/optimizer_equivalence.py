"""Isolated CPU comparison of gradient routing and ordinary ROI weighting.

Three synthetic B2/256 updates, no dataset inference or training-process
access. Measures the optimizer consequence, not accuracy or convergence.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from mmengine.optim import build_optim_wrapper

ROOT = Path('/home/pwt/Paper1')
sys.path.insert(0, str(ROOT / 'tests'))
from test_instance_semantic_gradscale import build, config, data

torch.set_num_threads(2)
images, samples = data()
cfg = config('scdown_p3_semantic_gradscale')
options = deepcopy(cfg.optim_wrapper)
options.type = 'OptimWrapper'
options.pop('loss_scale')
models = dict(
    routed=build('scdown_p3_semantic_gradscale'),
    ordinary_loss_quarter=build('scdown_p3_semantic'),
    original_roi=build('scdown_p3_semantic'))
models['ordinary_loss_quarter'].bbox_head.instance_semantic.loss_weight = .025
initial = {name: p.detach().clone() for name, p in models['routed'].named_parameters()}
optimizers = {name: build_optim_wrapper(model, deepcopy(options)).optimizer
              for name, model in models.items()}
records = []


def difference(first, second, prefix):
    a, b = dict(first.named_parameters()), dict(second.named_parameters())
    keys = [n for n in a if n.startswith('bbox_head.instance_semantic.') == prefix]
    errors = [(a[n].detach().double() - b[n].detach().double()) for n in keys]
    changes = [(a[n].detach().double() - initial[n].double()) for n in keys]
    absolute = sum(x.square().sum() for x in errors).sqrt()
    update = sum(x.square().sum() for x in changes).sqrt()
    return dict(max_absolute_parameter_difference=max(float(x.abs().max()) for x in errors),
                parameter_difference_l2=float(absolute), reference_update_l2=float(update),
                difference_over_reference_update=float(absolute / update.clamp_min(1e-30)))


for step in range(3):
    gradients = {}
    for name, model in models.items():
        model.train()
        optimizer = optimizers[name]
        optimizer.zero_grad()
        sum(model.loss(images, samples).values()).backward()
        gradients[name] = model.bbox_head.instance_semantic.classifier.weight.grad.detach().clone()
        optimizer.step()
    comparisons = {}
    for name in ('ordinary_loss_quarter', 'original_roi'):
        comparisons[name] = dict(
            auxiliary_classifier=difference(models['routed'], models[name], True),
            detector=difference(models['routed'], models[name], False))
    row = dict(step=step + 1, comparisons=comparisons,
               classifier_gradient_l2={n: float(g.norm()) for n, g in gradients.items()})
    records.append(row)
    print(json.dumps(row), flush=True)

report = dict(status='passed', scope='CPU FP32, identical synthetic B2/256 input, three AdamW updates; no AMP, warmup, updates to existing experiments, checkpoints or AP evaluation',
              main_structure='scdown_p3_semantic', alpha=.25,
              optimizer=dict(type='AdamW', lr=options.optimizer.lr,
                             weight_decay=options.optimizer.weight_decay,
                             eps=optimizers['routed'].defaults['eps']),
              records=records,
              interpretation='AdamW moment normalization can make a fixed ROI classifier-gradient rescaling nearly invariant; this is not an accuracy test.')
Path(__file__).with_name('optimizer_equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
