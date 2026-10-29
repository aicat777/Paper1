"""Compare completed validation epochs; never change training state."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
METHODS = {
    'baseline': 'rtmdet_tiny_dior_640_gpu1_bs36',
    'roi': 'instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36',
    'scdown_p3': 'scdown_p3_rtmdet_tiny_dior_640_gpu1_bs36',
    'gsce': 'scdown_p3_g4_rtmdet_tiny_dior_640_gpu1_bs36',
    'gsce_roi': 'scdown_p3_g4_semantic_rtmdet_tiny_dior_640_gpu1_bs36',
}
METRICS = ['pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50',
           'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
runs = {}
training = {}
for method, directory in METHODS.items():
    candidates = []
    for candidate in (ROOT / 'work_dirs' / directory).glob('*/vis_data/scalars.json'):
        # A running writer may leave its final JSON line incomplete.
        candidate_rows = []
        for line in candidate.read_text().splitlines():
            try:
                candidate_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        epochs = [int(x['step']) for x in candidate_rows if 'pascal_voc/mAP' in x]
        if epochs:
            candidates.append((max(epochs), str(candidate), candidate_rows))
    _, chosen, rows = max(candidates, key=lambda x: (x[0], x[1]))
    path = Path(chosen)
    vals = {int(x['step']): x for x in rows if 'pascal_voc/mAP' in x}
    trains = [x for x in rows if 'loss_cls' in x]
    runs[method] = dict(source=str(path), values=vals,
                        last_validation_epoch=max(vals), last_train_epoch=trains[-1]['epoch'])
    training[method] = {}
    for lo, hi in [(51, 100), (101, 150), (151, 190), (201, 250), (251, 280), (281, 290)]:
        selected = [x for x in trains if lo <= x['epoch'] <= hi]
        keys = ['loss_cls', 'loss_bbox', 'loss_instance_semantic']
        training[method][f'{lo}_{hi}'] = dict(
            logged_steps=len(selected),
            means={k: float(np.mean([x[k] for x in selected]))
                   for k in keys if selected and all(k in x for x in selected)})
    assert all(math.isfinite(float(x[k])) for x in trains
               for k in ('loss_cls', 'loss_bbox'))

last = runs['gsce_roi']['last_validation_epoch']
matched = {key: {k: 100 * run['values'][last][k] for k in METRICS}
           for key, run in runs.items()}
paired = []
interaction = []
for epoch in sorted(runs['gsce_roi']['values']):
    a, b = runs['gsce']['values'][epoch], runs['gsce_roi']['values'][epoch]
    paired.append(dict(epoch=epoch, delta_percent={k: 100 * (b[k] - a[k]) for k in METRICS}))
    if epoch not in runs['roi']['values'] or epoch not in runs['baseline']['values']:
        continue
    original_roi = runs['roi']['values'][epoch]
    baseline = runs['baseline']['values'][epoch]
    interaction.append(dict(epoch=epoch,
        roi_effect_on_baseline_percent={k: 100 * (original_roi[k] - baseline[k]) for k in METRICS},
        roi_effect_on_gsce_percent={k: 100 * (b[k] - a[k]) for k in METRICS},
        difference_of_effects_percent={k: 100 * ((b[k] - a[k]) - (original_roi[k] - baseline[k])) for k in METRICS}))
classes = {}
for key, run in runs.items():
    classes[key] = {k: 100 * v for k, v in run['values'][last].items()
                    if k.startswith('coco/') and k.endswith('_precision')}
class_delta = sorted(
    ({'class': k.removeprefix('coco/').removesuffix('_precision'),
      'gsce': classes['gsce'][k], 'gsce_roi': classes['gsce_roi'][k],
      'delta': classes['gsce_roi'][k] - classes['gsce'][k]}
     for k in classes['gsce']), key=lambda x: x['delta'])
checks = json.loads((ROOT / 'work_dirs/scdown_p3_g4_semantic_validation/checks.json').read_text())
hashes = checks['source_sha256']
mismatches = [name for name, wanted in hashes.items()
              if not (ROOT / name).exists()
              or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != wanted]
result = dict(snapshot='2026-10-27, Asia/Shanghai', comparison_epoch=last,
              gsce_definition='Four SCDown transitions plus seven G4 CSP conv1s',
              gsce_roi_definition='GSCE plus unchanged original all-valid-GT ROI CE, weight 0.1',
              same_epoch_percent=matched,
              completed_controls_final_percent={key: {k: 100 * run['values'][300][k] for k in METRICS}
                                                for key, run in runs.items() if 300 in run['values']},
              roi_minus_gsce_curve=paired,
              descriptive_interaction=interaction,
              coco_delta_positive_epochs=sum(x['delta_percent']['coco/bbox_mAP'] > 1e-9 for x in paired),
              coco_delta_zero_epochs=sum(abs(x['delta_percent']['coco/bbox_mAP']) <= 1e-9 for x in paired),
              coco_delta_negative_epochs=sum(x['delta_percent']['coco/bbox_mAP'] < -1e-9 for x in paired),
              class_delta_vs_gsce=class_delta, training_loss_means=training,
              source_hashes_checked=len(hashes), source_hash_mismatches=mismatches,
              structure_checks=checks['structure'],
              limits=(['Combined training has not completed 300 epochs.'] if last < 300 else []) + [
                      'AP75 depends on scores and localization; it does not isolate regression errors.',
                      'Training-loss means describe logged moving averages, not independent causal factors.',
                      'A small single-seed difference does not demonstrate an intrinsic module conflict.'],
              sources={key: {k: v for k, v in run.items() if k != 'values'} for key, run in runs.items()})
gradient_path = OUT / 'gradient_probe.json'
if gradient_path.exists():
    result['gradient_probe'] = json.loads(gradient_path.read_text())
paired_path = OUT / 'paired_parameter_probe.json'
if paired_path.exists():
    result['paired_parameter_probe'] = json.loads(paired_path.read_text())
(OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
labels = {'baseline': 'Baseline', 'roi': 'Original ROI', 'scdown_p3': 'SCDown P3',
          'gsce': 'GSCE (SCDown P3 + G4)', 'gsce_roi': 'GSCE + ROI'}
colors = ['#666666', '#15945b', '#377eb8', '#ed8b23', '#bc3053']
for ax, metric, title in zip(axes, ['pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_75'],
                             ['VOC mAP@0.5', 'COCO AP', 'COCO AP75']):
    for (key, run), color in zip(runs.items(), colors):
        epochs = sorted(run['values'])
        ax.plot(epochs, [100 * run['values'][e][metric] for e in epochs],
                label=labels[key], color=color, linewidth=2.2 if key == 'gsce_roi' else 1.4)
    ax.axvline(last, color='#999999', linestyle=':', linewidth=1)
    ax.set(xlabel='Epoch', ylabel='AP (%)', title=title, xlim=(100, 300))
    ax.set_ylim(60 if metric == 'pascal_voc/mAP' else 42, 78 if metric == 'pascal_voc/mAP' else 64)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle(f'GSCE + ROI: latest completed validation at epoch {last}; controls run to 300')
fig.tight_layout()
fig.savefig(OUT / 'curves.png', dpi=160)
plt.close(fig)
print(json.dumps({k: result[k] for k in ['comparison_epoch', 'same_epoch_percent',
                 'coco_delta_positive_epochs', 'coco_delta_zero_epochs',
                 'coco_delta_negative_epochs', 'class_delta_vs_gsce']}, indent=2))
