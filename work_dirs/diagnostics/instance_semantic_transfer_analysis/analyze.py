"""Read completed ROI/transfer records without touching the running trainer."""
from pathlib import Path
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/pwt/Paper1')
OUT = Path(__file__).resolve().parent
METHODS = {'baseline': 'rtmdet_tiny_dior_640_gpu1_bs36',
           'roi': 'instance_semantic_rtmdet_tiny_dior_640_gpu1_bs36',
           'transfer': 'instance_semantic_transfer_rtmdet_tiny_dior_640_gpu1_bs36'}
METRICS = ['pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50',
           'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']
runs = {}
for method, directory in METHODS.items():
    candidates = []
    for p in (ROOT / 'work_dirs' / directory).glob('*/vis_data/scalars.json'):
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        vals = {int(x['step']): x for x in rows if 'pascal_voc/mAP' in x}
        if vals:
            candidates.append((max(vals), str(p), rows, vals))
    _, source, rows, vals = max(candidates, key=lambda x: (x[0], x[1]))
    trains = [x for x in rows if 'loss_cls' in x]
    assert all(math.isfinite(x[k]) for x in trains for k in x
               if k.startswith('loss'))
    runs[method] = dict(source=source, rows=trains, vals=vals)

phases = []
for lo, hi in [(51, 100), (101, 150), (151, 200), (201, 250), (251, 280), (281, 300)]:
    item = dict(epochs=[lo, hi])
    for method, run in runs.items():
        rows = [x for x in run['rows'] if lo <= x['epoch'] <= hi]
        keys = ['loss_cls', 'loss_bbox', 'loss_instance_semantic', 'loss_instance_transfer',
                'semantic_all_gt', 'transfer_matched_points', 'transfer_reliable_points',
                'transfer_selected_points', 'transfer_selected_gt', 'transfer_mean_kl']
        item[method] = {k: statistics.mean(x[k] for x in rows)
                        for k in keys if rows and k in rows[0]}
        if method == 'transfer':
            active = [x for x in rows if x['transfer_selected_gt'] > 0]
            item[method]['zero_selected_logged_fraction'] = 1 - len(active) / len(rows)
            item[method]['median_G_over_M_when_active'] = statistics.median(
                x['semantic_all_gt'] / x['transfer_selected_gt'] for x in active)
    phases.append(item)
shared_epochs = sorted(set(runs['roi']['vals']) & set(runs['transfer']['vals']))
curve = [dict(epoch=e,
              roi_percent={k: 100 * runs['roi']['vals'][e][k] for k in METRICS},
              transfer_percent={k: 100 * runs['transfer']['vals'][e][k] for k in METRICS},
              delta_pp={k: 100 * (runs['transfer']['vals'][e][k] - runs['roi']['vals'][e][k])
                        for k in METRICS}) for e in shared_epochs]
a, b = runs['roi']['vals'][300], runs['transfer']['vals'][300]
classes = sorted([dict(name=k[5:-10], roi=100 * a[k], transfer=100 * b[k],
                       delta_pp=100 * (b[k] - a[k]))
                  for k in a if k.startswith('coco/') and k.endswith('_precision')],
                 key=lambda x: x['delta_pp'])
manifest = json.loads((ROOT / 'work_dirs/instance_semantic_transfer_queue/manifest.json').read_text())
bad_sources = [k for k, v in manifest['source_sha256'].items()
               if hashlib.sha256((ROOT / k).read_bytes()).hexdigest() != v]
result = dict(
    completed_epoch=300, source_hash_mismatches=bad_sources,
    final_percent={method: {k: 100 * run['vals'][300][k] for k in METRICS}
                   for method, run in runs.items()},
    same_epoch_curve=curve, class_deltas=classes, phases=phases,
    source_paths={k: v['source'] for k, v in runs.items()},
    formulation='L_transfer = 0.1/M sum_selected_GT mean_selected_points KL(ROI||aux_point); '
                'M is selected GT count, G is all valid GT count. Teacher and auxiliary '
                'classifier parameters are detached for this term.',
    limits=['Logged loss moving averages and sampled current-step counts have different windows; '
            'do not infer exact per-step relationships from their co-occurrence.',
            'Original ROI completed scalar file starts after a resume; compare available '
            'common validation epochs and complete training phases >=51.',
            'Single-seed AP and endpoint gradient probes do not uniquely decompose causal AP effects.'])
probe = OUT / 'gradient_probe.json'
if probe.exists():
    result['gradient_probe'] = json.loads(probe.read_text())
    summaries = {}
    for method, measurement in result['gradient_probe']['methods'].items():
        pts = measurement['points']
        n = len(pts)
        flags = ['auxiliary_point_correct', 'roi_less_gt_confident_than_point',
                 'qfl_increased_first_order', 'margin_worsened_first_order',
                 'hard_aux_ce_increases_qfl_first_order',
                 'combined_gradient_reverses_main', 'original_roi_gradient_reverses_main']
        weaker = [x for x in pts if x['roi_less_gt_confident_than_point']]
        summaries[method] = dict(
            selected_points=n,
            counts={k: sum(x[k] for x in pts) for k in flags},
            median_point_transfer_over_main_norm=statistics.median(
                x['transfer_over_main_norm'] for x in pts) if n else None,
            weaker_teacher_subset=dict(
                points=len(weaker), qfl_increased=sum(x['qfl_increased_first_order'] for x in weaker),
                hard_aux_gt_ce_qfl_increased=sum(x['hard_aux_ce_increases_qfl_first_order'] for x in weaker)),
            normalization_counterfactual=dict(
                scope='Same selected points and gradients; main+transfer only, ROI term omitted. '
                      'Changing M to G scales transfer gradient by M/G; no parameter update.',
                actual_main_plus_transfer_reversals=sum(
                    1 + x['transfer_over_main_norm'] * x['cosine_with_main'] < 0 for x in pts),
                all_gt_denominator_main_plus_transfer_reversals=sum(
                    1 + x['transfer_over_main_norm'] * x['cosine_with_main'] / x['G_over_M'] < 0
                    for x in pts)))
    result['probe_summary'] = summaries
(OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for ax, metric, title in zip(axes, METRICS[:2], ['VOC mAP@0.5', 'COCO AP']):
    for method, color in [('baseline', '#777777'), ('roi', '#15945b'), ('transfer', '#bc3053')]:
        vals = runs[method]['vals']
        epochs = sorted(vals)
        ax.plot(epochs, [100 * vals[e][metric] for e in epochs], label=method, color=color)
    ax.set(xlabel='Epoch', ylabel='AP (%)', title=title)
    ax.grid(alpha=.2)
axes[0].legend()
fig.tight_layout()
fig.savefig(OUT / 'curves.png', dpi=160)
plt.close(fig)
print(json.dumps(dict(final_percent=result['final_percent'], class_deltas=classes,
                      source_hash_mismatches=bad_sources), indent=2))
