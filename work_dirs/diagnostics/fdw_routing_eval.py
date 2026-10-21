"""Paired, inference-only FDW routing diagnostic on the full official DIOR val."""
from pathlib import Path
import copy
import hashlib
import json
import time

import torch
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

ROOT = Path('/home/pwt/Paper1')
REPORT = ROOT / 'work_dirs/diagnostics/fdw_routing_eval.json'


def main():
    register_all_modules()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.12)
    cfg = Config.fromfile(str(ROOT / 'configs/fdw/rtmdet_tiny_dior_640.py'))
    cfg.val_dataloader.update(batch_size=4, num_workers=4, persistent_workers=False)
    loader = Runner.build_dataloader(cfg.val_dataloader, seed=42)
    assert len(loader.dataset) == 5863
    calibration = json.loads((ROOT / 'work_dirs/diagnostics/fdw_final_mechanism.json').read_text())
    checkpoint_path = ROOT / 'work_dirs/fdw_rtmdet_tiny_dior_640_gpu1_bs36/epoch_300.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert checkpoint['meta']['epoch'] == 300
    model = MODELS.build(cfg.model)
    # MMEngine's EMA hook saves validation/EMA weights under state_dict.
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda().eval()
    del checkpoint
    evaluators = {name: Evaluator(copy.deepcopy(cfg.val_evaluator))
                  for name in ('adaptive', 'fixed_mean')}
    for evaluator in evaluators.values():
        evaluator.dataset_meta = loader.dataset.metainfo
    mode = ['adaptive']
    counters = {name: 0 for name in calibration['layers']}
    handles = []
    for name, stats in calibration['layers'].items():
        module = model.get_submodule(name)
        mean = torch.tensor(stats['effective_gate_mean_by_group'], device='cuda')[None] / module.kernel_num
        def hook(router, inputs, output, mean=mean, name=name):
            if mode[0] == 'fixed_mean':
                counters[name] += output.shape[0]
                return mean.to(output).expand(output.shape[0], -1)
            return output
        handles.append(module.router.register_forward_hook(hook))
    report = dict(status='running', checkpoint=str(checkpoint_path),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  weights='state_dict (EMA)', training_changed=False, images=5863,
                  batch_size=4, amp=False, allocator_fraction=.12, gpu='GPU0',
                  calibration=dict(source='fdw_final_mechanism.json',
                                   images=calibration['images'],
                                   split='official val; cached unlabeled routing statistics',
                                   purpose='Post-training mechanism diagnostic, not a new benchmark result'),
                  interpretation='All three routing outputs are fixed simultaneously; this changes inference '
                                 'inputs to later layers and does not equal training a static counterpart.',
                  metrics={})
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            for name, evaluator in evaluators.items():
                mode[0] = name
                predictions = model.val_step(batch)
                for sample in predictions:
                    assert torch.isfinite(sample.pred_instances.bboxes).all()
                    assert torch.isfinite(sample.pred_instances.scores).all()
                evaluator.process(data_samples=predictions, data_batch=batch)
            if index % 100 == 0 or index == len(loader) - 1:
                report.update(processed_images=min((index + 1) * 4, 5863),
                              elapsed_seconds=time.perf_counter() - started)
                REPORT.write_text(json.dumps(report, indent=2) + '\n')
                print('progress', report['processed_images'], 'elapsed', round(report['elapsed_seconds'], 1), flush=True)
    for handle in handles:
        handle.remove()
    assert all(count == 5863 for count in counters.values())
    for name, evaluator in evaluators.items():
        print('evaluating', name, flush=True)
        report['metrics'][name] = evaluator.evaluate(len(loader.dataset))
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
    report.update(status='passed', routing_overrides_per_layer=counters,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    report['fixed_minus_adaptive_pp'] = {
        key: (report['metrics']['fixed_mean'][key] - value) * 100
        for key, value in report['metrics']['adaptive'].items()}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('result', json.dumps({name: {key: value for key, value in metrics.items()
                                      if key in ('pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_s',
                                                 'coco/bbox_mAP_m', 'coco/bbox_mAP_l')}
                                for name, metrics in report['metrics'].items()}), flush=True)


if __name__ == '__main__':
    main()
