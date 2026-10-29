"""Queue classification-only ROI after the queued neck/classification ROI control."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/pwt/Paper1')
PYTHON = '/home/pwt/miniconda3/envs/rtmdet/bin/python'
METHOD = 'scdown_p3_g4_semantic_clstower'
PARENT = 'scdown_p3_g4_semantic_neckhead'
sys.path.insert(0, str(ROOT))
from scripts.queue_experiment import check_sources, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    validation = ROOT / f'work_dirs/{METHOD}_validation'
    checks = read(validation / 'checks.json')
    assert checks['status'] == 'passed' and checks['method'] == METHOD
    assert checks['unit_tests']['status'] == 'passed'
    assert checks['unit_tests']['test_cases'] == 7
    assert Path(checks['unit_tests']['log']).read_text().rstrip().endswith('OK')
    structure, smoke = checks['structure'], checks['official_gpu_smoke']
    assert structure['status'] == smoke['status'] == 'passed'
    assert structure['parameters'] == 3105704 and structure['inference_parameters'] == 3103764
    assert structure['new_parameters'] == 0
    assert structure['inference_conv_linear_gmac'] == 6.78524544
    assert structure['main_losses_and_matching_unchanged']
    assert structure['original_roi_preserved'] and structure['all_gt_and_original_normalization_preserved']
    assert structure['backbone_roi_gradient_scale'] == 0.
    assert structure['neck_roi_gradient_scale'] == 0.
    assert structure['classification_roi_gradient_scale'] == .25
    assert structure['auxiliary_parameter_tensors'] == 16
    assert structure['auxiliary_classifier_weight'] == .1
    assert structure['training_protocol_unchanged'] and structure['scratch']
    assert structure['formal_batch'] == 36 and structure['accumulation'] == 1
    assert structure['watched_parameter_tensors'] == smoke['watched_parameter_tensors'] == 145
    assert smoke['optimizer_updates'] == 16 and smoke['val_passes'] == 2
    assert smoke['amp_skipped_updates'] == 0 and smoke['full_state_ema_roundtrips'] == 2
    assert smoke['model_optimizer_scaler_serialization']
    assert smoke['classifier_uses_all_gt'] and smoke['one_bn_update_per_train_step']
    assert smoke['feature_gradient_scale'] == .25 and not smoke['saved_checkpoints']
    assert smoke['backbone_neck_forward_counts'] == dict(backbone=16, neck=16)
    gradient = checks['gpu_gradient_reference']
    assert gradient['status'] == 'passed' and len(gradient['cases']) == 4
    assert all(row['roi_backbone_gradient_exactly_zero'] and row['roi_neck_gradient_exactly_zero']
               and row['main_backbone_gradient_nonzero'] and row['main_neck_gradient_nonzero']
               for row in gradient['cases'])
    capacity = checks['shared_gpu_capacity']
    assert capacity['status'] == 'passed' and capacity['microbatch'] == 8
    assert capacity['optimizer_updates'] == 3 and capacity['amp_skipped_updates'] == 0
    check_sources(dict(root=str(ROOT), source_sha256=checks['source_sha256']))

    parent_dir = ROOT / f'work_dirs/{PARENT}_queue'
    parent = read(parent_dir / 'status.json')
    identity = read(parent_dir / 'queue_identity.json')
    parent_manifest = read(parent_dir / 'manifest.json')
    assert parent['state'] == 'waiting_scdown_p3_g4_semantic_ibqfl', parent
    assert parent['queue_pid'] == identity['pid']
    assert parent['queue_start_ticks'] == identity['start_ticks']
    assert process_alive(identity['pid'], identity['start_ticks'])
    assert str(parent_dir / 'manifest.json') in Path(
        f'/proc/{identity["pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert parent['experiment'] == parent_manifest['experiment']
    check_sources(parent_manifest)
    upstream = parent_manifest['dependency']
    upstream_status = read(upstream['status'])
    assert upstream_status['state'] == 'training'
    assert upstream_status['queue_pid'] == upstream['pid']
    assert upstream_status['queue_start_ticks'] == upstream['start_ticks']
    assert process_alive(upstream['pid'], upstream['start_ticks'])
    assert upstream['work_dir'] in Path(
        f'/proc/{upstream_status["child_pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    upstream_dir = Path(upstream['status']).parent
    check_sources(read(upstream_dir / 'manifest.json'))
    # A third live queue here would indicate that a successor already exists.
    for path in (ROOT / 'work_dirs').glob('*_queue/status.json'):
        state = read(path)
        if state.get('queue_start_ticks') and process_alive(
                state.get('queue_pid', -1), state['queue_start_ticks']):
            assert path.parent in (parent_dir, upstream_dir), path
    dependency = dict(
        name=PARENT, pid=identity['pid'], start_ticks=identity['start_ticks'],
        status=str(parent_dir / 'status.json'),
        work_dir=parent['experiment']['work_dir'], log=parent['experiment']['log'])

    directory = ROOT / f'work_dirs/{METHOD}_queue'
    work = ROOT / f'work_dirs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36'
    log = ROOT / f'logs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36.log'
    assert not directory.exists() and not work.exists() and not log.exists()
    formal = Path(checks['formal_config'])
    hashes = dict(checks['source_sha256'])
    assert hashes[str(formal.relative_to(ROOT))] == hashlib.sha256(formal.read_bytes()).hexdigest()
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    directory.mkdir()
    frozen = directory / f'{METHOD}_640_bs36.py'
    frozen.write_bytes(formal.read_bytes())
    hashes[str(frozen.relative_to(ROOT))] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    common = [PYTHON, '-u', str(ROOT / 'scripts/instance_semantic_clstower_probe.py'),
              '--config', str(frozen)]
    manifest = dict(
        root=str(ROOT), python=PYTHON, gpu_uuid=parent_manifest['gpu_uuid'],
        min_free_mib=46000, dependency=dependency,
        experiment=dict(
            name=f'{METHOD}_640_bs36',
            source_config=str(ROOT / f'configs/{METHOD}/rtmdet_tiny_dior_640.py'),
            config=str(frozen), work_dir=str(work), log=str(log),
            preflight_report=str(validation / 'queue_preflight.json'),
            capacity_report=str(validation / 'capacity_gpu1_bs36.json'),
            validation_summary=str(validation / 'checks.json')),
        source_sha256=hashes,
        preflight_command=common + ['--mode', 'preflight', '--report',
                                    str(validation / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--report',
                                   str(validation / 'capacity_gpu1_bs36.json'),
                                   '--allocator-fraction', '.88'],
        capacity_expected=dict(
            method=METHOD, microbatch=36, formal_batch=36, accumulation=1,
            optimizer_updates=3, gt_per_image=1200, allocator_fraction=.88,
            formal_batch_capacity_tested=True, amp_skipped_updates=0,
            saved_checkpoints=False, watched_parameter_tensors=145, feature_gradient_scale=.25, backbone_roi_gradient_scale=0., neck_roi_gradient_scale=0.,
            cudnn_benchmark=True,
            config_sha256=hashes[str(frozen.relative_to(ROOT))]))
    check_sources(manifest)
    write(directory / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'),
               str(directory / 'manifest.json')]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (directory / 'scheduler.log').open('x') as output:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()[19]
    write(directory / 'queue_identity.json', dict(
        pid=process.pid, start_ticks=ticks,
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError((directory / 'scheduler.log').read_text()[-6000:])
        if (directory / 'status.json').exists():
            state = read(directory / 'status.json')
            assert state['queue_pid'] == process.pid and state['queue_start_ticks'] == ticks
            if state['state'] == 'failed':
                raise RuntimeError(state)
            if state['state'] == 'waiting_' + PARENT:
                checks.update(queued=True, queue_manifest=str(directory / 'manifest.json'),
                              formal_capacity=dict(status='pending', formal_batch=36, accumulation=1))
                write(validation / 'checks.json', checks)
                print('QUEUED', process.pid, state['state'], flush=True)
                return
        time.sleep(.5)
    raise RuntimeError('Scheduler did not finish preflight and enter dependency wait')


if __name__ == '__main__':
    main()
