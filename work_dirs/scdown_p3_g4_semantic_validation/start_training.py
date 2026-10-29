"""Launch the checked SCDown4+G4+ROI experiment through the existing scheduler."""
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
METHOD = 'scdown_p3_g4_semantic'
sys.path.insert(0, str(ROOT))
from scripts.queue_experiment import check_sources, completed, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    validation = ROOT / f'work_dirs/{METHOD}_validation'
    structure = read(validation / 'preflight.json')
    smoke = read(validation / 'smoke_gpu1.json')
    assert structure['status'] == smoke['status'] == 'passed'
    assert structure['parameters'] == 3105704 and structure['inference_parameters'] == 3103764
    assert structure['inference_conv_linear_gmac'] == 6.78524544
    assert structure['watched_parameter_tensors'] == smoke['watched_parameter_tensors'] == 67
    assert smoke['optimizer_updates'] == 16 and smoke['val_passes'] == 2
    assert smoke['amp_skipped_updates'] == 0 and smoke['full_state_ema_roundtrips'] == 2
    assert smoke['model_optimizer_scaler_serialization']
    assert not smoke['saved_checkpoints'] and not (validation / 'smoke').exists()
    unit_log = validation / 'unit_tests.log'
    assert '\nRan 6 tests in ' in unit_log.read_text() and unit_log.read_text().rstrip().endswith('OK')
    assert read(validation / 'replacement_cleanup.json')['status'] == 'passed'
    for path in (ROOT / 'work_dirs').glob('*_queue/status.json'):
        state = read(path)
        if state.get('queue_start_ticks'):
            assert not process_alive(state.get('queue_pid', -1), state['queue_start_ticks']), path

    # Retain the identity of the last completed job, rather than referring to
    # the cancelled experiment whose partial outputs have been removed.
    parent_dir = ROOT / 'work_dirs/scdown_p3_g8_queue'
    parent, identity = read(parent_dir / 'status.json'), read(parent_dir / 'queue_identity.json')
    assert parent['state'] == 'complete' and parent['queue_pid'] == identity['pid']
    assert parent['queue_start_ticks'] == identity['start_ticks']
    assert not process_alive(identity['pid'], identity['start_ticks'])
    assert completed(parent['experiment']['work_dir'], parent['experiment']['log'])
    dependency = dict(name='scdown_p3_g8', pid=identity['pid'], start_ticks=identity['start_ticks'],
                      status=str(parent_dir / 'status.json'), work_dir=parent['experiment']['work_dir'],
                      log=parent['experiment']['log'])

    reference = read(ROOT / 'work_dirs/scdown_p3_g4_queue/manifest.json')
    check_sources(reference)
    sources = {name for name in reference['source_sha256'] if not name.startswith('work_dirs/')}
    for module in ('instance_semantic', METHOD):
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT / f'custom_models/{module}').glob('*.py'))
    sources.update((f'configs/{METHOD}/rtmdet_tiny_dior_640.py',
                    'configs/instance_semantic/rtmdet_tiny_dior_640.py',
                    f'scripts/{METHOD}_probe.py', f'tests/test_{METHOD}.py',
                    f'work_dirs/{METHOD}_validation/formal_640_bs36.py',
                    f'work_dirs/{METHOD}_validation/start_training.py'))
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(sources)}
    formal = validation / 'formal_640_bs36.py'
    assert hashes[str(formal.relative_to(ROOT))] == structure['config_sha256']
    checks = dict(status='passed', method=METHOD, reference='scdown_p3_g4',
                  unit_tests=dict(status='passed', test_cases=6, log=str(unit_log)),
                  structure=structure, official_gpu_smoke=smoke, source_sha256=hashes,
                  formal_config=str(formal),
                  formal_capacity=dict(status='pending', formal_batch=36))
    write(validation / 'checks.json', checks)

    directory = ROOT / f'work_dirs/{METHOD}_queue'
    work = ROOT / f'work_dirs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36'
    log = ROOT / f'logs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36.log'
    assert not directory.exists() and not work.exists() and not log.exists()
    directory.mkdir()
    frozen = directory / f'{METHOD}_640_bs36.py'
    frozen.write_bytes(formal.read_bytes())
    queue_hashes = dict(hashes)
    queue_hashes[str(frozen.relative_to(ROOT))] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    common = [PYTHON, '-u', str(ROOT / f'scripts/{METHOD}_probe.py'), '--config', str(frozen)]
    manifest = dict(root=str(ROOT), python=PYTHON, gpu_uuid=reference['gpu_uuid'], min_free_mib=46000,
        dependency=dependency,
        experiment=dict(name=f'{METHOD}_640_bs36', source_config=str(ROOT / f'configs/{METHOD}/rtmdet_tiny_dior_640.py'),
            config=str(frozen), work_dir=str(work), log=str(log),
            preflight_report=str(validation / 'queue_preflight.json'),
            capacity_report=str(validation / 'capacity_gpu1_bs36.json'), validation_summary=str(validation / 'checks.json')),
        source_sha256=queue_hashes,
        preflight_command=common + ['--mode', 'preflight', '--report', str(validation / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--report', str(validation / 'capacity_gpu1_bs36.json'),
                                  '--allocator-fraction', '.88'],
        capacity_expected=dict(method=METHOD, microbatch=36, formal_batch=36, accumulation=1,
            optimizer_updates=3, gt_per_image=1200, allocator_fraction=.88,
            formal_batch_capacity_tested=True, amp_skipped_updates=0, saved_checkpoints=False,
            watched_parameter_tensors=67, cudnn_benchmark=True, config_sha256=queue_hashes[str(frozen.relative_to(ROOT))]))
    check_sources(manifest)
    write(directory / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'), str(directory / 'manifest.json')]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (directory / 'scheduler.log').open('x') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()[19]
    write(directory / 'queue_identity.json', dict(pid=process.pid, start_ticks=ticks,
          created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    # Return once the scheduler has accepted its identity. The parent agent
    # verifies capacity and real training separately before reporting success.
    for _ in range(30):
        if process.poll() is not None:
            raise RuntimeError((directory / 'scheduler.log').read_text()[-6000:])
        if (directory / 'status.json').exists():
            state = read(directory / 'status.json')
            if state.get('queue_pid') == process.pid and str(state.get('queue_start_ticks')) == ticks:
                assert state['state'] != 'failed', state
                checks.update(queued=True, queue_manifest=str(directory / 'manifest.json'))
                write(validation / 'checks.json', checks)
                print('SCHEDULER_STARTED', process.pid, state['state'], flush=True)
                return
        time.sleep(1)
    raise RuntimeError('Scheduler did not publish its own status')


if __name__ == '__main__':
    main()
