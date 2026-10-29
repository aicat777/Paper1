"""Append the three verified ROI variants without touching the active trainer."""
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
METHODS = ('private', 'levels', 'contrast')
sys.path.insert(0, str(ROOT))
from scripts.queue_experiment import check_sources, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parent_dir = ROOT / 'work_dirs/scdown_p3_g4_semantic_clstower_queue'
    parent = read(parent_dir / 'status.json')
    parent_manifest = read(parent_dir / 'manifest.json')
    assert parent['state'] == 'training'
    assert process_alive(parent['queue_pid'], parent['queue_start_ticks'])
    assert parent['experiment']['work_dir'] in Path(
        f'/proc/{parent["child_pid"]}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert parent['experiment'] == parent_manifest['experiment']
    check_sources(parent_manifest)
    for path in (ROOT / 'work_dirs').glob('*_queue/status.json'):
        state = read(path)
        if state.get('queue_start_ticks') and process_alive(
                state.get('queue_pid', -1), state['queue_start_ticks']):
            assert path.parent == parent_dir, 'An existing successor is already alive: ' + str(path)

    # Validate all three first, so a missing check is found before any launch.
    hashes = dict(parent_manifest['source_sha256'])
    validations = {}
    for short in METHODS:
        method = 'scdown_p3_g4_semantic_' + short
        validation = ROOT / f'work_dirs/{method}_validation'
        checks = read(validation / 'checks.json')
        assert checks['status'] == 'passed' and checks['method'] == method
        assert checks['shared_cpu_integration_tests'] == 13
        assert checks['gpu_reference_cases'] == 4
        assert checks['official_smoke_optimizer_updates'] == 16
        assert checks['val_passes'] == checks['ema_roundtrips'] == 2
        assert checks['amp_skipped_updates'] == 0
        assert checks['capacity_microbatch'] == 8
        assert checks['capacity_gt_per_image'] == 1200
        assert checks['capacity_optimizer_updates'] == 3
        assert checks['whole_model_fp32_reference']['status'] == 'passed'
        for mode, path in checks['reports'].items():
            report = read(path)
            assert report['status'] == 'passed', (short, mode)
            recorded = report.get('source_code_sha256', {})
            check_sources(dict(root=str(ROOT), source_sha256=recorded))
            for relative, expected in recorded.items():
                assert hashes.get(relative, expected) == expected, relative
                hashes[relative] = expected
        assert not (ROOT / f'work_dirs/{method}_queue').exists()
        assert not (ROOT / f'work_dirs/{method}_rtmdet_tiny_dior_640_gpu1_bs36').exists()
        assert not (ROOT / f'logs/{method}_rtmdet_tiny_dior_640_gpu1_bs36.log').exists()
        for path in (ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py',
                     validation / 'formal_config.py'):
            hashes[str(path.relative_to(ROOT))] = digest(path)
        validations[short] = (validation, checks)
    assert (ROOT / 'work_dirs/roi_auxiliary_variants_tests.log').read_text().rstrip().endswith('OK')
    for path in (Path(__file__).resolve(), ROOT / 'tests/test_roi_auxiliary_variants.py',
                 ROOT / 'scripts/candidate_capacity.py', ROOT / 'scripts/candidate_probe.py'):
        hashes[str(path.relative_to(ROOT))] = digest(path)

    chain = []
    for short in METHODS:
        method = 'scdown_p3_g4_semantic_' + short
        validation, checks = validations[short]
        dependency = dict(name=parent_dir.name.removesuffix('_queue'),
            pid=parent['queue_pid'], start_ticks=parent['queue_start_ticks'],
            status=str(parent_dir / 'status.json'), work_dir=parent['experiment']['work_dir'],
            log=parent['experiment']['log'])
        directory = ROOT / f'work_dirs/{method}_queue'
        directory.mkdir()
        frozen = directory / f'{method}_640_bs36.py'
        frozen.write_bytes((validation / 'formal_config.py').read_bytes())
        hashes[str(frozen.relative_to(ROOT))] = digest(frozen)
        common = [PYTHON, '-u', str(ROOT / 'scripts/roi_auxiliary_variants_probe.py'),
                  '--method', short, '--config', str(frozen)]
        manifest = dict(root=str(ROOT), python=PYTHON, gpu_uuid=parent_manifest['gpu_uuid'],
            min_free_mib=46000, dependency=dependency,
            experiment=dict(name=f'{method}_640_bs36',
                source_config=str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'),
                config=str(frozen),
                work_dir=str(ROOT / f'work_dirs/{method}_rtmdet_tiny_dior_640_gpu1_bs36'),
                log=str(ROOT / f'logs/{method}_rtmdet_tiny_dior_640_gpu1_bs36.log'),
                preflight_report=str(validation / 'queue_preflight.json'),
                capacity_report=str(validation / 'capacity_gpu1_bs36.json'),
                validation_summary=str(validation / 'checks.json')),
            source_sha256=dict(hashes),
            preflight_command=common + ['--mode', 'preflight', '--report',
                                       str(validation / 'queue_preflight.json')],
            capacity_command=common + ['--mode', 'capacity', '--batch', '36', '--report',
                str(validation / 'capacity_gpu1_bs36.json'), '--allocator-fraction', '.88'],
            capacity_expected=dict(method=method, microbatch=36, accumulation=1,
                optimizer_updates=3, training_microsteps=3, input_size=[640, 640],
                gt_per_image=1200, allocator_fraction=.88, formal_batch_capacity_tested=True,
                amp_skipped_updates=0, saved_checkpoints=False,
                watched_parameter_tensors=2 if short == 'levels' else 6,
                config_sha256=digest(frozen)))
        check_sources(manifest)
        write(directory / 'manifest.json', manifest)
        command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'),
                   str(directory / 'manifest.json')]
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (directory / 'scheduler.log').open('x') as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
        ticks = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()[19]
        write(directory / 'queue_identity.json', dict(pid=process.pid, start_ticks=ticks,
            created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError((directory / 'scheduler.log').read_text()[-5000:])
            status_path = directory / 'status.json'
            if status_path.is_file():
                state = read(status_path)
                assert state['queue_pid'] == process.pid and state['queue_start_ticks'] == ticks
                if state['state'] == 'failed':
                    raise RuntimeError(state)
                if state['state'] == 'waiting_' + dependency['name']:
                    break
            time.sleep(.5)
        else:
            raise RuntimeError('Queue preflight did not finish')
        checks.update(full_training_queued=True, queue_manifest=str(directory / 'manifest.json'),
                      formal_capacity=dict(status='pending', formal_batch=36, accumulation=1))
        checks['reports']['queue_preflight'] = manifest['experiment']['preflight_report']
        write(validation / 'checks.json', checks)
        chain.append(dict(method=method, queue_pid=process.pid, queue_start_ticks=ticks,
            dependency=dependency['name'], state=state['state'], manifest=str(directory / 'manifest.json')))
        print(json.dumps(chain[-1]), flush=True)
        parent_dir, parent, parent_manifest = directory, state, manifest
    summary_path = ROOT / 'work_dirs/roi_auxiliary_variants_validation/checks.json'
    summary = read(summary_path)
    summary.update(full_training_queued=True, queue_order=[row['method'] for row in chain],
                   queues=chain, formal_capacity=dict(status='pending', formal_batch=36))
    write(summary_path, summary)
    write(Path(__file__).parent / 'verified_chain.json', chain)


if __name__ == '__main__':
    main()
