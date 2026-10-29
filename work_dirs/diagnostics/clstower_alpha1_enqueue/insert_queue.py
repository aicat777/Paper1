"""Insert checked alpha=1 ROI immediately after private ROI; preserve its trainer."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path('/home/pwt/Paper1')
PYTHON = '/home/pwt/miniconda3/envs/rtmdet/bin/python'
METHOD = 'scdown_p3_g4_semantic_clstower_alpha1'
ACTIVE = 'scdown_p3_g4_semantic_private'
LEVELS = 'scdown_p3_g4_semantic_levels'
CONTRAST = 'scdown_p3_g4_semantic_contrast'
AUDIT = Path(__file__).resolve().parent
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


def inspect(method, expected_state):
    directory = ROOT / f'work_dirs/{method}_queue'
    state, manifest = read(directory / 'status.json'), read(directory / 'manifest.json')
    assert state['state'] == expected_state, (method, state)
    assert process_alive(state['queue_pid'], state['queue_start_ticks']), method
    command = Path(f'/proc/{state["queue_pid"]}/cmdline').read_bytes().split(b'\0')
    assert str(directory / 'manifest.json').encode() in command
    assert state['experiment'] == manifest['experiment']
    check_sources(manifest)
    return directory, state, manifest


def dependency(method, state, directory):
    return dict(name=method, pid=state['queue_pid'], start_ticks=state['queue_start_ticks'],
        status=str(directory / 'status.json'), work_dir=state['experiment']['work_dir'],
        log=state['experiment']['log'])


def launch(directory, manifest):
    check_sources(manifest)
    write(directory / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'), str(directory / 'manifest.json')]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (directory / 'scheduler.log').open('a') as output:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rpartition(') ')[2].split()[19]
    write(directory / 'queue_identity.json', dict(pid=child.pid, start_ticks=ticks,
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError((directory / 'scheduler.log').read_text()[-4000:])
        path = directory / 'status.json'
        if path.is_file():
            state = read(path)
            if state['queue_pid'] == child.pid and state['queue_start_ticks'] == ticks:
                if state['state'] == 'failed':
                    raise RuntimeError(state)
                if state['state'] == 'waiting_' + manifest['dependency']['name']:
                    print(directory.name, state['state'], child.pid, flush=True)
                    return state
        time.sleep(.5)
    raise RuntimeError('New scheduler did not finish startup checks: ' + str(directory))


def stop_waiter(directory, original):
    current = read(directory / 'status.json')
    assert current['state'] == original['state'] and current['state'].startswith('waiting_')
    assert current['queue_pid'] == original['queue_pid']
    assert current['queue_start_ticks'] == original['queue_start_ticks']
    assert 'child_pid' not in current, 'Never stop a trainer or preflight child'
    assert process_alive(current['queue_pid'], current['queue_start_ticks'])
    cmd = Path(f'/proc/{current["queue_pid"]}/cmdline').read_bytes().split(b'\0')
    assert str(directory / 'manifest.json').encode() in cmd
    assert not Path(current['experiment']['work_dir']).exists()
    assert not Path(current['experiment']['log']).exists()
    os.kill(current['queue_pid'], signal.SIGTERM)
    for _ in range(100):
        if not process_alive(current['queue_pid'], current['queue_start_ticks']):
            return
        time.sleep(.1)
    raise RuntimeError('Waiting scheduler did not exit')


def main():
    validation = ROOT / f'work_dirs/{METHOD}_validation'
    checks = read(validation / 'checks.json')
    assert checks['status'] == 'passed'
    assert checks['unit_tests']['test_cases'] == 3
    assert Path(checks['unit_tests']['log']).read_text().rstrip().endswith('OK')
    assert checks['structure']['feature_gradient_scale'] == 1.
    assert len(checks['gpu_gradient_reference']['cases']) == 4
    assert checks['official_gpu_smoke']['optimizer_updates'] == 16
    assert checks['official_gpu_smoke']['amp_skipped_updates'] == 0
    assert checks['official_gpu_smoke']['feature_gradient_scale'] == 1.
    assert checks['shared_gpu_capacity']['microbatch'] == 8
    assert checks['shared_gpu_capacity']['optimizer_updates'] == 3
    assert checks['shared_gpu_capacity']['amp_skipped_updates'] == 0
    check_sources(dict(root=str(ROOT), source_sha256=checks['source_sha256']))
    active_dir, active, active_manifest = inspect(ACTIVE, 'training')
    trainer_pid = active['child_pid']
    assert active['experiment']['work_dir'].encode() in Path(f'/proc/{trainer_pid}/cmdline').read_bytes()
    trainer_ticks = Path(f'/proc/{trainer_pid}/stat').read_text().rpartition(') ')[2].split()[19]
    levels_dir, levels, levels_manifest = inspect(LEVELS, 'waiting_' + ACTIVE)
    contrast_dir, contrast, contrast_manifest = inspect(CONTRAST, 'waiting_' + LEVELS)
    assert levels_manifest['dependency']['pid'] == active['queue_pid']
    assert contrast_manifest['dependency']['pid'] == levels['queue_pid']
    for path in (ROOT / 'work_dirs').glob('*_queue/status.json'):
        row = read(path)
        if row.get('queue_start_ticks') and process_alive(row.get('queue_pid', -1), row['queue_start_ticks']):
            assert path.parent in (active_dir, levels_dir, contrast_dir), path
    backup = AUDIT / 'before_insertion.json'
    assert not backup.exists()
    write(backup, dict(preserved_trainer_pid=trainer_pid, preserved_trainer_ticks=trainer_ticks,
        queues={name: dict(state=state, manifest=manifest,
            identity=read(directory / 'queue_identity.json'))
            for name, directory, state, manifest in (
                (ACTIVE, active_dir, active, active_manifest),
                (LEVELS, levels_dir, levels, levels_manifest),
                (CONTRAST, contrast_dir, contrast, contrast_manifest))}))

    directory = ROOT / f'work_dirs/{METHOD}_queue'
    assert not directory.exists()
    directory.mkdir()
    frozen = directory / f'{METHOD}_640_bs36.py'
    frozen.write_bytes(Path(checks['formal_config']).read_bytes())
    hashes = dict(active_manifest['source_sha256'])
    hashes.update(checks['source_sha256'])
    for path in (Path(__file__).resolve(), frozen):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    common = [PYTHON, '-u', str(ROOT / 'scripts/instance_semantic_clstower_alpha1_probe.py'),
              '--config', str(frozen)]
    manifest = dict(root=str(ROOT), python=PYTHON, gpu_uuid=active_manifest['gpu_uuid'],
        min_free_mib=46000, dependency=dependency(ACTIVE, active, active_dir),
        experiment=dict(name=f'{METHOD}_640_bs36',
            source_config=str(ROOT / f'configs/{METHOD}/rtmdet_tiny_dior_640.py'),
            config=str(frozen), work_dir=str(ROOT / f'work_dirs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36'),
            log=str(ROOT / f'logs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36.log'),
            preflight_report=str(validation / 'queue_preflight.json'),
            capacity_report=str(validation / 'capacity_gpu1_bs36.json'),
            validation_summary=str(validation / 'checks.json')),
        source_sha256=hashes,
        preflight_command=common + ['--mode', 'preflight', '--report', str(validation / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--batch', '36', '--allocator-fraction', '.88',
                                  '--report', str(validation / 'capacity_gpu1_bs36.json')],
        capacity_expected=dict(method=METHOD, microbatch=36, formal_batch=36, accumulation=1,
            feature_gradient_scale=1., classification_roi_gradient_scale=1.,
            backbone_roi_gradient_scale=0., neck_roi_gradient_scale=0.,
            auxiliary_classifier_weight=.1, optimizer_updates=3, gt_per_image=1200,
            allocator_fraction=.88, formal_batch_capacity_tested=True,
            amp_skipped_updates=0, saved_checkpoints=False, watched_parameter_tensors=145,
            cudnn_benchmark=True, config_sha256=digest(frozen)))
    new_state = launch(directory, manifest)
    # Stop leaf first: an unchanged child must not interpret a deliberately
    # replaced parent scheduler as a failed predecessor.
    stop_waiter(contrast_dir, contrast)
    stop_waiter(levels_dir, levels)
    levels_manifest['dependency'] = dependency(METHOD, new_state, directory)
    new_levels = launch(levels_dir, levels_manifest)
    contrast_manifest['dependency'] = dependency(LEVELS, new_levels, levels_dir)
    new_contrast = launch(contrast_dir, contrast_manifest)

    assert process_alive(trainer_pid, trainer_ticks)
    active_now = read(active_dir / 'status.json')
    assert active_now['state'] == 'training' and active_now['child_pid'] == trainer_pid
    chain = []
    for method, folder, state, current_manifest in (
            (ACTIVE, active_dir, active_now, active_manifest),
            (METHOD, directory, new_state, manifest),
            (LEVELS, levels_dir, new_levels, levels_manifest),
            (CONTRAST, contrast_dir, new_contrast, contrast_manifest)):
        check_sources(current_manifest)
        assert process_alive(state['queue_pid'], state['queue_start_ticks'])
        chain.append(dict(method=method, state=state['state'], queue_pid=state['queue_pid'],
            queue_start_ticks=state['queue_start_ticks'], dependency=current_manifest['dependency']['name'],
            manifest=str(folder / 'manifest.json')))
    checks.update(queued=True, queue_manifest=str(directory / 'manifest.json'),
        formal_capacity=dict(status='pending', formal_batch=36, accumulation=1))
    write(validation / 'checks.json', checks)
    for method in (LEVELS, CONTRAST):
        path = ROOT / f'work_dirs/{method}_validation/checks.json'
        row = read(path)
        row['queue_manifest'] = str(ROOT / f'work_dirs/{method}_queue/manifest.json')
        write(path, row)
    summary_path = ROOT / 'work_dirs/roi_auxiliary_variants_validation/checks.json'
    summary = read(summary_path)
    summary.update(queue_order=[row['method'] for row in chain], queues=chain)
    write(summary_path, summary)
    previous_chain = ROOT / 'work_dirs/diagnostics/roi_auxiliary_variants_enqueue/verified_chain.json'
    write(AUDIT / 'previous_verified_chain.json', read(previous_chain))
    write(previous_chain, chain)
    write(AUDIT / 'verified_chain.json', dict(preserved_trainer_pid=trainer_pid,
        preserved_trainer_ticks=trainer_ticks, chain=chain))
    print('INSERTION_VERIFIED; original trainer preserved', trainer_pid, flush=True)


if __name__ == '__main__':
    main()
