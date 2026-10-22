"""Append SCDown4+G2 after the existing P3 control B, preserving live jobs."""
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
METHOD = 'scdown_p3_g2'
sys.path.insert(0, str(ROOT))
from scripts.queue_experiment import check_sources, completed, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dependency(method):
    directory = ROOT / f'work_dirs/{method}_queue'
    state, identity = read(directory / 'status.json'), read(directory / 'queue_identity.json')
    assert state['queue_pid'] == identity['pid']
    assert state['queue_start_ticks'] == identity['start_ticks']
    assert process_alive(identity['pid'], identity['start_ticks']), method
    command = Path(f'/proc/{identity["pid"]}/cmdline').read_bytes().split(b'\0')
    assert str(directory / 'manifest.json').encode() in command
    check_sources(read(directory / 'manifest.json'))
    return dict(name=method, pid=identity['pid'], start_ticks=identity['start_ticks'],
                status=str(directory / 'status.json'), work_dir=state['experiment']['work_dir'],
                log=state['experiment']['log'])


def main():
    existing = ('scdown_dsconv', 'scdown_p3_dense', 'scdown_p3_noact')
    before = {name: dependency(name) for name in existing}
    active = read(before[existing[0]]['status'])
    assert active['state'] == 'training' and Path(f'/proc/{active["child_pid"]}').exists()
    for method in ('scdown_g2', 'scdown_p3'):
        state = read(ROOT / f'work_dirs/{method}_queue/status.json')
        assert state['state'] == 'complete'
        assert completed(state['experiment']['work_dir'], state['experiment']['log'])
    predecessor = before['scdown_p3_noact']
    template = read(ROOT / 'work_dirs/scdown_p3_noact_queue/manifest.json')
    validation = ROOT / f'work_dirs/{METHOD}_validation'
    structure = read(validation / 'preflight.json')
    smoke = read(validation / 'smoke_gpu0.json')
    capacity = read(validation / 'capacity_gpu0_bs8.json')
    assert structure['status'] == smoke['status'] == capacity['status'] == 'passed'
    assert structure['parameters'] == 3342228 and structure['watched_parameter_tensors'] == 45
    assert smoke['optimizer_updates'] == 16 and smoke['val_passes'] == 2
    assert smoke['amp_skipped_updates'] == 0 and capacity['amp_skipped_updates'] == 0
    assert capacity['microbatch'] == 8 and capacity['optimizer_updates'] == 3
    assert capacity['gt_per_image'] == 1200 and not capacity['formal_batch_capacity_tested']
    assert not (validation / 'smoke').exists()
    unit_log = validation / 'unit_tests.log'
    unit_text = unit_log.read_text()
    assert '\nRan 5 tests in ' in unit_text and unit_text.rstrip().endswith('OK')

    sources = set()
    for name in ('scdown_g2', 'scdown_p3_noact'):
        for source in read(ROOT / f'work_dirs/{name}_queue/manifest.json')['source_sha256']:
            if source.startswith('work_dirs/') or 'scdown_p3_noact' in source or 'scdown_p3_controls' in source:
                continue
            sources.add(source)
    sources.update(('configs/scdown_p3_g2/rtmdet_tiny_dior_640.py',
                    'scripts/scdown_p3_g2_probe.py', 'tests/test_scdown_p3_g2.py',
                    'work_dirs/scdown_p3_g2_validation/formal_640_bs36.py'))
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT / 'custom_models/scdown_p3_g2').glob('*.py'))
    hashes = {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sorted(sources)}
    formal = validation / 'formal_640_bs36.py'
    assert hashlib.sha256(formal.read_bytes()).hexdigest() == structure['config_sha256'] == capacity['config_sha256']
    checks = dict(status='passed', method=METHOD, reference='scdown_p3',
        unit_tests=dict(status='passed', test_cases=5,
            command=[PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_scdown_p3_g2.py', '-v'],
            log=str(unit_log)),
        structure=structure, official_gpu_smoke=smoke, shared_gpu_capacity=capacity,
        formal_config=str(formal), source_sha256=hashes, queued=False,
        formal_capacity=dict(status='pending', formal_batch=36,
            reason='Runs on GPU1 after predecessor succeeds, before official training'))
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
    common = [PYTHON, '-u', str(ROOT / 'scripts/scdown_p3_g2_probe.py'),
              '--method', METHOD, '--config', str(frozen)]
    manifest = dict(root=str(ROOT), python=PYTHON, gpu_uuid=template['gpu_uuid'], min_free_mib=46000,
        dependency=predecessor,
        experiment=dict(name=f'{METHOD}_640_bs36',
            source_config=str(ROOT / f'configs/{METHOD}/rtmdet_tiny_dior_640.py'),
            config=str(frozen), work_dir=str(work), log=str(log),
            preflight_report=str(validation / 'queue_preflight.json'),
            capacity_report=str(validation / 'capacity_gpu1_bs36.json'),
            validation_summary=str(validation / 'checks.json')),
        source_sha256=queue_hashes,
        preflight_command=common + ['--mode', 'preflight', '--report', str(validation / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--report', str(validation / 'capacity_gpu1_bs36.json'),
                                   '--allocator-fraction', '.88'],
        capacity_expected=dict(method=METHOD, microbatch=36, formal_batch=36, accumulation=1,
            optimizer_updates=3, gt_per_image=1200, allocator_fraction=.88,
            formal_batch_capacity_tested=True, amp_skipped_updates=0, saved_checkpoints=False,
            watched_parameter_tensors=45, config_sha256=queue_hashes[str(frozen.relative_to(ROOT))]))
    check_sources(manifest)
    write(directory / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'), str(directory / 'manifest.json')]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (directory / 'scheduler.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    write(directory / 'queue_identity.json', dict(pid=process.pid, start_ticks=stat[19],
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    expected = 'waiting_' + predecessor['name']
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((directory / 'scheduler.log').read_text()[-5000:])
        status_path = directory / 'status.json'
        if status_path.exists():
            state = read(status_path)
            if state['state'] == 'failed':
                raise RuntimeError(state)
            if state['state'] == expected:
                break
        time.sleep(1)
    else:
        raise RuntimeError('New scheduler did not finish startup verification')
    queued_preflight = read(validation / 'queue_preflight.json')
    assert queued_preflight['status'] == 'passed'
    assert queued_preflight['config_sha256'] == manifest['capacity_expected']['config_sha256']
    checks.update(queued=True, queue_manifest=str(directory / 'manifest.json'), queued_config=str(frozen))
    write(validation / 'checks.json', checks)
    chain = []
    for method in (*existing, METHOD):
        identity = dependency(method)
        state = read(identity['status'])
        if method in before:
            assert identity == before[method]
        if chain:
            assert state['dependency'] == chain[-1]['identity']
            assert state['state'] == 'waiting_' + chain[-1]['identity']['name']
            assert not state.get('child_pid')
        else:
            assert state['state'] == 'training' and state['child_pid'] == active['child_pid']
        chain.append(dict(identity=identity, state=state['state']))
    write(ROOT / 'work_dirs/diagnostics/scdown_p3_g2_enqueue/verified_chain.json',
          dict(status='passed', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               preserved_trainer_pid=active['child_pid'], chain=chain))
    print('QUEUE_APPEND_VERIFIED', METHOD, process.pid, expected, flush=True)


if __name__ == '__main__':
    main()
