"""Append the two checked compression variants without restarting existing jobs."""
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
sys.path.insert(0, str(ROOT))
from scripts.queue_experiment import check_sources, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dependency(method):
    d = ROOT / f'work_dirs/{method}_queue'
    state, identity = read(d / 'status.json'), read(d / 'queue_identity.json')
    assert state['queue_pid'] == identity['pid']
    assert state['queue_start_ticks'] == identity['start_ticks']
    assert process_alive(identity['pid'], identity['start_ticks'])
    return dict(name=method, pid=identity['pid'], start_ticks=identity['start_ticks'],
                status=str(d / 'status.json'), work_dir=state['experiment']['work_dir'],
                log=state['experiment']['log'])


def append(method, predecessor, template):
    val = ROOT / f'work_dirs/{method}_validation'
    checks = read(val / 'checks.json')
    assert checks['status'] == checks['unit_tests']['status'] == 'passed'
    for key in ('structure', 'official_gpu_smoke', 'shared_gpu_capacity'):
        assert checks[key]['status'] == 'passed'
    for source, sha in checks['source_sha256'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == sha, source
    d = ROOT / f'work_dirs/{method}_queue'
    work = ROOT / f'work_dirs/{method}_rtmdet_tiny_dior_640_gpu1_bs36'
    log = ROOT / f'logs/{method}_rtmdet_tiny_dior_640_gpu1_bs36.log'
    assert not work.exists() and not log.exists()
    d.mkdir(exist_ok=False)
    frozen = d / f'{method}_640_bs36.py'
    frozen.write_bytes(Path(checks['formal_config']).read_bytes())
    hashes = dict(checks['source_sha256'])
    for source in (str(frozen.relative_to(ROOT)), 'scripts/queue_experiment.py'):
        hashes[source] = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    frozen_sha = hashes[str(frozen.relative_to(ROOT))]
    common = [PYTHON, '-u', str(ROOT / 'scripts/scdown_compression_probe.py'),
              '--method', method, '--config', str(frozen)]
    manifest = dict(root=str(ROOT), python=PYTHON, gpu_uuid=template['gpu_uuid'], min_free_mib=46000,
        dependency=predecessor,
        experiment=dict(name=f'{method}_640_bs36', source_config=str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'),
            config=str(frozen), work_dir=str(work), log=str(log),
            preflight_report=str(val / 'queue_preflight.json'),
            capacity_report=str(val / 'capacity_gpu1_bs36.json'), validation_summary=str(val / 'checks.json')),
        source_sha256=hashes,
        preflight_command=common + ['--mode', 'preflight', '--report', str(val / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--report', str(val / 'capacity_gpu1_bs36.json'),
                                   '--allocator-fraction', '.88'],
        capacity_expected=dict(method=method, microbatch=36, formal_batch=36, accumulation=1,
            optimizer_updates=3, gt_per_image=1200, allocator_fraction=.88,
            formal_batch_capacity_tested=True, amp_skipped_updates=0, saved_checkpoints=False,
            watched_parameter_tensors=checks['structure']['watched_parameter_tensors'], config_sha256=frozen_sha))
    check_sources(manifest)
    write(d / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'), str(d / 'manifest.json')]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (d / 'scheduler.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    write(d / 'queue_identity.json', dict(pid=process.pid, start_ticks=stat[19],
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    expected = 'waiting_' + predecessor['name']
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((d / 'scheduler.log').read_text()[-6000:])
        if (d / 'status.json').exists():
            state = read(d / 'status.json')
            if state['state'] == 'failed':
                raise RuntimeError(state)
            if state['state'] == expected:
                break
        time.sleep(1)
    else:
        raise RuntimeError(f'{method} did not enter {expected}')
    preflight = read(val / 'queue_preflight.json')
    assert preflight['status'] == 'passed' and preflight['config_sha256'] == frozen_sha
    checks['queued'] = True
    checks['queue_manifest'] = str(d / 'manifest.json')
    checks['queued_config'] = str(frozen)
    checks['formal_capacity'] = dict(status='pending', formal_batch=36,
        reason='Runs on GPU1 after predecessor completes, before official training')
    write(val / 'checks.json', checks)
    print(method, process.pid, expected, flush=True)
    return dependency(method)


def main():
    old = ('scdown_p3', 'instance_semantic_matched', 'instance_semantic_random', 'scdown_p2p3')
    for method in old:
        dependency(method)
        check_sources(read(ROOT / f'work_dirs/{method}_queue/manifest.json'))
    current = read(ROOT / 'work_dirs/scdown_p3_queue/status.json')
    assert current['state'] == 'training' and current['child_pid'] == 2503071
    assert Path('/proc/2503071').exists()
    predecessor = dependency('scdown_p2p3')
    template = read(ROOT / 'work_dirs/scdown_p2p3_queue/manifest.json')
    for method in ('scdown_g4', 'scdown_dsconv'):
        predecessor = append(method, predecessor, template)
    chain = []
    for method in (*old, 'scdown_g4', 'scdown_dsconv'):
        identity = dependency(method)
        state = read(identity['status'])
        check_sources(read(ROOT / f'work_dirs/{method}_queue/manifest.json'))
        if chain:
            assert state['dependency'] == chain[-1]['identity']
            assert state['state'] == 'waiting_' + chain[-1]['identity']['name']
            assert not state.get('child_pid')
            assert not Path(f'/proc/{identity["pid"]}/task/{identity["pid"]}/children').read_text().strip()
        else:
            assert state['state'] == 'training' and state['child_pid'] == 2503071
        chain.append(dict(identity=identity, state=state['state']))
    write(ROOT / 'work_dirs/diagnostics/scdown_compression_enqueue/verified_chain.json', dict(status='passed', chain=chain))
    print('QUEUE_APPEND_VERIFIED; original trainer and original schedulers preserved', flush=True)


if __name__ == '__main__':
    main()
