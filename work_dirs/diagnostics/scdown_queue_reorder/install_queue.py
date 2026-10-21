"""One-time installation of the user-authorized queue; preserve the active trainer."""
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
from scripts.scdown_variants_probe import SPECS
from scripts.queue_experiment import check_sources, process_alive


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dependency(name):
    d = ROOT / f'work_dirs/{name}_queue'
    identity = read(d / 'queue_identity.json')
    state = read(d / 'status.json')
    assert state['queue_pid'] == identity['pid']
    assert state['queue_start_ticks'] == identity['start_ticks']
    assert process_alive(identity['pid'], identity['start_ticks'])
    return dict(name=name, pid=identity['pid'], start_ticks=identity['start_ticks'],
                status=str(d / 'status.json'), work_dir=state['experiment']['work_dir'],
                log=state['experiment']['log'])


def launch(name, manifest):
    d = ROOT / f'work_dirs/{name}_queue'
    check_sources(manifest)
    assert not Path(manifest['experiment']['work_dir']).exists()
    assert not Path(manifest['experiment']['log']).exists()
    previous_identity = d / 'queue_identity.json'
    if previous_identity.exists():
        previous = read(previous_identity)
        assert not process_alive(previous['pid'], previous['start_ticks']), name
    write(d / 'manifest.json', manifest)
    command = [PYTHON, '-u', str(ROOT / 'scripts/queue_experiment.py'), str(d / 'manifest.json')]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (d / 'scheduler.log').open('a') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    write(previous_identity, dict(pid=process.pid, start_ticks=stat[19],
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), command=command))
    expected = 'waiting_' + manifest['dependency']['name']
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((d / 'scheduler.log').read_text()[-5000:])
        status = d / 'status.json'
        if status.exists():
            state = read(status)
            if state['queue_pid'] == process.pid:
                if state['state'] == 'failed':
                    raise RuntimeError(state)
                if state['state'] == expected:
                    break
        time.sleep(1)
    else:
        raise RuntimeError(f'{name} did not enter {expected}')
    (d / 'reorder_hold.json').unlink(missing_ok=True)
    print(name, process.pid, expected, flush=True)
    return dependency(name)


def new_manifest(method, predecessor, old_template):
    spec = SPECS[method]
    d = ROOT / f'work_dirs/{method}_queue'
    snapshot = d / f'{method}_640_bs36.py'
    val = ROOT / f'work_dirs/{method}_validation'
    for report in ('preflight.json', 'smoke_gpu0.json', 'capacity_gpu0_bs8.json'):
        assert read(val / report)['status'] == 'passed', (method, report)
    unit = read(ROOT / 'work_dirs/scdown_variants_validation/unit_tests.json')
    assert unit['status'] == 'passed'
    write(val / 'checks.json', dict(status='passed', method=method,
        unit_tests=unit, structure=read(val / 'preflight.json'),
        official_gpu_smoke=read(val / 'smoke_gpu0.json'),
        shared_gpu_capacity=read(val / 'capacity_gpu0_bs8.json'),
        formal_capacity=dict(status='pending', microbatch=36,
                            reason='Runs on GPU1 after predecessor completion, before training')))
    sources = {p for p in old_template['source_sha256'] if p.startswith(('mmyolo/', 'mmdetection/'))}
    sources.update(['configs/rtmdet_tiny_dior_300e.py', 'configs/scdown/rtmdet_tiny_dior_640.py',
                    f'configs/{method}/rtmdet_tiny_dior_640.py', 'custom_models/__init__.py',
                    'scripts/queue_experiment.py', 'scripts/scdown_variants_probe.py',
                    'scripts/scdown_probe.py', 'scripts/candidate_probe.py', 'scripts/candidate_capacity.py',
                    'tests/test_scdown_variants.py', 'tests/test_scdown.py',
                    str(snapshot.relative_to(ROOT))])
    for package in ('scdown', method):
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT / f'custom_models/{package}').glob('*.py'))
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(sources)}
    sha = hashes[str(snapshot.relative_to(ROOT))]
    common = [PYTHON, '-u', str(ROOT / 'scripts/scdown_variants_probe.py'), '--method', method,
              '--config', str(snapshot)]
    return dict(root=str(ROOT), python=PYTHON, gpu_uuid=old_template['gpu_uuid'], min_free_mib=46000,
        dependency=predecessor,
        experiment=dict(name=f'{method}_640_bs36',
            source_config=str(ROOT / f'configs/{method}/rtmdet_tiny_dior_640.py'), config=str(snapshot),
            work_dir=str(ROOT / f'work_dirs/{method}_rtmdet_tiny_dior_640_gpu1_bs36'),
            log=str(ROOT / f'logs/{method}_rtmdet_tiny_dior_640_gpu1_bs36.log'),
            preflight_report=str(val / 'queue_preflight.json'),
            capacity_report=str(val / 'capacity_gpu1_bs36.json'), validation_summary=str(val / 'checks.json')),
        source_sha256=hashes,
        preflight_command=common + ['--mode', 'preflight', '--report', str(val / 'queue_preflight.json')],
        capacity_command=common + ['--mode', 'capacity', '--report', str(val / 'capacity_gpu1_bs36.json'),
                                   '--allocator-fraction', '.88'],
        capacity_expected=dict(method=method, microbatch=36, formal_batch=36, accumulation=1,
            optimizer_updates=3, gt_per_image=1200, allocator_fraction=.88,
            formal_batch_capacity_tested=True, amp_skipped_updates=0, saved_checkpoints=False,
            watched_parameter_tensors=spec['watched'], config_sha256=sha))


def main():
    center = dependency('instance_semantic_center')
    assert center['pid'] == 3107492 and center['start_ticks'] == '43330188'
    assert read(center['status'])['state'] == 'training'
    assert Path('/proc/3110304').exists()
    old = {m: read(ROOT / f'work_dirs/{m}_queue/manifest.json')
           for m in ('instance_semantic_matched', 'instance_semantic_random')}
    for manifest in old.values():
        check_sources(manifest)
    predecessor = center
    for method in ('scdown_g2', 'scdown_p3'):
        manifest = new_manifest(method, predecessor, old['instance_semantic_matched'])
        predecessor = launch(method, manifest)
    for method in ('instance_semantic_matched', 'instance_semantic_random'):
        manifest = old[method]
        manifest['dependency'] = predecessor
        predecessor = launch(method, manifest)
    manifest = new_manifest('scdown_p2p3', predecessor, old['instance_semantic_matched'])
    launch('scdown_p2p3', manifest)
    names = ('instance_semantic_center', 'scdown_g2', 'scdown_p3',
             'instance_semantic_matched', 'instance_semantic_random', 'scdown_p2p3')
    chain = []
    for index, name in enumerate(names):
        state = read(ROOT / f'work_dirs/{name}_queue/status.json')
        identity = dependency(name)
        manifest = read(ROOT / f'work_dirs/{name}_queue/manifest.json')
        check_sources(manifest)
        if index:
            assert state['state'] == 'waiting_' + names[index - 1]
            assert state['dependency'] == chain[-1]['identity']
            assert not state.get('child_pid')
            pid = identity['pid']
            assert not Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        else:
            assert state['state'] == 'training' and state['child_pid'] == 3110304
        chain.append(dict(identity=identity, state=state['state']))
    write(ROOT / 'work_dirs/diagnostics/scdown_queue_reorder/verified_chain.json', dict(status='passed', chain=chain))
    print('QUEUE_ORDER_VERIFIED; original center trainer still alive', flush=True)


if __name__ == '__main__':
    main()
