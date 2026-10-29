"""Keep A/B, then prioritize SCDown4+DSConv/G2/G4 before the G8/G16 sweep."""
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

ROOT=Path('/home/pwt/Paper1')
PYTHON='/home/pwt/miniconda3/envs/rtmdet/bin/python'
sys.path.insert(0,str(ROOT))
from scripts.queue_experiment import check_sources, process_alive
from work_dirs.diagnostics.scdown_group_sweep_enqueue.install_queue import dependency,read,write


def launch(method,manifest,predecessor,reusing=False):
    directory=ROOT/f'work_dirs/{method}_queue'
    manifest['dependency']=predecessor
    check_sources(manifest)
    write(directory/'manifest.json',manifest)
    command=[PYTHON,'-u',str(ROOT/'scripts/queue_experiment.py'),str(directory/'manifest.json')]
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (directory/'scheduler.log').open('a' if reusing else 'x') as output:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                 stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    write(directory/'queue_identity.json',dict(pid=process.pid,start_ticks=stat[19],
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),command=command))
    expected='waiting_'+predecessor['name']
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((directory/'scheduler.log').read_text()[-6000:])
        state_path=directory/'status.json'
        if state_path.exists():
            state=read(state_path)
            belongs_to_process = (state.get('queue_pid') == process.pid and
                                  str(state.get('queue_start_ticks')) == stat[19])
            if belongs_to_process and state['state']=='failed':
                raise RuntimeError(state)
            if (belongs_to_process and state['state']==expected and
                    state.get('dependency') == predecessor):
                break
        time.sleep(1)
    else:
        raise RuntimeError(f'{method} did not enter {expected}')
    report=read(manifest['experiment']['preflight_report'])
    assert report['status']=='passed' and report['config_sha256']==manifest['capacity_expected']['config_sha256']
    checks=read(manifest['experiment']['validation_summary'])
    checks.update(queued=True,queue_manifest=str(directory/'manifest.json'),queued_config=manifest['experiment']['config'])
    write(manifest['experiment']['validation_summary'],checks)
    print(method,process.pid,expected,flush=True)
    return dependency(method)


def prepare_new(method,template):
    validation=ROOT/f'work_dirs/{method}_validation'
    structure=read(validation/'preflight.json')
    smoke=read(validation/'smoke_gpu0.json')
    capacity=read(validation/'capacity_gpu0_bs8.json')
    assert structure['status']==smoke['status']==capacity['status']=='passed'
    watched=66 if method=='scdown_p3_dsconv' else 45
    assert structure['watched_parameter_tensors']==smoke['watched_parameter_tensors']==capacity['watched_parameter_tensors']==watched
    assert smoke['optimizer_updates']==16 and smoke['val_passes']==2 and smoke['amp_skipped_updates']==0
    assert capacity['microbatch']==8 and capacity['optimizer_updates']==3 and capacity['gt_per_image']==1200
    assert not capacity['formal_batch_capacity_tested'] and capacity['amp_skipped_updates']==0
    assert not (validation/'smoke').exists()
    unit_log=validation/'unit_tests.log'
    assert '\nRan 5 tests in ' in unit_log.read_text() and unit_log.read_text().rstrip().endswith('OK')
    sources=set()
    for manifest in (template,read(ROOT/'work_dirs/scdown_dsconv_queue/manifest.json')):
        for source in manifest['source_sha256']:
            if not source.startswith('work_dirs/') and 'scdown_p3_g2' not in source:
                sources.add(source)
    for source in (f'configs/{method}/rtmdet_tiny_dior_640.py',f'scripts/{method}_probe.py',
                   f'tests/test_{method}.py',f'work_dirs/{method}_validation/formal_640_bs36.py'):
        sources.add(source)
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/f'custom_models/{method}').glob('*.py'))
    if method=='scdown_p3_g4':
        sources.add('configs/scdown_g4/rtmdet_tiny_dior_640.py')
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'custom_models/scdown_g4').glob('*.py'))
    hashes={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest() for source in sorted(sources)}
    formal=validation/'formal_640_bs36.py'
    assert hashes[str(formal.relative_to(ROOT))]==structure['config_sha256']==capacity['config_sha256']
    checks=dict(status='passed',method=method,reference='scdown_p3',
        unit_tests=dict(status='passed',test_cases=5,log=str(unit_log),
            command=[PYTHON,'-m','unittest','discover','-s','tests','-p',f'test_{method}.py','-v']),
        structure=structure,official_gpu_smoke=smoke,shared_gpu_capacity=capacity,
        formal_config=str(formal),source_sha256=hashes,queued=False,
        formal_capacity=dict(status='pending',formal_batch=36,
            reason='Runs on GPU1 after predecessor succeeds, before official training'))
    write(validation/'checks.json',checks)
    directory=ROOT/f'work_dirs/{method}_queue'
    work=ROOT/f'work_dirs/{method}_rtmdet_tiny_dior_640_gpu1_bs36'
    log=ROOT/f'logs/{method}_rtmdet_tiny_dior_640_gpu1_bs36.log'
    assert not directory.exists() and not work.exists() and not log.exists()
    directory.mkdir()
    frozen=directory/f'{method}_640_bs36.py'
    frozen.write_bytes(formal.read_bytes())
    queue_hashes=dict(hashes)
    queue_hashes[str(frozen.relative_to(ROOT))]=hashlib.sha256(frozen.read_bytes()).hexdigest()
    common=[PYTHON,'-u',str(ROOT/f'scripts/{method}_probe.py'),
            '--method',method,'--config',str(frozen)]
    return dict(root=str(ROOT),python=PYTHON,gpu_uuid=template['gpu_uuid'],min_free_mib=46000,
        experiment=dict(name=f'{method}_640_bs36',source_config=str(ROOT/f'configs/{method}/rtmdet_tiny_dior_640.py'),
            config=str(frozen),work_dir=str(work),log=str(log),
            preflight_report=str(validation/'queue_preflight.json'),
            capacity_report=str(validation/'capacity_gpu1_bs36.json'),validation_summary=str(validation/'checks.json')),
        source_sha256=queue_hashes,
        preflight_command=common+['--mode','preflight','--report',str(validation/'queue_preflight.json')],
        capacity_command=common+['--mode','capacity','--report',str(validation/'capacity_gpu1_bs36.json'),
                                 '--allocator-fraction','.88'],
        capacity_expected=dict(method=method,microbatch=36,formal_batch=36,accumulation=1,
            optimizer_updates=3,gt_per_image=1200,allocator_fraction=.88,
            formal_batch_capacity_tested=True,amp_skipped_updates=0,saved_checkpoints=False,
            watched_parameter_tensors=watched,config_sha256=queue_hashes[str(frozen.relative_to(ROOT))]))


def verify_existing_schedule():
    expected=read(ROOT/'work_dirs/diagnostics/scdown_p3_combinations_enqueue/preserved_identities.json')
    methods=('scdown_p3_dense','scdown_p3_noact','scdown_p3_dsconv','scdown_p3_g2',
             'scdown_p3_g4','scdown_g8','scdown_g16')
    chain=[]
    for method in methods:
        identity=dependency(method)
        state=read(identity['status'])
        if method in expected['kept']:
            assert identity==expected['kept'][method]
        if chain:
            assert state['dependency']==chain[-1]['identity']
            assert state['state']=='waiting_'+chain[-1]['identity']['name']
            assert not state.get('child_pid')
        else:
            assert state['state']=='training' and state['child_pid']==expected['trainer_pid']
            assert Path(f"/proc/{state['child_pid']}").exists()
        manifest=read(ROOT/f'work_dirs/{method}_queue/manifest.json')
        report=read(manifest['experiment']['preflight_report'])
        assert report['status']=='passed'
        assert report['config_sha256']==manifest['capacity_expected']['config_sha256']
        chain.append(dict(identity=identity,state=state['state']))
    write(ROOT/'work_dirs/diagnostics/scdown_p3_combinations_enqueue/verified_chain.json',
          dict(status='passed',checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               preserved_trainer_pid=expected['trainer_pid'],
               kept_schedulers=list(expected['kept'].values()),chain=chain))
    print('COMBINATIONS_PRIORITY_VERIFIED; A/B and active training preserved',flush=True)


def main():
    kept=('scdown_p3_dense','scdown_p3_noact')
    reordered=('scdown_g8','scdown_g16','scdown_p3_g2')
    before={method:dependency(method) for method in (*kept,*reordered)}
    active=read(before['scdown_p3_dense']['status'])
    assert active['state']=='training' and Path(f'/proc/{active["child_pid"]}').exists()
    b=read(before['scdown_p3_noact']['status'])
    assert b['state']=='waiting_scdown_p3_dense' and not b.get('child_pid')
    assert b['dependency']==before['scdown_p3_dense']
    write(ROOT/'work_dirs/diagnostics/scdown_p3_combinations_enqueue/preserved_identities.json',
          dict(kept={name:before[name] for name in kept},trainer_pid=active['child_pid']))
    snapshots={}
    manifests={}
    for method in reordered:
        identity=before[method]
        state=read(identity['status'])
        assert state['state'].startswith('waiting_') and not state.get('child_pid')
        assert not Path(identity['work_dir']).exists() and not Path(identity['log']).exists()
        assert not Path(f'/proc/{identity["pid"]}/task/{identity["pid"]}/children').read_text().strip()
        directory=ROOT/f'work_dirs/{method}_queue'
        manifests[method]=read(directory/'manifest.json')
        snapshots[method]=dict(identity=identity,state=state,manifest=manifests[method])
        (ROOT/f'work_dirs/diagnostics/scdown_p3_combinations_enqueue/previous_{method}_preflight.log').write_bytes((directory/'preflight.log').read_bytes())
    write(ROOT/'work_dirs/diagnostics/scdown_p3_combinations_enqueue/previous_queues.json',snapshots)
    # Verify the existing validated G2 combination before changing its predecessor.
    checks=read(ROOT/'work_dirs/scdown_p3_g2_validation/checks.json')
    assert checks['official_gpu_smoke']['status']==checks['shared_gpu_capacity']['status']=='passed'
    for source,expected in checks['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source
    template=manifests['scdown_p3_g2']
    prepared={method:prepare_new(method,template) for method in ('scdown_p3_dsconv','scdown_p3_g4')}
    # Stop only idle schedulers, tail first, so no descendant sees a failed predecessor.
    for method in reversed(reordered):
        identity=before[method]
        assert process_alive(identity['pid'],identity['start_ticks'])
        os.kill(identity['pid'],signal.SIGTERM)
        for _ in range(100):
            if not process_alive(identity['pid'],identity['start_ticks']):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Idle scheduler did not exit: '+method)
    predecessor=before['scdown_p3_noact']
    for method in ('scdown_p3_dsconv','scdown_p3_g2','scdown_p3_g4','scdown_g8','scdown_g16'):
        manifest=prepared[method] if method in prepared else manifests[method]
        predecessor=launch(method,manifest,predecessor,reusing=method in manifests)
    verify_existing_schedule()


if __name__=='__main__':
    if sys.argv[1:]==['--verify-only']:
        verify_existing_schedule()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Only --verify-only is supported')
