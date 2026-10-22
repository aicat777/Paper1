"""Insert G8/G16 after control B and before the unstarted SCDown4+G2 job."""
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


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def dependency(method):
    directory=ROOT/f'work_dirs/{method}_queue'
    state,identity=read(directory/'status.json'),read(directory/'queue_identity.json')
    assert state['queue_pid']==identity['pid']
    assert state['queue_start_ticks']==identity['start_ticks']
    assert process_alive(identity['pid'],identity['start_ticks']),method
    command=Path(f'/proc/{identity["pid"]}/cmdline').read_bytes().split(b'\0')
    assert str(directory/'manifest.json').encode() in command
    check_sources(read(directory/'manifest.json'))
    return dict(name=method,pid=identity['pid'],start_ticks=identity['start_ticks'],
                status=str(directory/'status.json'),work_dir=state['experiment']['work_dir'],log=state['experiment']['log'])


def prepare(method,template):
    validation=ROOT/f'work_dirs/{method}_validation'
    structure=read(validation/'preflight.json')
    smoke=read(validation/'smoke_gpu0.json')
    capacity=read(validation/'capacity_gpu0_bs8.json')
    assert structure['status']==smoke['status']==capacity['status']=='passed'
    assert structure['watched_parameter_tensors']==39
    assert smoke['optimizer_updates']==16 and smoke['val_passes']==2 and smoke['amp_skipped_updates']==0
    assert capacity['microbatch']==8 and capacity['optimizer_updates']==3 and capacity['gt_per_image']==1200
    assert not capacity['formal_batch_capacity_tested'] and capacity['amp_skipped_updates']==0
    assert not (validation/'smoke').exists()
    unit_log=ROOT/'work_dirs/scdown_group_sweep_validation/unit_tests.log'
    assert '\nRan 6 tests in ' in unit_log.read_text() and unit_log.read_text().rstrip().endswith('OK')
    sources=set()
    for reference in ('scdown_g2','scdown_g4'):
        for source in read(ROOT/f'work_dirs/{reference}_queue/manifest.json')['source_sha256']:
            if not source.startswith('work_dirs/'):
                sources.add(source)
    sources.update(('scripts/scdown_group_sweep_probe.py','tests/test_scdown_group_sweep.py',
                    f'work_dirs/{method}_validation/formal_640_bs36.py'))
    for candidate in ('scdown_g8','scdown_g16'):
        sources.add(f'configs/{candidate}/rtmdet_tiny_dior_640.py')
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/f'custom_models/{candidate}').glob('*.py'))
    hashes={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest() for source in sorted(sources)}
    formal=validation/'formal_640_bs36.py'
    assert hashes[str(formal.relative_to(ROOT))]==structure['config_sha256']==capacity['config_sha256']
    checks=dict(status='passed',method=method,reference='scdown',
        unit_tests=dict(status='passed',test_cases=6,log=str(unit_log),
            command=[PYTHON,'-m','unittest','discover','-s','tests','-p','test_scdown_group_sweep.py','-v']),
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
    common=[PYTHON,'-u',str(ROOT/'scripts/scdown_group_sweep_probe.py'),
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
            watched_parameter_tensors=39,config_sha256=queue_hashes[str(frozen.relative_to(ROOT))]))


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
            if state['state']=='failed':
                raise RuntimeError(state)
            if state['state']==expected:
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


def main():
    existing=('scdown_dsconv','scdown_p3_dense','scdown_p3_noact')
    before={method:dependency(method) for method in existing}
    active=read(before['scdown_dsconv']['status'])
    assert active['state']=='training' and Path(f'/proc/{active["child_pid"]}').exists()
    old=dependency('scdown_p3_g2')
    state=read(old['status'])
    assert state['state']=='waiting_scdown_p3_noact' and not state.get('child_pid')
    assert not Path(old['work_dir']).exists() and not Path(old['log']).exists()
    assert not Path(f'/proc/{old["pid"]}/task/{old["pid"]}/children').read_text().strip()
    directory=ROOT/'work_dirs/scdown_p3_g2_queue'
    old_manifest=read(directory/'manifest.json')
    write(ROOT/'work_dirs/diagnostics/scdown_group_sweep_enqueue/previous_combination_queue.json',
          dict(identity=old,status=state,manifest=old_manifest))
    for filename in ('preflight.log','queue_identity.json'):
        (ROOT/f'work_dirs/diagnostics/scdown_group_sweep_enqueue/previous_combination_{filename}').write_bytes((directory/filename).read_bytes())
    prepared={method:prepare(method,old_manifest) for method in ('scdown_g8','scdown_g16')}
    assert not (ROOT/'work_dirs/scdown_p3_g4_queue').exists()
    # Only the idle tail scheduler is stopped. It has never started a trainer.
    assert process_alive(old['pid'],old['start_ticks'])
    os.kill(old['pid'],signal.SIGTERM)
    for _ in range(100):
        if not process_alive(old['pid'],old['start_ticks']):
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Waiting combination scheduler did not exit')
    predecessor=before['scdown_p3_noact']
    for method in ('scdown_g8','scdown_g16'):
        predecessor=launch(method,prepared[method],predecessor)
    launch('scdown_p3_g2',old_manifest,predecessor,reusing=True)
    chain=[]
    for method in (*existing,'scdown_g8','scdown_g16','scdown_p3_g2'):
        identity=dependency(method)
        state=read(identity['status'])
        if method in before:
            assert identity==before[method]
        if chain:
            assert state['dependency']==chain[-1]['identity']
            assert state['state']=='waiting_'+chain[-1]['identity']['name']
            assert not state.get('child_pid')
        else:
            assert state['state']=='training' and state['child_pid']==active['child_pid']
        chain.append(dict(identity=identity,state=state['state']))
    write(ROOT/'work_dirs/diagnostics/scdown_group_sweep_enqueue/verified_chain.json',
          dict(status='passed',checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               preserved_trainer_pid=active['child_pid'],old_waiting_combination_pid=old['pid'],
               chain=chain,scdown_p3_g4_queued=False))
    print('GROUP_SWEEP_INSERTION_VERIFIED; original training and A/B preserved',flush=True)


if __name__=='__main__':
    main()
