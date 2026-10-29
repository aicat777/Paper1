"""Append checked SCDown4+G8 after the live ROI split-gradient experiment."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path('/home/pwt/Paper1')
PYTHON='/home/pwt/miniconda3/envs/rtmdet/bin/python'
METHOD='scdown_p3_g8'
PREDECESSOR='instance_semantic_split'
sys.path.insert(0,str(ROOT))
from scripts.queue_experiment import check_sources,process_alive


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
                status=str(directory/'status.json'),work_dir=state['experiment']['work_dir'],
                log=state['experiment']['log'])


def main():
    predecessor=dependency(PREDECESSOR)
    active=read(predecessor['status'])
    assert active['state']=='training'
    trainer_pid=active['child_pid']
    assert Path(f'/proc/{trainer_pid}').exists()
    trainer_command=Path(f'/proc/{trainer_pid}/cmdline').read_bytes()
    assert predecessor['work_dir'].encode() in trainer_command
    existing=[]
    for path in sorted((ROOT/'work_dirs').glob('*_queue/status.json')):
        state=read(path)
        if state.get('queue_start_ticks') and process_alive(state.get('queue_pid',-1),state['queue_start_ticks']):
            existing.append(path.parent.name)
    assert existing==[PREDECESSOR+'_queue'],existing
    template=read(ROOT/f'work_dirs/{PREDECESSOR}_queue/manifest.json')
    validation=ROOT/f'work_dirs/{METHOD}_validation'
    structure=read(validation/'preflight.json')
    smoke=read(validation/'smoke_gpu0.json')
    capacity=read(validation/'capacity_gpu0_bs8.json')
    assert structure['status']==smoke['status']==capacity['status']=='passed'
    assert structure['parameters']==2984532 and structure['inference_conv_linear_gmac']==6.66912384
    assert structure['watched_parameter_tensors']==smoke['watched_parameter_tensors']==capacity['watched_parameter_tensors']==45
    assert smoke['optimizer_updates']==16 and smoke['val_passes']==2 and smoke['amp_skipped_updates']==0
    assert capacity['microbatch']==8 and capacity['optimizer_updates']==3 and capacity['gt_per_image']==1200
    assert not capacity['formal_batch_capacity_tested'] and capacity['amp_skipped_updates']==0
    assert not (validation/'smoke').exists()
    unit_log=validation/'unit_tests.log'
    assert '\nRan 5 tests in ' in unit_log.read_text() and unit_log.read_text().rstrip().endswith('OK')
    sources=set()
    for reference in ('scdown_p3_g4','scdown_g8'):
        previous=read(ROOT/f'work_dirs/{reference}_queue/manifest.json')
        check_sources(previous)
        for source in previous['source_sha256']:
            if not source.startswith('work_dirs/'):
                sources.add(source)
    sources.update((f'configs/{METHOD}/rtmdet_tiny_dior_640.py',f'scripts/{METHOD}_probe.py',
                    f'tests/test_{METHOD}.py',f'work_dirs/{METHOD}_validation/formal_640_bs36.py'))
    sources.update(str(path.relative_to(ROOT)) for path in (ROOT/f'custom_models/{METHOD}').glob('*.py'))
    hashes={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest() for source in sorted(sources)}
    formal=validation/'formal_640_bs36.py'
    assert hashes[str(formal.relative_to(ROOT))]==structure['config_sha256']==capacity['config_sha256']
    checks=dict(status='passed',method=METHOD,reference='scdown_p3',
        unit_tests=dict(status='passed',test_cases=5,log=str(unit_log),
            command=[PYTHON,'-m','unittest','discover','-s','tests','-p',f'test_{METHOD}.py','-v']),
        structure=structure,official_gpu_smoke=smoke,shared_gpu_capacity=capacity,
        formal_config=str(formal),source_sha256=hashes,queued=False,
        formal_capacity=dict(status='pending',formal_batch=36,
            reason='Runs on GPU1 after predecessor succeeds, before official training'))
    write(validation/'checks.json',checks)
    directory=ROOT/f'work_dirs/{METHOD}_queue'
    work=ROOT/f'work_dirs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36'
    log=ROOT/f'logs/{METHOD}_rtmdet_tiny_dior_640_gpu1_bs36.log'
    assert not directory.exists() and not work.exists() and not log.exists()
    directory.mkdir()
    frozen=directory/f'{METHOD}_640_bs36.py'
    frozen.write_bytes(formal.read_bytes())
    queue_hashes=dict(hashes)
    queue_hashes[str(frozen.relative_to(ROOT))]=hashlib.sha256(frozen.read_bytes()).hexdigest()
    queue_hashes['scripts/queue_experiment.py']=hashlib.sha256((ROOT/'scripts/queue_experiment.py').read_bytes()).hexdigest()
    common=[PYTHON,'-u',str(ROOT/f'scripts/{METHOD}_probe.py'),
            '--method',METHOD,'--config',str(frozen)]
    manifest=dict(root=str(ROOT),python=PYTHON,gpu_uuid=template['gpu_uuid'],min_free_mib=46000,
        dependency=predecessor,
        experiment=dict(name=f'{METHOD}_640_bs36',source_config=str(ROOT/f'configs/{METHOD}/rtmdet_tiny_dior_640.py'),
            config=str(frozen),work_dir=str(work),log=str(log),
            preflight_report=str(validation/'queue_preflight.json'),
            capacity_report=str(validation/'capacity_gpu1_bs36.json'),validation_summary=str(validation/'checks.json')),
        source_sha256=queue_hashes,
        preflight_command=common+['--mode','preflight','--report',str(validation/'queue_preflight.json')],
        capacity_command=common+['--mode','capacity','--report',str(validation/'capacity_gpu1_bs36.json'),
                                 '--allocator-fraction','.88'],
        capacity_expected=dict(method=METHOD,microbatch=36,formal_batch=36,accumulation=1,
            optimizer_updates=3,gt_per_image=1200,allocator_fraction=.88,
            formal_batch_capacity_tested=True,amp_skipped_updates=0,saved_checkpoints=False,
            watched_parameter_tensors=45,config_sha256=queue_hashes[str(frozen.relative_to(ROOT))]))
    check_sources(manifest)
    write(directory/'manifest.json',manifest)
    command=[PYTHON,'-u',str(ROOT/'scripts/queue_experiment.py'),str(directory/'manifest.json')]
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (directory/'scheduler.log').open('x') as output:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                 stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path(f'/proc/{process.pid}/stat').read_text().rpartition(') ')[2].split()
    write(directory/'queue_identity.json',dict(pid=process.pid,start_ticks=stat[19],
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),command=command))
    expected='waiting_'+PREDECESSOR
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError((directory/'scheduler.log').read_text()[-6000:])
        if (directory/'status.json').exists():
            state=read(directory/'status.json')
            ours=(state.get('queue_pid')==process.pid and str(state.get('queue_start_ticks'))==stat[19])
            if ours and state['state']=='failed':
                raise RuntimeError(state)
            if ours and state['state']==expected and state['dependency']==predecessor:
                break
        time.sleep(1)
    else:
        raise RuntimeError('New queue did not finish startup verification')
    preflight=read(validation/'queue_preflight.json')
    assert preflight['status']=='passed' and preflight['config_sha256']==manifest['capacity_expected']['config_sha256']
    checks.update(queued=True,queue_manifest=str(directory/'manifest.json'),queued_config=str(frozen))
    write(validation/'checks.json',checks)
    assert dependency(PREDECESSOR)==predecessor
    state=read(predecessor['status'])
    assert state['state']=='training' and state['child_pid']==trainer_pid
    assert Path(f'/proc/{trainer_pid}/cmdline').read_bytes()==trainer_command
    current=dependency(METHOD)
    assert not read(current['status']).get('child_pid')
    write(ROOT/'work_dirs/diagnostics/scdown_p3_g8_enqueue/verified_chain.json',
          dict(status='passed',checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               preserved_trainer_pid=trainer_pid,chain=[
                   dict(identity=predecessor,state=state['state']),dict(identity=current,state=expected)]))
    print('QUEUE_APPEND_VERIFIED',METHOD,process.pid,expected,'preserved trainer',trainer_pid,flush=True)


if __name__=='__main__':
    main()
