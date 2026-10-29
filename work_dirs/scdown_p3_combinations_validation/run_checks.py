"""Validate new SCDown4 combinations on shared GPU0, preserving GPU1 training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/home/pwt/Paper1')
PYTHON='/home/pwt/miniconda3/envs/rtmdet/bin/python'
environment=os.environ.copy()
environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='GPU-baa609c6-d42c-136e-0555-5548678429b9',
                   PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for method in ['scdown_p3_dsconv','scdown_p3_g4']:
    directory=ROOT/f'work_dirs/{method}_validation'
    unit=(directory/'unit_tests.log').read_text()
    assert '\nRan 5 tests in ' in unit and unit.rstrip().endswith('OK')
    common=[PYTHON,'-u',str(ROOT/f'scripts/{method}_probe.py'),
            '--method',method,'--config',str(directory/'formal_640_bs36.py')]
    for mode,filename,extras in [('preflight','preflight',[]),
                               ('smoke','smoke_gpu0',['--allocator-fraction','.22']),
                               ('capacity','capacity_gpu0_bs8',['--allocator-fraction','.22','--batch','8'])]:
        report=directory/(filename+'.json')
        if method=='scdown_p3_g4' and mode=='preflight':
            data=json.loads(report.read_text())
            assert data['status']=='passed'
            assert data['config_sha256']==hashlib.sha256((directory/'formal_640_bs36.py').read_bytes()).hexdigest()
            print(method,mode,'existing passed report reused',flush=True)
            continue
        assert not report.exists(),report
        with (directory/(filename+'.log')).open('x') as output:
            result=subprocess.run(common+['--mode',mode,'--report',str(report)]+extras,
                                  cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((directory/(filename+'.log')).read_text()[-5000:])
        assert json.loads(report.read_text())['status']=='passed'
        print(method,mode,'passed',flush=True)
print('COMBINATIONS_VALIDATION_PASSED',flush=True)
