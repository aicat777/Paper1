"""Check SCDown4+G8 on shared GPU0, leaving GPU1 training untouched."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/home/pwt/Paper1')
PYTHON='/home/pwt/miniconda3/envs/rtmdet/bin/python'
METHOD='scdown_p3_g8'
validation=ROOT/f'work_dirs/{METHOD}_validation'
unit=(validation/'unit_tests.log').read_text()
assert '\nRan 5 tests in ' in unit and unit.rstrip().endswith('OK')
environment=os.environ.copy()
environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='GPU-baa609c6-d42c-136e-0555-5548678429b9',
                   PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
common=[PYTHON,'-u',str(ROOT/f'scripts/{METHOD}_probe.py'),
        '--method',METHOD,'--config',str(validation/'formal_640_bs36.py')]
for mode,filename,extras in [('preflight','preflight',[]),
                           ('smoke','smoke_gpu0',['--allocator-fraction','.22']),
                           ('capacity','capacity_gpu0_bs8',['--allocator-fraction','.22','--batch','8'])]:
    report=validation/(filename+'.json')
    assert not report.exists(),report
    with (validation/(filename+'.log')).open('x') as output:
        result=subprocess.run(common+['--mode',mode,'--report',str(report)]+extras,
                              cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((validation/(filename+'.log')).read_text()[-5000:])
    assert json.loads(report.read_text())['status']=='passed'
    print(METHOD,mode,'passed',flush=True)
print('SCDOWN4_G8_VALIDATION_PASSED',flush=True)
