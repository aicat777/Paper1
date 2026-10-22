"""Isolated G8/G16 audits on the shared GPU0; formal GPU1 jobs stay untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/pwt/Paper1')
PYTHON='/home/pwt/miniconda3/envs/rtmdet/bin/python'
unit=ROOT/'work_dirs/scdown_group_sweep_validation/unit_tests.log'
assert '\nRan 6 tests in ' in unit.read_text() and unit.read_text().rstrip().endswith('OK')
environment=os.environ.copy()
environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='GPU-baa609c6-d42c-136e-0555-5548678429b9',
                   PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for method in ['scdown_g8','scdown_g16']:
    directory=ROOT/f'work_dirs/{method}_validation'
    common=[PYTHON,'-u',str(ROOT/'scripts/scdown_group_sweep_probe.py'),
            '--method',method,'--config',str(directory/'formal_640_bs36.py')]
    for mode,filename,extras in [('preflight','preflight',[]),('smoke','smoke_gpu0',['--allocator-fraction','.22']),
                               ('capacity','capacity_gpu0_bs8',['--allocator-fraction','.22','--batch','8'])]:
        report=directory/(filename+'.json')
        assert not report.exists(),report
        with (directory/(filename+'.log')).open('x') as output:
            result=subprocess.run(common+['--mode',mode,'--report',str(report)]+extras,
                                  cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((directory/(filename+'.log')).read_text()[-5000:])
        data=json.loads(report.read_text())
        assert data['status']=='passed'
        print(method,mode,'passed',flush=True)
print('GROUP_SWEEP_VALIDATION_PASSED',flush=True)
