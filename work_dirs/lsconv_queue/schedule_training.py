"""Wait for verified EFS completion, check capacity, run official LSConv training."""
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

DIRECTORY = Path(__file__).resolve().parent


def process_alive(pid, start_ticks):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return stat[0] != 'Z' and stat[19] == str(start_ticks)
    except FileNotFoundError:
        return False


def completed(work_dir, log):
    checkpoint = Path(work_dir) / 'epoch_300.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not Path(log).is_file():
        return False
    with Path(log).open('rb') as handle:
        handle.seek(max(0, Path(log).stat().st_size - 512 * 1024))
        tail = handle.read().decode(errors='replace')
    if re.search(r'Traceback|CUDA out of memory|loss\w*:\s*(nan|inf)', tail):
        return False
    if not re.search(r'Epoch\(val\)\s+\[300\]\[\d+/\d+\].*pascal_voc/mAP:', tail):
        return False
    for scalar_file in Path(work_dir).glob('*/vis_data/scalars.json'):
        for line in scalar_file.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('step') == 300 and all(
                    key in row and math.isfinite(row[key])
                    for key in ('pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50')):
                return True
    return False


def main():
    manifest = json.loads((DIRECTORY / 'manifest.json').read_text())
    lock = (DIRECTORY / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def record(state, **details):
        status = dict(state=state, queue_pid=os.getpid(),
                      updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      dependency=manifest['dependency'], experiment=manifest['experiment'], **details)
        temporary = DIRECTORY / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(DIRECTORY / 'status.json')
        print(json.dumps(status), flush=True)
    def wait_for_memory():
        waiting = False
        while True:
            free = int(subprocess.check_output([
                'nvidia-smi', '--id=' + manifest['gpu_uuid'],
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
            if free >= manifest['min_free_mib']:
                return
            if not waiting:
                record('waiting_gpu_memory', free_mib=free, required_mib=manifest['min_free_mib'])
                waiting = True
            time.sleep(20)
    try:
        dependency = manifest['dependency']
        record('waiting_efs')
        while process_alive(dependency['pid'], dependency['start_ticks']):
            time.sleep(20)
        if not completed(dependency['work_dir'], dependency['log']):
            raise RuntimeError('EFS did not complete finite epoch-300 validation; LSConv not started')
        experiment = manifest['experiment']
        if Path(experiment['work_dir']).exists() or Path(experiment['log']).exists():
            raise RuntimeError('Existing LSConv outputs would be overwritten')
        environment = os.environ.copy()
        environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],
                           PYTHONPATH=manifest['root'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        wait_for_memory()
        with (DIRECTORY / 'capacity.log').open('w') as log:
            command = [manifest['python'], '-u', str(DIRECTORY / 'capacity.py'),
                       experiment['config'], experiment['capacity_report']]
            probe = subprocess.Popen(command, cwd=manifest['root'], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record('checking_batch36_capacity', probe_pid=probe.pid)
            code = probe.wait()
        if code != 0:
            raise RuntimeError(f'LSConv capacity check failed (exit {code}); training not started')
        capacity = json.loads(Path(experiment['capacity_report']).read_text())
        if capacity['status'] != 'passed':
            raise RuntimeError('LSConv capacity check did not pass')
        summary_path = Path(manifest['root']) / 'work_dirs/lsconv_validation/checks.json'
        summary = json.loads(summary_path.read_text())
        summary['batch36_capacity'] = capacity
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        wait_for_memory()
        command = [manifest['python'], '-u', str(Path(manifest['root']) / 'mmyolo/tools/train.py'),
                   experiment['config'], '--work-dir', experiment['work_dir']]
        with Path(experiment['log']).open('x') as log:
            training = subprocess.Popen(command, cwd=manifest['root'], env=environment,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            record('training', training_pid=training.pid, command=command)
            code = training.wait()
        if code != 0 or not completed(experiment['work_dir'], experiment['log']):
            raise RuntimeError(f'LSConv training failed (exit {code}) or final validation is incomplete')
        record('complete')
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
