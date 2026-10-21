"""One-off GPU1 queue; each experiment uses the official MMYOLO entry."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone

DIRECTORY = Path(__file__).resolve().parent
MANIFEST = json.loads((DIRECTORY / 'manifest.json').read_text())


def record(state, **details):
    payload = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                   queue_pid=os.getpid(), **details)
    temporary = DIRECTORY / 'status.tmp'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(DIRECTORY / 'status.json')
    print(json.dumps(payload), flush=True)


def process_alive(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return fields[0] != 'Z' and fields[19] == str(start_ticks)
    except FileNotFoundError:
        return False


def completed(entry):
    checkpoint = Path(entry['work_dir']) / 'epoch_300.pth'
    log = Path(entry['log'])
    if not checkpoint.is_file() or not log.is_file():
        return False
    with log.open('rb') as handle:
        handle.seek(max(0, log.stat().st_size - 512 * 1024))
        tail = handle.read().decode(errors='replace')
    final = re.search(r'Epoch\(val\)\s+\[300\]\[\d+/\d+\].*'
                      r'pascal_voc/mAP:\s*([0-9.]+)', tail)
    fatal = re.search(r'Traceback|CUDA out of memory|loss\w*:\s*(nan|inf)', tail)
    return bool(final and not fatal and checkpoint.stat().st_size > 0)


def free_memory_mib():
    result = subprocess.run(
        ['nvidia-smi', '--id=' + MANIFEST['gpu_uuid'],
         '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def main():
    # Prevent accidental duplicate dispatchers for this same sequence.
    lock = (DIRECTORY / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = MANIFEST['current']
    record('waiting_current', experiment=current['name'],
           training_pid=current['pid'], pending=[e['name'] for e in MANIFEST['pending']])
    while process_alive(current['pid'], current['start_ticks']):
        time.sleep(MANIFEST['poll_seconds'])
    if not completed(current):
        raise RuntimeError('Current full experiment did not finish epoch 300 validation; queue stopped')
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       CUDA_VISIBLE_DEVICES=MANIFEST['gpu_uuid'],
                       PYTHONPATH=MANIFEST['root'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for index, entry in enumerate(MANIFEST['pending']):
        free = free_memory_mib()
        if free < MANIFEST['min_free_mib']:
            record('waiting_gpu_memory', experiment=entry['name'], free_mib=free,
                   required_mib=MANIFEST['min_free_mib'])
            while free < MANIFEST['min_free_mib']:
                time.sleep(MANIFEST['poll_seconds'])
                free = free_memory_mib()
        work = Path(entry['work_dir'])
        if work.exists():
            raise RuntimeError(f'Output directory already exists: {work}')
        command = [MANIFEST['python'], '-u',
                   str(Path(MANIFEST['root']) / 'mmyolo/tools/train.py'),
                   entry['config'], '--work-dir', entry['work_dir']]
        with Path(entry['log']).open('x') as log:
            child = subprocess.Popen(command, cwd=MANIFEST['root'], env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
            record('training', experiment=entry['name'], training_pid=child.pid,
                   command=command, log=entry['log'],
                   pending=[e['name'] for e in MANIFEST['pending'][index + 1:]])
            returncode = child.wait()
        if returncode != 0 or not completed(entry):
            raise RuntimeError(f"{entry['name']} failed (exit {returncode}); queue stopped")
        record('experiment_completed', experiment=entry['name'])
    record('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('failed', error=str(error))
        raise
