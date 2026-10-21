"""Queue one official MMYOLO experiment after verified predecessor completion."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo


def process_alive(pid, start_ticks):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()
        return stat[0] != 'Z' and stat[19] == str(start_ticks)
    except FileNotFoundError:
        return False


def completed(work_dir, log, epoch=300):
    checkpoint = Path(work_dir) / f'epoch_{epoch}.pth'
    log = Path(log)
    if not checkpoint.is_file() or not checkpoint.stat().st_size or not log.is_file():
        return False
    final_validation = False
    fatal = re.compile(r'Traceback|CUDA out of memory|\bloss(?:_\w+)?:\s*[+-]?(?:nan|inf)\b', re.I)
    validation = re.compile(rf'Epoch\(val\)\s+\[{epoch}\]\[\d+/\d+\].*pascal_voc/mAP:')
    for line in log.read_text(errors='replace').splitlines():
        if fatal.search(line):
            return False
        final_validation |= bool(validation.search(line))
    if not final_validation:
        return False
    for path in Path(work_dir).glob('*/vis_data/scalars.json'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get('step') == epoch and all(
                        key in row and math.isfinite(row[key])
                        for key in ('pascal_voc/mAP', 'coco/bbox_mAP', 'coco/bbox_mAP_50')):
                    return True
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return False


def check_sources(manifest):
    for relative, expected in manifest['source_sha256'].items():
        actual = hashlib.sha256((Path(manifest['root']) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Queued source changed: ' + relative)


def validate_resume_checkpoint(experiment):
    """Require an explicit, complete training checkpoint from this experiment."""
    checkpoint = Path(experiment['resume_from']).resolve()
    work_dir = Path(experiment['work_dir']).resolve()
    if checkpoint.parent != work_dir:
        raise RuntimeError('Resume checkpoint must belong to the experiment work directory')
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise RuntimeError('Resume checkpoint is missing or empty')
    if not Path(experiment['log']).is_file():
        raise RuntimeError('Resume requires the existing experiment log')
    import torch
    saved = torch.load(checkpoint, map_location='cpu')
    required = ('meta', 'state_dict', 'optimizer', 'param_schedulers', 'ema_state_dict')
    if not isinstance(saved, dict) or any(key not in saved for key in required):
        raise RuntimeError('Resume requires model, optimizer, schedulers and EMA training state')
    epoch, iteration = saved['meta'].get('epoch'), saved['meta'].get('iter')
    if not isinstance(epoch, int) or not 0 < epoch < 300 or not isinstance(iteration, int) or iteration <= 0:
        raise RuntimeError('Resume checkpoint must contain an unfinished training epoch and iteration')
    return dict(checkpoint=str(checkpoint), epoch=epoch, iteration=iteration)


def run_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    lock = (directory / 'queue.lock').open('a')
    # A second scheduler must not overwrite the first scheduler's status.
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    stat = Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(') ')[2].split()
    start_ticks = stat[19]
    dependency, experiment = manifest['dependency'], manifest['experiment']
    resume_from = experiment.get('resume_from')
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],
                       PYTHONPATH=manifest['root'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def record(state, **details):
        value = dict(state=state, queue_pid=os.getpid(), queue_start_ticks=start_ticks,
                     updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                     dependency=dependency, experiment=experiment, **details)
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(directory / 'status.json')
        print(json.dumps(value), flush=True)

    def run_command(command, handle, state):
        process = subprocess.Popen(command, cwd=manifest['root'], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        record(state, child_pid=process.pid, command=command)
        code = process.wait()
        if code:
            raise RuntimeError(f'{state} failed with exit {code}; see {handle.name}')

    def require_available_outputs():
        if resume_from:
            if not Path(resume_from).is_file() or not Path(experiment['log']).is_file():
                raise RuntimeError('Resume checkpoint or existing log disappeared')
        elif Path(experiment['work_dir']).exists() or Path(experiment['log']).exists():
            raise RuntimeError('Existing experiment outputs would be overwritten')

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
        # A killed scheduler can leave its trainer alive. Do not launch a second
        # trainer merely because the queue lock was released.
        status_path = directory / 'status.json'
        if status_path.is_file():
            previous = json.loads(status_path.read_text())
            child_pid = previous.get('child_pid')
            if child_pid:
                try:
                    child_stat = Path(f'/proc/{child_pid}/stat').read_text().rpartition(') ')[2].split()
                    child_cmd = Path(f'/proc/{child_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                except FileNotFoundError:
                    pass
                else:
                    if child_stat[0] != 'Z' and experiment['work_dir'] in child_cmd:
                        raise RuntimeError('Previous trainer is still alive')
        require_available_outputs()
        check_sources(manifest)
        if resume_from:
            record('checking_resume_checkpoint', **validate_resume_checkpoint(experiment))
        # Exercise the real interpreter, entrypoint and custom imports now,
        # rather than discovering a startup error hours after the predecessor.
        with (directory / 'preflight.log').open('w') as handle:
            run_command([manifest['python'], '-u', str(Path(manifest['root']) / 'mmyolo/tools/train.py'),
                         '--help'], handle, 'checking_official_entrypoint')
            run_command(manifest['preflight_command'], handle, 'checking_config_and_structure')
        preflight = json.loads(Path(experiment['preflight_report']).read_text())
        if preflight.get('status') != 'passed':
            raise RuntimeError('Launch preflight did not pass')
        record('waiting_' + dependency['name'])
        while process_alive(dependency['pid'], dependency['start_ticks']):
            time.sleep(20)
        parent = json.loads(Path(dependency['status']).read_text())
        if parent.get('state') != 'complete' or parent.get('queue_pid') != dependency['pid']:
            raise RuntimeError('Predecessor failed or completion identity is incorrect')
        if 'queue_start_ticks' in parent and parent['queue_start_ticks'] != dependency['start_ticks']:
            raise RuntimeError('Predecessor completion belongs to another process')
        if not completed(dependency['work_dir'], dependency['log']):
            raise RuntimeError('Predecessor did not finish finite epoch-300 validation')
        require_available_outputs()
        wait_for_memory()
        check_sources(manifest)
        with (directory / 'capacity.log').open('w') as handle:
            run_command(manifest['capacity_command'], handle, 'checking_capacity')
        capacity = json.loads(Path(experiment['capacity_report']).read_text())
        if capacity.get('status') != 'passed' or any(
                capacity.get(key) != value for key, value in manifest['capacity_expected'].items()):
            raise RuntimeError('Capacity check did not pass for the queued training batch')
        summary_path = Path(experiment['validation_summary'])
        summary = json.loads(summary_path.read_text())
        summary['formal_capacity'] = capacity
        temporary = summary_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(summary_path)
        wait_for_memory()
        check_sources(manifest)
        require_available_outputs()
        command = [manifest['python'], '-u', str(Path(manifest['root']) / 'mmyolo/tools/train.py'),
                   experiment['config'], '--work-dir', experiment['work_dir']]
        if resume_from:
            command.extend(['--resume', resume_from])
        with Path(experiment['log']).open('a' if resume_from else 'x') as handle:
            run_command(command, handle, 'training')
        if not completed(experiment['work_dir'], experiment['log']):
            raise RuntimeError('Training exited without complete finite epoch-300 validation')
        record('complete')
    except Exception as error:
        record('failed', error=str(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    run_manifest(parser.parse_args().manifest)


if __name__ == '__main__':
    main()
