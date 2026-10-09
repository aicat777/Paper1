#!/usr/bin/env python3
"""Train the DIOR baseline with validation, durable status, timing and checkpoints."""
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work_dirs' / 'rtmdet_tiny_dior_scratch_300e_gpu1'
CONFIG = ROOT / 'configs' / 'rtmdet_tiny_dior_300e.py'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path[:0] = [str(ROOT / 'mmyolo'), str(ROOT / 'mmdetection')]
TZ = ZoneInfo('Asia/Shanghai')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / '.training.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A full DIOR training process is already running.', flush=True)
        return 2
    started = time.time()
    status = {'state': 'initializing', 'pid': os.getpid(), 'gpu_index': 1,
              'config': str(CONFIG), 'work_dir': str(WORK),
              'started_at': datetime.now(TZ).isoformat(), 'started_unix_time': started,
              'max_epochs': 300, 'completed_epochs': 0, 'train_images': 5862,
              'val_images': 5863, 'batch_size': 16, 'amp': True,
              'initialization': 'random', 'pretrained': False, 'seed': 42,
              'validation_epochs': [1] + list(range(10, 280, 10)) + list(range(280, 301)),
              'epoch_history': [], 'validation_history': [],
              'initial_estimate_hours': [13, 15], 'exit_code': None}
    if '--resume' in sys.argv and (WORK / 'status.json').exists():
        previous = json.loads((WORK / 'status.json').read_text())
        if previous.get('initialization') != 'random':
            raise RuntimeError('Only this experiment\'s randomly initialized checkpoints may be resumed')
        for key in ('epoch_history', 'validation_history', 'completed_epochs',
                    'best_mAP', 'best_AP50', 'latest_validation'):
            if key in previous:
                status[key] = previous[key]
        status['attempts'] = previous.get('attempts', []) + [{
            'pid': previous['pid'], 'state': previous['state'],
            'started_at': previous['started_at'],
            'finished_at': previous.get('finished_at'), 'error': previous.get('error')}]
    write_json(WORK / 'status.json', status)
    (WORK / 'train.pid').write_text(str(os.getpid()) + '\n')
    exit_code = 1
    try:
        # Resolve physical GPU 1 to its UUID, independent of CUDA enumeration.
        os.environ['CUDA_VISIBLE_DEVICES'] = subprocess.check_output(
            ['nvidia-smi', '-i', '1', '--query-gpu=uuid',
             '--format=csv,noheader'], text=True).strip()
        import torch
        from mmengine.config import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        from mmyolo.utils import register_all_modules

        total, used = [float(value) for value in subprocess.check_output(
            ['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used',
             '--format=csv,noheader,nounits'], text=True).strip().split(',')]
        free = total - used
        status['gpu_initial_free_mib'] = free
        if free < 8192:
            raise RuntimeError(f'GPU 1 has only {free:.0f} MiB free; training not started')
        assert torch.cuda.device_count() == 1
        properties = torch.cuda.get_device_properties(0)
        budget_mib = min(10240, free - 2048)
        torch.cuda.set_per_process_memory_fraction(budget_mib * 1024 ** 2 / properties.total_memory)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        status['pytorch_allocator_budget_mib'] = budget_mib
        status['torch'] = torch.__version__
        status['gpu_name'] = properties.name
        register_all_modules()
        config = Config.fromfile(CONFIG)
        config.work_dir = str(WORK)
        config.launcher = 'none'
        from initialization import (assert_random_initialization,
                                    assert_model_random_initialization)
        assert_random_initialization(config)
        if '--resume' in sys.argv:
            config.resume = True
            config.load_from = None
        elif (WORK / 'last_checkpoint').exists():
            raise RuntimeError('Existing training checkpoints found; explicitly use --resume')
        assert all(name == name.lower() for name in config.train_dataloader.dataset.metainfo.classes)
        assert config.train_dataloader.dataset.ann_file != config.val_dataloader.dataset.ann_file
        runner_metadata = dict(initialization='random', pretrained=False,
                               dataset='DIOR', seed=config.randomness.seed)
        status['git_commit'] = subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
            capture_output=True).stdout.strip() or None
        write_json(WORK / 'status.json', status)

        def update_estimate():
            recent = status['epoch_history'][-10:]
            if not recent:
                return
            seconds = statistics.mean(item['seconds'] for item in recent)
            mean_val = statistics.mean(item['seconds'] for item in status['validation_history']) if status['validation_history'] else 381
            completed_val = {item['epoch'] for item in status['validation_history']}
            remaining_val = sum(epoch > status['completed_epochs'] or
                (epoch == status['completed_epochs'] and epoch not in completed_val)
                for epoch in status['validation_epochs'])
            remaining = (status['max_epochs'] - status['completed_epochs']) * seconds + remaining_val * mean_val
            status.update({'mean_recent_epoch_seconds': seconds,
                           'mean_validation_seconds': mean_val,
                           'remaining_validation_runs': remaining_val,
                           'estimated_remaining_seconds': remaining,
                           'estimated_finish_at': (datetime.now(TZ) + timedelta(seconds=remaining)).isoformat()})

        class RunStatusHook(Hook):
            priority = 'HIGH'

            def before_save_checkpoint(self, runner, checkpoint):
                checkpoint.setdefault('meta', {}).update(runner_metadata)
                checkpoint['meta']['project_git_commit'] = status['git_commit']

            def before_train_epoch(self, runner):
                self.start = time.perf_counter()
                self.peak = self.reserved = 0
                torch.cuda.reset_peak_memory_stats()
                status.update({'state': 'training', 'current_epoch': runner.epoch + 1,
                               'iteration_in_epoch': 0})
                write_json(WORK / 'status.json', status)

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                self.peak = max(self.peak, torch.cuda.max_memory_allocated())
                self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
                for name, value in outputs.items():
                    if name.startswith('loss'):
                        number = float(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
                        if not math.isfinite(number):
                            raise RuntimeError(f'Non-finite {name} in epoch {runner.epoch + 1}')
                if (batch_idx + 1) % 50 == 0:
                    status.update({'iteration_in_epoch': batch_idx + 1,
                                   'train_loss': {name: float(value) for name, value in outputs.items()},
                                   'peak_allocated_mib_current_epoch': self.peak / 1024 ** 2})
                    write_json(WORK / 'status.json', status)

            def after_train_epoch(self, runner):
                torch.cuda.synchronize()
                item = {'epoch': runner.epoch + 1, 'seconds': time.perf_counter() - self.start,
                        'peak_allocated_mib': self.peak / 1024 ** 2,
                        'peak_reserved_mib': self.reserved / 1024 ** 2}
                status['epoch_history'].append(item)
                status['completed_epochs'] = runner.epoch + 1
                update_estimate()
                write_json(WORK / 'status.json', status)
                runner.logger.info(f'TRAIN_TIME epoch={item["epoch"]} seconds={item["seconds"]:.2f}')

            def before_val(self, runner):
                torch.cuda.synchronize()
                self.val_start = time.perf_counter()
                status['state'] = 'validating'
                write_json(WORK / 'status.json', status)

            def after_val_epoch(self, runner, metrics=None):
                torch.cuda.synchronize()
                values = {name: float(value) for name, value in (metrics or {}).items()}
                item = {'epoch': runner.epoch, 'seconds': time.perf_counter() - self.val_start,
                        'metrics': values}
                status['validation_history'].append(item)
                status['latest_validation'] = item
                for key, output_key in [('coco/bbox_mAP', 'best_mAP'), ('coco/bbox_mAP_50', 'best_AP50')]:
                    if key in values and values[key] > status.get(output_key, {}).get('score', -1):
                        status[output_key] = {'epoch': runner.epoch, 'score': values[key]}
                status['state'] = 'between_epochs'
                update_estimate()
                write_json(WORK / 'status.json', status)
                write_json(WORK / 'validation_results.json', status['validation_history'])
                runner.logger.info(f'VAL_TIME epoch={runner.epoch} seconds={item["seconds"]:.2f}; ETA={status.get("estimated_finish_at")}')

        runner = Runner.from_cfg(config)
        assert_model_random_initialization(runner.model)
        runner.logger.info('INITIALIZATION: random; load_from=None; no pretrained init_cfg; seed=42')
        runner.register_hook(RunStatusHook())
        if '--validate-first' in sys.argv:
            class RecoveryValidationHook(Hook):
                priority = 'VERY_LOW'

                def before_train(self, runner):
                    runner.logger.info('Re-running validation at resumed epoch before continuing training.')
                    runner.val_loop.run()
            runner.register_hook(RecoveryValidationHook())
        runner.train()
        assert status['completed_epochs'] == 300
        assert status['validation_history'][-1]['epoch'] == 300
        status['state'] = 'completed'
        exit_code = 0
    except BaseException as error:
        status['state'] = 'failed'
        status['error_type'] = type(error).__name__
        status['error'] = str(error)
        traceback.print_exc()
    finally:
        status['exit_code'] = exit_code
        status['finished_at'] = datetime.now(TZ).isoformat()
        status['elapsed_seconds'] = time.time() - started
        status['checkpoint_files'] = [str(path) for path in WORK.glob('*.pth')]
        write_json(WORK / 'status.json', status)
        print('FULL_TRAINING_RESULT ' + json.dumps(status, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
