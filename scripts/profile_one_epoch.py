#!/usr/bin/env python3
"""Run exactly one DIOR train epoch; record speed/memory and remove outputs' checkpoints."""
import csv
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work_dirs' / 'rtmdet_tiny_dior_scratch_1epoch_gpu1'
sys.path[:0] = [str(ROOT / 'mmyolo'), str(ROOT / 'mmdetection')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def gpu_sample():
    command = ['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits']
    values = subprocess.check_output(command, text=True, timeout=10).strip().split(', ')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    process_memory = 0
    for line in processes.splitlines():
        pid, memory = [value.strip() for value in line.split(',')]
        if pid == str(os.getpid()) and memory.isdigit():
            process_memory = int(memory)
    return {'time': time.time(), 'total_mib': float(values[0]), 'used_mib': float(values[1]),
            'gpu_util_percent': float(values[2]), 'power_w': float(values[3]),
            'process_memory_mib': process_memory}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    trial_lock = (WORK / '.trial.lock').open('a')
    fcntl.flock(trial_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = {'status': 'starting', 'gpu_index': 1, 'pid': os.getpid(), 'dataset': 'DIOR original train split',
              'expected_images': 5862, 'batch_size': 16, 'input_size': [640, 640],
              'amp': True, 'num_workers': 8, 'validation_enabled': False, 'requested_epochs': 1,
              'initialization': 'random', 'pretrained': False}
    samples = []
    rows = []
    stop = threading.Event()
    monitor = None
    started = time.perf_counter()
    exit_code = 1
    try:
        import torch
        from mmengine.config import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        from mmyolo.utils import register_all_modules

        baseline = gpu_sample()
        report['gpu_before'] = baseline
        free_mib = baseline['total_mib'] - baseline['used_mib']
        if free_mib < 8192:
            raise RuntimeError(f'GPU 1 has only {free_mib:.0f} MiB free; trial stopped')
        assert torch.cuda.device_count() == 1
        properties = torch.cuda.get_device_properties(0)
        # Leave at least 2 GiB outside this trial's allocator for the existing process.
        budget_mib = min(10240, free_mib - 2048)
        torch.cuda.set_per_process_memory_fraction(budget_mib * 1024 ** 2 / properties.total_memory, 0)
        report['pytorch_allocator_budget_mib'] = budget_mib
        report['gpu_name'] = properties.name
        report['torch'] = torch.__version__
        report['cuda_runtime'] = torch.version.cuda
        report['environment_python'] = sys.executable
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        report['tf32_allowed'] = True
        # A failed CUDA check stops this attempt before building/training the model.
        probe = torch.ones((32, 32), device='cuda')
        _ = probe @ probe
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()

        def monitor_gpu():
            while not stop.is_set():
                try:
                    samples.append(gpu_sample())
                except Exception as error:
                    report.setdefault('monitor_errors', []).append(str(error))
                stop.wait(1)

        monitor = threading.Thread(target=monitor_gpu, daemon=True)
        monitor.start()

        class MemoryPeakHook(Hook):
            # MMEngine's logger resets peak memory stats. Read them before it.
            priority = 'HIGH'

            def before_train_epoch(self, runner):
                self.allocated = self.reserved = 0

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
                self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())

            def after_train_epoch(self, runner):
                report['peak_allocated_mib'] = self.allocated / 1024 ** 2
                report['peak_reserved_mib'] = self.reserved / 1024 ** 2

        class ProfileHook(Hook):
            priority = 'VERY_LOW'

            def before_train_epoch(self, runner):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.epoch_start = self.last_end = time.perf_counter()
                report['epoch_start_unix_time'] = time.time()
                report['actual_dataset_images'] = len(runner.train_dataloader.dataset)
                report['expected_iterations'] = len(runner.train_dataloader)
                report['parameters'] = sum(parameter.numel() for parameter in runner.model.parameters())

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                torch.cuda.synchronize()
                self.iter_start = time.perf_counter()
                self.data_time = self.iter_start - self.last_end
                self.batch_images = int(data_batch['inputs'].shape[0])

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                torch.cuda.synchronize()
                end = time.perf_counter()
                loss_value = outputs.get('loss') if isinstance(outputs, dict) else None
                loss = float(loss_value.detach().cpu() if isinstance(loss_value, torch.Tensor) else loss_value) if loss_value is not None else None
                if loss is not None and not math.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at iteration {batch_idx + 1}')
                rows.append({'iteration': batch_idx + 1, 'images': self.batch_images,
                    'step_seconds': end - self.iter_start, 'data_wait_seconds': self.data_time,
                    'iteration_seconds': end - self.last_end, 'loss': loss,
                    'allocated_mib': torch.cuda.memory_allocated() / 1024 ** 2,
                    'reserved_mib': torch.cuda.memory_reserved() / 1024 ** 2})
                self.last_end = end

            def after_train_epoch(self, runner):
                torch.cuda.synchronize()
                duration = time.perf_counter() - self.epoch_start
                report['epoch_end_unix_time'] = time.time()
                count = sum(row['images'] for row in rows)
                steady = rows[20:] if len(rows) > 20 else rows
                report.update({'completed_epochs': 1, 'iterations': len(rows), 'images_processed': count,
                    'epoch_seconds': duration, 'epoch_images_per_second': count / duration,
                    'steady_mean_iteration_seconds': statistics.mean(row['iteration_seconds'] for row in steady),
                    'steady_median_iteration_seconds': statistics.median(row['iteration_seconds'] for row in steady),
                    'steady_images_per_second': sum(row['images'] for row in steady) / sum(row['iteration_seconds'] for row in steady),
                    'steady_mean_data_wait_seconds': statistics.mean(row['data_wait_seconds'] for row in steady),
                    'warmup_iterations_excluded': len(rows) - len(steady),
                    'peak_allocated_mib': report['peak_allocated_mib'],
                    'peak_reserved_mib': report['peak_reserved_mib']})

        register_all_modules()
        config = Config.fromfile(ROOT / 'configs' / 'rtmdet_tiny_dior_1epoch.py')
        config.work_dir = str(WORK)
        config.launcher = 'none'
        from initialization import (assert_random_initialization,
                                    assert_model_random_initialization)
        assert_random_initialization(config)
        assert all(name == name.lower() for name in config.train_dataloader.dataset.metainfo.classes)
        runner = Runner.from_cfg(config)
        assert_model_random_initialization(runner.model)
        runner.register_hook(MemoryPeakHook())
        runner.register_hook(ProfileHook())
        runner.train()
        assert report.get('completed_epochs') == 1
        assert report['actual_dataset_images'] == report['images_processed'] == 5862
        report['status'] = 'completed'
        exit_code = 0
    except BaseException as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        report['iterations_completed'] = len(rows)
        traceback.print_exc()
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=12)
        if samples:
            report['peak_process_memory_mib_nvidia_smi'] = max(s['process_memory_mib'] for s in samples)
            report['peak_gpu_total_used_mib'] = max(s['used_mib'] for s in samples)
            report['mean_sampled_gpu_util_percent'] = statistics.mean(s['gpu_util_percent'] for s in samples)
        report['total_script_seconds'] = time.perf_counter() - started
        deleted = []
        for pattern in ('*.pth', '*.pt', '*.ckpt', 'last_checkpoint'):
            for path in WORK.rglob(pattern):
                deleted.append(str(path.relative_to(ROOT)))
                path.unlink()
        report['deleted_checkpoints'] = deleted
        report['checkpoints_remaining_in_work_dir'] = []
        if rows:
            with (WORK / 'iterations.csv').open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (WORK / 'gpu_samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        (WORK / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
        print('BENCHMARK_RESULT ' + json.dumps(report), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
