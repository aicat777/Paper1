"""Exercise queue transitions without starting any real GPU experiment."""
from contextlib import redirect_stdout
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_experiment as scheduler


def completed_fixture(directory, log, finite=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'epoch_300.pth').write_bytes(b'checkpoint')
    log.write_text('Epoch(val) [300][733/733] pascal_voc/mAP: 0.75\n')
    scalars = directory / 'date/vis_data'
    scalars.mkdir(parents=True, exist_ok=True)
    row = dict(step=300, **{'pascal_voc/mAP': .75, 'coco/bbox_mAP': .56,
                          'coco/bbox_mAP_50': .77})
    if not finite:
        row['coco/bbox_mAP'] = float('nan')
    (scalars / 'scalars.json').write_text(json.dumps(row) + '\n')


class ExperimentQueue(unittest.TestCase):
    def test_process_identity_includes_start_time(self):
        stat = Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(') ')[2].split()
        self.assertTrue(scheduler.process_alive(os.getpid(), stat[19]))
        self.assertFalse(scheduler.process_alive(os.getpid(), str(int(stat[19]) + 1)))

    def test_completion_requires_checkpoint_finite_metrics_and_no_fatal_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, log = root / 'work', root / 'train.log'
            self.assertFalse(scheduler.completed(work, log))
            completed_fixture(work, log)
            self.assertTrue(scheduler.completed(work, log))
            (work / 'epoch_300.pth').write_bytes(b'')
            self.assertFalse(scheduler.completed(work, log))
            completed_fixture(work, log, finite=False)
            self.assertFalse(scheduler.completed(work, log))
            for message in ('CUDA out of memory', 'Traceback', 'loss: nan', 'loss_bbox: inf'):
                completed_fixture(work, log)
                log.write_text(message + '\n' + log.read_text())
                self.assertFalse(scheduler.completed(work, log), message)
            completed_fixture(work, log)
            log.write_text(log.read_text() + 'some_class_AP_small: nan\n')
            self.assertTrue(scheduler.completed(work, log))

    def fixture(self, root, case):
        directory = root / 'queue'
        directory.mkdir()
        dependency, dependency_log = root / 'fdw', root / 'fdw.log'
        status = root / 'fdw_status.json'
        completed_fixture(dependency, dependency_log, finite=case != 'incomplete')
        status.write_text(json.dumps(dict(
            state='failed' if case == 'parent_failed' else 'complete', queue_pid=7)))
        work, log = root / 'adapt', root / 'adapt.log'
        if case == 'existing_outputs':
            work.mkdir()
        summary = root / 'checks.json'
        summary.write_text('{}')
        source = root / 'model.py'
        source.write_text('source')
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(root=str(root), python='python', gpu_uuid='dummy', min_free_mib=45000,
                        dependency=dict(name='fdw', pid=7, start_ticks='1', status=str(status),
                                        work_dir=str(dependency), log=str(dependency_log)),
                        experiment=dict(config=str(root / 'config.py'), work_dir=str(work), log=str(log),
                                        capacity_report=str(root / 'capacity.json'),
                                        preflight_report=str(root / 'preflight.json'), validation_summary=str(summary)),
                        preflight_command=['python', 'probe.py', 'preflight'],
                        capacity_command=['python', 'probe.py', 'capacity'],
                        capacity_expected=dict(microbatch=9, accumulation=4),
                        source_sha256={'model.py': 'incorrect' if case == 'source_changed' else expected})
        path = directory / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_success_and_failure_transitions(self):
        cases = ('success', 'parent_failed', 'incomplete', 'existing_outputs', 'source_changed',
                 'preflight_failure', 'capacity_exit_failure', 'capacity_report_failure',
                 'capacity_wrong_batch', 'training_failure')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                path, manifest = self.fixture(Path(temporary), case)
                children = []
                class Child:
                    def __init__(self, command, **kwargs):
                        self.command = command
                        children.append(command)
                        self.pid = 100 + len(children)
                        if command[-1] == 'preflight':
                            Path(manifest['experiment']['preflight_report']).write_text('{"status": "passed"}')
                        elif command[-1] == 'capacity':
                            report = dict(status='failed' if case == 'capacity_report_failure' else 'passed',
                                          microbatch=2 if case == 'capacity_wrong_batch' else 9, accumulation=4)
                            Path(manifest['experiment']['capacity_report']).write_text(json.dumps(report))
                        elif '--work-dir' in command and case != 'training_failure':
                            completed_fixture(Path(manifest['experiment']['work_dir']),
                                              Path(manifest['experiment']['log']))
                    def wait(self):
                        if (case == 'preflight_failure' and self.command[-1] == '--help') or (
                                case == 'capacity_exit_failure' and self.command[-1] == 'capacity') or (
                                case == 'training_failure' and '--work-dir' in self.command):
                            return 1
                        return 0
                with patch.object(scheduler, 'process_alive', return_value=False), \
                        patch.object(scheduler.subprocess, 'Popen', Child), \
                        patch.object(scheduler.subprocess, 'check_output', return_value='49140\n'), \
                        redirect_stdout(io.StringIO()):
                    if case == 'success':
                        scheduler.run_manifest(path)
                    else:
                        with self.assertRaises(RuntimeError):
                            scheduler.run_manifest(path)
                status = json.loads((path.parent / 'status.json').read_text())
                self.assertEqual(status['state'], 'complete' if case == 'success' else 'failed')
                expected = 0 if case in ('existing_outputs', 'source_changed') else (
                    1 if case == 'preflight_failure' else 2 if case in ('parent_failed', 'incomplete') else (
                        3 if case.startswith('capacity_') else 4))
                self.assertEqual(len(children), expected)
                if case == 'success':
                    self.assertEqual(json.loads(Path(manifest['experiment']['validation_summary']).read_text())
                                     ['formal_capacity']['microbatch'], 9)

    def test_duplicate_scheduler_does_not_replace_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.fixture(Path(temporary), 'success')
            status = path.parent / 'status.json'
            status.write_text('{"state": "waiting_fdw"}')
            with (path.parent / 'queue.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    scheduler.run_manifest(path)
            self.assertEqual(status.read_text(), '{"state": "waiting_fdw"}')


    def test_resume_preserves_log_and_passes_explicit_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            path, manifest = self.fixture(Path(temporary), 'success')
            experiment = manifest['experiment']
            work, log = Path(experiment['work_dir']), Path(experiment['log'])
            work.mkdir()
            checkpoint = work / 'epoch_98.pth'
            saved = dict(meta=dict(epoch=98, iter=63896), state_dict={}, optimizer={},
                         param_schedulers=[], ema_state_dict={})
            torch.save(saved, checkpoint)
            log.write_text('Original training record\n')
            experiment['resume_from'] = str(checkpoint)
            path.write_text(json.dumps(manifest))
            children = []
            class Child:
                def __init__(self, command, **kwargs):
                    children.append(command)
                    self.pid = 100 + len(children)
                    if command[-1] == 'preflight':
                        Path(experiment['preflight_report']).write_text('{"status":"passed"}')
                    elif command[-1] == 'capacity':
                        Path(experiment['capacity_report']).write_text(
                            '{"status":"passed","microbatch":9,"accumulation":4}')
                    elif '--work-dir' in command:
                        previous = log.read_text()
                        completed_fixture(work, log)
                        log.write_text(previous + log.read_text())
                def wait(self):
                    return 0
            with patch.object(scheduler, 'process_alive', return_value=False), \
                    patch.object(scheduler.subprocess, 'Popen', Child), \
                    patch.object(scheduler.subprocess, 'check_output', return_value='49140\n'), \
                    redirect_stdout(io.StringIO()):
                scheduler.run_manifest(path)
            self.assertEqual(children[-1][-2:], ['--resume', str(checkpoint)])
            self.assertTrue(log.read_text().startswith('Original training record\n'))
            self.assertEqual(json.loads((path.parent / 'status.json').read_text())['state'], 'complete')
            self.assertEqual(torch.load(checkpoint)['meta']['epoch'], 98)

    def test_resume_rejects_foreign_or_incomplete_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work, log = root / 'work', root / 'train.log'
            work.mkdir()
            log.write_text('previous training\n')
            checkpoint = root / 'foreign.pth'
            torch.save({}, checkpoint)
            experiment = dict(work_dir=str(work), log=str(log), resume_from=str(checkpoint))
            with self.assertRaisesRegex(RuntimeError, 'belong'):
                scheduler.validate_resume_checkpoint(experiment)
            checkpoint = work / 'epoch_98.pth'
            experiment['resume_from'] = str(checkpoint)
            torch.save(dict(meta=dict(epoch=98, iter=63896), state_dict={}), checkpoint)
            with self.assertRaisesRegex(RuntimeError, 'optimizer'):
                scheduler.validate_resume_checkpoint(experiment)


if __name__ == '__main__':
    unittest.main()
