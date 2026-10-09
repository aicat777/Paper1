#!/usr/bin/env python3
"""Write a readable progress/result report for the existing training process."""
import fcntl
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work_dirs' / 'rtmdet_tiny_dior_scratch_300e_gpu1'


def render(status):
    validation = status.get('latest_validation', {})
    metrics = validation.get('metrics', {})
    state = status['state']
    names = {'training': '训练中', 'validating': '验证中', 'between_epochs': '验证完成',
             'completed': '完整训练已完成', 'failed': '训练已停止', 'initializing': '初始化'}
    lines = ['# RTMDet-tiny / DIOR 训练状态', '',
             f'更新时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}', '',
             f'状态：**{names.get(state, state)}**。已完成 {status["completed_epochs"]}/300 个 epoch。', '',
             'GPU 1，Conda `rtmdet`，batch 16，640×640，训练 AMP、验证 FP32。', '',
             '全模型随机初始化，未使用 COCO / ImageNet 预训练权重，seed=42。', '',
             'train 5,862 张，val 5,863 张，原始划分。每轮保存检查点，仅保留最近两个；额外保留最佳 mAP 和 AP50 模型。', '']
    if status.get('estimated_finish_at'):
        lines.extend([f'预计完成：**{status["estimated_finish_at"]}**（Asia/Shanghai）。', '',
            f'最近平均训练 epoch：{status["mean_recent_epoch_seconds"]:.1f} 秒；平均完整 val：{status["mean_validation_seconds"]:.1f} 秒。', ''])
    if metrics:
        lines.extend([f'最近验证：epoch {validation["epoch"]}，耗时 {validation["seconds"]:.1f} 秒。', '',
                      '| 指标 | 数值 |', '| --- | ---: |'])
        for key in ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
                    'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']:
            if key in metrics:
                lines.append(f'| {key} | {metrics[key]:.3f} |')
        lines.append('')
    for key, label in [('best_mAP', '最佳 mAP'), ('best_AP50', '最佳 AP50')]:
        if key in status:
            best = status[key]
            lines.append(f'{label}：{best["score"]:.3f}，epoch {best["epoch"]}。')
    if status.get('error'):
        lines.extend(['', f'停止原因：`{status["error_type"]}: {status["error"]}`。'])
    lines.extend(['', '详细状态：[status.json](status.json)。验证历史：[validation_results.json](validation_results.json)。', ''])
    return '\n'.join(lines)


def main():
    lock = (WORK / '.report.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            status = json.loads((WORK / 'status.json').read_text())
            report = WORK / 'RESULT.md'
            temporary = report.with_suffix('.md.tmp')
            temporary.write_text(render(status))
            temporary.replace(report)
            if status['state'] in ('completed', 'failed'):
                print('Training finished:',status['state'],flush=True)
                return
            if not Path(f'/proc/{status["pid"]}').exists():
                with report.open('a') as output:
                    output.write('\n训练进程已退出，未写入最终状态；请检查完整训练日志。\n')
                return
        except (OSError, json.JSONDecodeError) as error:
            print('Status read failed:',error,flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
