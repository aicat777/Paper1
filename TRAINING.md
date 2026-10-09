# RTMDet-tiny / DIOR：从零训练

本项目基线完全从零训练。`load_from=None`、`resume=False`、`model.backbone.init_cfg=None`，不使用 COCO、ImageNet 或其他预训练权重。网络按各模块默认初始化规则初始化，全部参数参与训练。启动程序检查合并后的模型配置，发现预训练初始化会直接报错。

## 实验设置

| 项目 | 当前设置 |
| --- | --- |
| 模型 | 官方 MMYOLO RTMDet-tiny，DIOR 20 类 |
| 数据 | 原始 train 5,862 张、val 5,863 张，划分独立 |
| 初始化 | 随机初始化，seed=42；不使用预训练 |
| 设备与环境 | 物理 GPU 1；Conda `rtmdet` |
| 输入 / batch | 640×640 / 16 |
| 数据加载进程 | train 8，val 4 |
| 精度 | 训练 AMP，验证 FP32；TF32 开启 |
| 优化器 | AdamW，lr=0.00025，weight decay=0.05 |
| 学习率 | 前 1000 iteration warmup；第 150–300 轮 cosine，最低 lr=0.0000125 |
| 总轮数 | 300；前 280 轮强增强，最后 20 轮弱增强；EMA |
| 验证 | 第 1 轮；第 10–270 轮每 10 轮；第 280–300 轮每轮，共 49 次 |
| 检查点 | 每轮保存，仅保留最近两个；额外保留最佳 mAP 和 AP50 |

val 使用 COCO bbox 评估实现，计算 mAP（IoU 0.50:0.95）、AP50、AP75、各尺度和每类别 AP。这里的 COCO 指指标格式，不涉及 COCO 训练数据或预训练权重。评估 maxDets=100，模型最多输出 300 框/图。

测试集不参与训练或模型选择，本次只运行 train 和 val。后续改进模型的对比应保持初始化、划分、评价方式和训练协议一致。

启动程序解析物理 GPU 1 的 UUID，仅使用这张卡；PyTorch 分配器上限 10 GiB，并至少为已有 GPU 进程预留 2 GiB。

## 启动与状态

```bash
conda activate rtmdet
cd /home/pwt/Paper1
mkdir -p logs
nohup python -u scripts/train_dior_full.py > logs/rtmdet_tiny_dior_scratch_300e_gpu1.log 2>&1 &
```

工作目录：`work_dirs/rtmdet_tiny_dior_scratch_300e_gpu1/`。

- `status.json`：初始化方式、训练 PID、Git commit、速度、显存、ETA 和验证结果。
- `validation_results.json`：所有验证轮次及指标。
- `RESULT.md`：可读进度；由 `scripts/watch_training_result.py` 每 20 秒更新。
- `epoch_*.pth`、`last_checkpoint`：最近检查点和续训路径。
- `best_coco_bbox_mAP_*.pth`、`best_coco_bbox_mAP_50_*.pth`：验证最优模型。

查看日志：

```bash
tail -f logs/rtmdet_tiny_dior_scratch_300e_gpu1.log
```

可单独启动进度记录程序：

```bash
nohup python -u scripts/watch_training_result.py > logs/training_result_report.log 2>&1 &
```

仅当训练进程已退出时，使用本次随机初始化实验的检查点续训：

```bash
python -u scripts/train_dior_full.py --resume
```

续训是继续本次从零训练的进度。程序拒绝其他初始化方式的历史实验。工作目录使用进程锁，避免重复训练。

## 时间估计

300 轮训练及 49 次完整 val 初步预计 13–15 小时。启动后根据本次实验最近 10 轮训练及完整 val 的实际时长，动态更新 `status.json` 的 `estimated_finish_at`，时区为 Asia/Shanghai。估计受共享 GPU 负载、目标数量和 CPU 指标计算开销影响。

之前使用预训练权重的训练、测速记录、检查点及下载权重均已删除；本次从第 1 轮重新开始。新训练产生的检查点保留用于验证和续训，不加入 Git。
