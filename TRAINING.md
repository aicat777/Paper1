# RTMDet-tiny / DIOR：从零训练

使用 MMYOLO 官方入口 `mmyolo/tools/train.py`，配置为 `configs/rtmdet_tiny_dior_300e.py`，不添加自定义训练程序、Hook、预训练拦截或监控程序。

## 配置

- 不使用预训练：`load_from=None`、`resume=False`、`model.backbone.init_cfg=None`，保留框架默认初始化。
- DIOR 20 类，原始 train 5,862 张、val 5,863 张；test 11,738 张单独配置，训练过程不运行 test。
- GPU 0，Conda `rtmdet`，输入 640×640，训练 batch 20、验证 batch 16。
- 训练 AMP、验证 FP32；AdamW，基础学习率 0.0003125，weight decay 0.05。
- 1000 iteration warmup；第 150–300 轮 cosine；随机种子 42。
- 共 300 轮；前 280 轮强增强，最后 20 轮弱增强；使用官方 EMA 和 PipelineSwitchHook。
- 每 10 轮验证一次：第 10、20、…、300 轮，共 30 次；最后 20 轮也保持这一间隔。
- 每轮保存检查点，仅保留最近两个，另保留 val mAP 和 AP50 最佳模型。
- COCO JSON 和 COCO bbox 指标仅指格式与评价方法，不涉及 COCO 数据或预训练。

## 官方训练命令

```bash
conda activate rtmdet
cd /home/pwt/Paper1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python mmyolo/tools/train.py configs/rtmdet_tiny_dior_300e.py --work-dir work_dirs/rtmdet_tiny_dior_300e_gpu0
```

需要后台运行时，可直接使用 shell 的 `nohup`，不需要额外启动脚本。

续训使用官方 `--resume` 参数和对应工作目录；日志、指标和检查点使用框架原生输出。

本次从第 1 轮启动，使用固定每 10 轮验证的调度。配置文件修改在下次启动或续训时读取。

旧训练的检查点、日志和结果文件已清理，DIOR 图像、标注和原始划分保留。本次使用官方入口后台从第 1 轮训练，不使用 `--resume`。

GPU 1 的上一轮训练在动态标签分配阶段 OOM，进程已退出。其检查点、日志和结果均已清理。GPU 0 启动前空闲约 21.4 GiB，训练 batch 调整为 20，以保留显存余量；占用会随每批标注数量变化。

当前日志：`logs/rtmdet_tiny_dior_300e_gpu0.log`。框架原生日志、指标与检查点位于 `work_dirs/rtmdet_tiny_dior_300e_gpu0/`。

```bash
tail -f /home/pwt/Paper1/logs/rtmdet_tiny_dior_300e_gpu0.log
```
