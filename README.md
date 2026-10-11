# Paper1：RTMDet-tiny / DIOR

使用 MMYOLO 官方 RTMDet-tiny，在 DIOR 水平框数据上从零训练。输入 800×800，使用官方 train 5862 张训练、val 5863 张评估；test 11738 张不参与当前实验。

主指标为 VOC2007 mAP@0.5（11 点插值），同时报告 COCO AP50:95、AP50、AP75。COCO JSON 仅是标注格式，不使用 COCO 数据或预训练权重。

## 环境

```bash
git clone --recurse-submodules https://github.com/aicat777/Paper1.git
cd Paper1
conda env create -f environment/conda.yml
conda activate rtmdet
python -m pip install -r environment/requirements-lock.txt
python -m pip install --no-deps -e ./mmdetection -e ./mmyolo
```

当前环境：Python 3.10、PyTorch 2.0.1+cu118、MMCV 2.0.1、MMEngine 0.10.7、MMDetection 3.3.0、MMYOLO 0.6.0。官方代码由 Git 子模块固定版本；依赖版本在 `environment/`。

## 数据

数据位于 `datasets/DIOR/`，沿用 `ImageSets/Main/` 官方列表。原始图像、XML 和划分保留；来源与校验信息在 `data_metadata/DIOR/`。

```bash
python scripts/download_dior.py
python scripts/verify_dior.py
python scripts/convert_dior_to_coco.py
```

[DIOR 作者主页](https://gcheng-nwpu.github.io/#Datasets)；[下载镜像](https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/tree/f58d220a2b74b610badebaefa122d1b61628bf73/DIOR)。标注转换沿用 MMDetection VOC 坐标规则，去除零面积框，原始 XML 不修改。

## 训练与文献

训练设置、官方命令及检查点见 [TRAINING.md](TRAINING.md)。配置：`configs/rtmdet_tiny_dior_300e.py`。

## 独立候选方法

每套包含两个模块，代码和配置分别存放；共同继承上述基线，不修改 MMYOLO 官方实现。全部从零训练。这些是借鉴机制后设计的候选组合，不代表原论文的完整复现或已验证的精度提升。

| 方法 | 两个模块 | 代码 | 配置 |
|---|---|---|---|
| A：状态引导 | 压缩隐状态上下文、状态引导局部上采样 | [state_guided](custom_models/state_guided/) | [完整组合](configs/state_guided/full.py)、[仅上下文](configs/state_guided/context_only.py)、[仅上采样](configs/state_guided/upsample_only.py)、[关闭直接引导](configs/state_guided/no_guidance.py) |
| B：幅值与细节 | 幅值感知线性上下文、门控浅层细节补充 | [magnitude_detail](custom_models/magnitude_detail/) | [完整组合](configs/magnitude_detail/full.py)、[仅上下文](configs/magnitude_detail/context_only.py)、[仅细节](configs/magnitude_detail/detail_only.py)、[普通线性上下文对照](configs/magnitude_detail/linear_context_only.py) |
| C：频率引导 | 空间频带调制、频率引导局部上采样 | [frequency_guided](custom_models/frequency_guided/) | [完整组合](configs/frequency_guided/full.py)、[仅频带调制](configs/frequency_guided/frequency_only.py)、[仅上采样](configs/frequency_guided/upsample_only.py)、[关闭直接引导](configs/frequency_guided/no_guidance.py) |

模块原理、借鉴来源、运行命令与验证范围见 [TRAINING.md](TRAINING.md#候选方法)。数值与配置检查：`python -m unittest discover -s tests -v`。

DIOR 文献对比采用 [SED-DETR（IEEE TGRS，2025）](https://doi.org/10.1109/TGRS.2025.3571033)表 III，转录值在 `dior_sed_detr_table3.csv`。引用行标为文献结果；该文 DETR 模型使用预训练，且未说明 VOC 插值版本。本文基线、改进模型和消融统一从零训练。

数据、论文 PDF、日志、检查点及运行结果不纳入版本控制。
