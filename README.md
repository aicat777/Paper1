# Paper1：RTMDet-tiny 遥感目标检测基线

使用 RTMDet-tiny 在 DIOR 水平框目标检测数据集上**完全从零训练**，作为后续修改骨干、Neck 或检测头的比较基线。无需任何预训练权重。

## 代码与环境

官方 MMYOLO 和 MMDetection 以固定版本 Git 子模块保存。克隆时包含子模块：

```bash
git clone --recurse-submodules https://github.com/aicat777/Paper1.git
cd Paper1
conda env create -f environment/conda.yml
conda activate rtmdet
python -m pip install -r environment/requirements-lock.txt
python -m pip install --no-deps -e ./mmdetection -e ./mmyolo
python -m pip check
```

已验证环境：Python 3.10、PyTorch 2.0.1+cu118、MMCV 2.0.1、MMEngine 0.10.7、MMDetection 3.3.0、MMYOLO 0.6.0。完整版本记录见 `environment/conda-audit.yml`，安装依赖见 `environment/requirements-lock.txt`。

## 数据与训练

DIOR 图像和原始标注保存在本地 `datasets/DIOR/`，不上传 Git。固定来源、哈希、划分和转换统计见 `data_metadata/DIOR/`。

```bash
python scripts/download_dior.py
python scripts/verify_dior.py
python scripts/convert_dior_to_coco.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python mmyolo/tools/train.py configs/rtmdet_tiny_dior_300e.py --work-dir work_dirs/rtmdet_tiny_dior_300e_gpu0
```

当前训练配置：GPU 0，训练 batch 20、验证 batch 16，640×640，300 轮，train 5,862 张 / val 5,863 张，随机种子 42。加载路径 `load_from=None`，骨干预训练初始化 `init_cfg=None`。

详细协议、官方训练命令、验证频率和检查点见 [TRAINING.md](TRAINING.md)。数据来源及转换规则见 [DOWNLOADS.md](DOWNLOADS.md)。

数据、论文 PDF、预训练权重、检查点、日志和运行结果均不纳入版本控制。
