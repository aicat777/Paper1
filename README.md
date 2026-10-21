# Paper1：RTMDet-tiny / DIOR

使用 MMYOLO 官方 RTMDet-tiny，在 DIOR 水平框数据上从零训练。输入640×640，官方 train 5862张、val 5863张；test 11738张不参与当前实验。主指标为 VOC2007 mAP@0.5（11点插值），同时报告 COCO AP50:95、AP50、AP75 和各尺度 AP。COCO JSON 是标注格式；不使用外部预训练权重。

## 环境与数据

```bash
git clone --recurse-submodules https://github.com/aicat777/Paper1.git
cd Paper1
conda env create -f environment/conda.yml
conda activate rtmdet
python -m pip install -r environment/requirements-lock.txt
python -m pip install --no-deps -e ./mmdetection -e ./mmyolo
```

Python3.10、PyTorch2.0.1+cu118、MMCV2.0.1、MMEngine0.10.7、MMDetection3.3.0、MMYOLO0.6.0；官方代码固定为 Git 子模块版本，依赖锁定在 `environment/`。

数据位于 `datasets/DIOR/`，沿用 `ImageSets/Main/` 官方列表。图像、XML 与划分保留，下载与校验信息见 `data_metadata/DIOR/`。

```bash
python scripts/download_dior.py
python scripts/verify_dior.py
python scripts/convert_dior_to_coco.py
```

[DIOR 作者主页](https://gcheng-nwpu.github.io/#Datasets)；[下载镜像](https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/tree/f58d220a2b74b610badebaefa122d1b61628bf73/DIOR)。转换沿用 MMDetection VOC 坐标规则，去除零面积框，不修改原始 XML。

## 模型与训练

**SCDown与实例类别排序监督版均已完成300轮**，640输入、真实batch36、不累积、从零训练、每10轮验证。SCDown的VOC mAP75.87%、COCO AP56.5%，排序监督版75.56%/55.7%；原实例语义监督76.31%/56.9%，基线75.65%/56.4%。LSConv三处骨干版75.57%/55.9%、检测头版75.27%/54.7%、neck输出层版75.12%/55.6%，neck内部块版已取消。原理、检查和接续规则见 [TRAINING.md](TRAINING.md)。

中心点辅助与实验1 SCDown＋G2均已完成；G2的VOC mAP75.74%、COCO AP56.4%。GPU1当前训练 **实验2 SCDown扩展P3 → 仅匹配GT的ROI辅助 → 随机等数量GT的ROI辅助 → 实验3 SCDown扩展P2/P3 → G4 → DSConv**。各组均为640、真实batch36、不累积、从零300轮、每10轮验证；每组启动前核验正式容量，前序失败则停止接续。状态在各方法 `work_dirs/<方法>_queue/status.json`。G4与标准DSConv已按此顺序接入队列，使用独立代码、冻结配置和检查报告。

| 配置目录 | 功能 |
|---|---|
| `configs/rtmdet_tiny_dior_300e.py` | 官方三尺度基线 |
| `configs/scdown/` | 原始基线上，仅替换骨干两处深层及PAN最后一次降采样；先1×1混合，再DW3×3降采样 |
| `configs/scdown_g2/` | 实验1：三处SCDown＋骨干P3–P5及neck内部七个CSP的3×3卷积分两组 |
| `configs/scdown_g4/` | 三处SCDown＋相同七处3×3改为groups=4，作为分组数对照 |
| `configs/scdown_dsconv/` | 三处SCDown＋相同七处3×3改为DW3×3/BN/SiLU→PW1×1/BN/SiLU |
| `configs/scdown_p3/` | 实验2：三处SCDown＋骨干P3入口，共四处；不修改CSP卷积分组 |
| `configs/scdown_p2p3/` | 实验3：三处SCDown＋骨干P2、P3入口，共五处；仍使用三尺度检测头 |
| `configs/efs/` | 独立 EFS 四尺度实验，使用官方组件 |
| `configs/efs_adapt/` | EFS保留96通道P3侧向特征，P2检测塔独立，P3–P5沿用共享卷积 |
| `configs/lsconv/` | 完整 LSConv 替换骨干三处3×3卷积，保持三尺度检测 |
| `configs/lsconv_full/` | LSConv扩展至骨干、颈部与检测塔，保留显式1×1通道混合和尺度独立归一化 |
| `configs/lsconv_20/` | 骨干4处、颈部内部4处和检测塔12处LSConv；保留原颈部输出，保护LSConv作者初始化 |
| `configs/lsconv_csp/` | 旧三处接法扩到骨干4处和颈部内部4处，只替换conv1，保留原后续卷积 |
| `configs/lsblock_csp/` | 上述8个位置替换整个内部块，使用LSConv＋残差FFN，保留原颈部输出与检测塔 |
| `configs/lsconv_neck/` | 原三处LSConv骨干＋neck内部四个完整LSBlock；保留颈部输出卷积和原检测头 |
| `configs/lsconv_head/` | 原三处LSConv骨干＋检测塔第二层完整LSBlock；保留原neck和检测塔第一层 |
| `configs/lsconv_neck_out/` | 原三处LSConv骨干＋三个neck输出：先在96/192/384通道做LSConv，再以1×1映射到96通道 |
| `configs/fdw/` | FDW频域权重与输入条件核混合，独立替换骨干三处3×3卷积 |
| `configs/fdw_ksm/` | 原三处FDW补回KSM局部分支，按输入通道、输出通道和核位置细化权重调制 |
| `configs/freq_fusion/` | 完整FreqFusion：两处自顶向下融合使用低通、高通、相似性偏移和增强初始融合 |
| `configs/freq_fusion_residual/` | CSP融合后注入高频残差：主路径融合浅层原特征，细节经独立1×1映射后相加 |
| `configs/instance_semantic/` | 训练时从分类塔按GT取样，每个实例额外监督一次类别；推理沿用基线 |
| `configs/instance_semantic_center/` | 所有GT、原辅助分类器，仅把区域平均改为GT中心的单次双线性取样 |
| `configs/instance_semantic_matched/` | 原ROI辅助任务仅选择已匹配GT，分母仍为全部有效GT数 |
| `configs/instance_semantic_random/` | 从全部有效GT随机取当前匹配GT数量，使用独立、可恢复随机状态 |
| `configs/instance_semantic_rank/` | 保留整图GT语义监督，按真实匹配GT均衡监督检测头类别logits；不增加参数和推理计算 |
| `configs/instance_semantic_multiscale/` | 增强前每图裁剪一个GT，六种尺度随机取一种，经共享骨干、neck与分类塔监督类别 |
| `configs/tiny_assignment/` | 细小目标局部补匹配与回归权重下限，保留真实IoU分类质量标签 |
| `configs/factorized_head/` | 分类/回归独立的PW–DW–PW检测塔，跨尺度共享卷积、BN按尺度独立 |
| `configs/state_guided/` | 压缩状态上下文、局部上采样，各模块与组合分别配置 |
| `configs/magnitude_detail/` | 幅值线性上下文、浅层细节，各模块与组合分别配置 |
| `configs/frequency_guided/` | 频带选择、局部上采样，各模块与组合分别配置 |

各自定义方法的代码分别位于 `custom_models/` 下同名目录；新neck/检测头两组仅共用已验证且不修改的 `custom_models/lsconv` 骨干，新增模块各自独立。原EFS使用官方组件。EFS适配的两项结构分别位于 `custom_models/efs_adapt/neck.py`、`head.py`；FDW算子与骨干接入分别位于 `custom_models/fdw/operator.py`、`backbone.py`。

`scdown`使用独立代码与配置，不叠加LSConv或实例监督。总参数3,854,964、卷积7.623901 GMAC，较基线减少20.98%和4.55%。9项测试、24组作者算子GPU前后向对照、官方入口真实数据训练/验证及GPU1正式batch36压力检查通过；完整训练后VOC比基线高0.22个百分点、COCO AP高0.1个百分点。入口为 `scripts/scdown_probe.py`，汇总在 `work_dirs/scdown_validation/checks.json`。

FDW＋局部KSM已独立实现并通过作者算子对照、GPU训练/验证和batch12压力检查，配置为 [configs/fdw_ksm/rtmdet_tiny_dior_640.py](configs/fdw_ksm/rtmdet_tiny_dior_640.py)，尚未加入完整训练队列。正式配置沿用FDW的batch36，启动前仍需核验该batch的容量。

四项候选各用同名独立代码与配置，不相互组合。FreqFusion按作者检测配置迁移完整算子，外层保留基线CSP/PAN融合；其余三项是本项目设计。均使用640、从零训练、官方train/val、真实batch36且不累积梯度，与640基线优化协议一致；已通过结构/数值核对和GPU真实数据短测。实例语义监督仅在辅助损失中排除增强后无面积的框，所有正面积小框和原检测损失保留。正式容量检查失败就停止，不自动修改batch。已完成LSConv扩展使用12×3，其BN批统计不同。说明见 [TRAINING.md](TRAINING.md)，复核入口为 `scripts/candidate_probe.py` 和 `scripts/candidate_capacity.py`。

干净目标多尺度监督使用独立的 `custom_models/instance_semantic_multiscale/` 和同名配置，替代旧版的整图GT辅助取样。裁剪分支与主图共享参数，BN统计隔离；两次反传后只更新一次参数，推理路径与旧版完全一致。实现和检查入口为 `scripts/instance_semantic_multiscale_probe.py`；已在后台排队，状态见 `work_dirs/instance_semantic_multiscale_queue/status.json`。

`freq_fusion_residual`为独立结构变体，不修改旧FreqFusion：完整滤波与偏移算子保留，高频残差在FP32中直接提取，分别经192→96、96→96的无偏置1×1卷积，在两处完整CSP融合块后注入。总参数5.015M、Conv/Linear 8.286 GMAC。6项测试、6组作者算子GPU FP32/AMP对照及官方入口短测通过，正式batch36容量检查通过；已完成300轮，VOC mAP76.29%、COCO AP56.9%。状态在 `work_dirs/freq_fusion_residual_queue/status.json`，复核入口为 `scripts/freq_fusion_residual_probe.py`。

`lsconv_csp`与`lsblock_csp`为两项新增独立实现，均沿用640、从零训练、真实batch36和基线优化协议，当前未排入完整训练。8项测试、32组作者代码FP32/AMP对照、官方入口真实数据短跑及batch12压力检查通过；普通扩展缩回三处时与旧版初始状态及输出完全一致。正式batch36容量尚未验证。复核入口为 `scripts/lsconv_csp_probe.py`。

`lsconv_neck`/`lsconv_head`两组均从零训练，复用三处版的结构和已有对照结果，不加载其权重。参数/MAC分别为4.251M/7.395G、4.461M/7.149G，MAC含SKA、不含归一化与激活。8项集成测试、16组作者模块GPU前向/梯度对照、各16次真实更新与两次验证通过；neck版已取消，检测头版已完成。复核入口为 `scripts/lsconv_neck_head_probe.py`。

`lsconv_neck_out`仅替换三个最终输出3×3，采用LSConv→1×1/BN/SiLU，FPN/PAN内部和原检测头保留。总参数4,277,142、7.108 GMAC（含SKA）；5项测试、12组作者算子GPU FP32/AMP前向/梯度对照、官方入口16次实际更新及两次验证通过。已完成300轮：VOC mAP75.12%、COCO AP55.6%；相对三处骨干版少5.31%参数、7.27% MAC，精度分别低0.46和0.3个百分点。结果与曲线见 `work_dirs/diagnostics/lsconv_neck_out_analysis/`，复核入口为 `scripts/lsconv_neck_out_probe.py`。

`lsconv_20`独立配置与代码已完成300轮：3.856M参数、5.180 GMAC，较基线减少20.97%和35.15%。9项集成检查、20组作者算子前向/梯度对照和GPU真实数据训练/验证通过；GPU1正式batch36、每图500框压力检查通过，保持batch36、不累积。复核入口为 `scripts/lsconv_20_probe.py`。

检查：`python -m unittest discover -s tests -v`。DIOR 文献对比表见 `dior_sed_detr_table3.csv`，来源为 [SED-DETR，IEEE TGRS 2025](https://doi.org/10.1109/TGRS.2025.3571033)表III；文献行与本项目实测结果区分标注。

`logs/`与`work_dirs/`保留完整实验记录并提交Git，包括原始日志、逐轮指标、实际运行配置、队列状态和检查产物，不只保存汇总。模型检查点（`.pth`、`.pt`、`.ckpt`）、研究下载资料和PDF解析依赖不上传；数据集与缓存继续忽略。`work_dirs/research/`中的实验检查日志仍保留。运行中的记录以每次提交时的快照为准。
