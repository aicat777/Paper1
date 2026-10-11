# RTMDet-tiny / DIOR 训练设置

使用 Conda `rtmdet` 和 MMYOLO 官方 `mmyolo/tools/train.py`。配置：`configs/rtmdet_tiny_dior_300e.py`。

| 项目 | 设置 |
|---|---|
| 输入 | 640×640；训练最终裁剪/填充、评估保持比例缩放并填充 |
| 数据 | 官方 train 5862 → val 5863；test 11738 不参与本次训练评估 |
| 初始化 | 随机初始化；`init_cfg=None`、`load_from=None`、`resume=False` |
| 主指标 | `pascal_voc/mAP`：IoU=0.5，VOC2007 11 点插值 |
| 其他指标 | COCO AP50:95、AP50、AP75、各尺度 AP |
| GPU / batch | GPU0；训练 12，验证 8 |
| 精度 | 训练 AMP，验证 FP32 |
| 训练预算 | 300 epochs；每 10 轮验证，包含第 300 轮 |
| 优化器 | AdamW；lr=0.0001875，weight decay=0.05 |
| 调度 | warmup 1000 iterations；第 150–300 轮 cosine，最低 lr 为初始值的 5% |
| 增强 | 前 280 轮官方 tiny Mosaic/MixUp；最后 20 轮弱增强；EMA |
| 随机种子 | 42 |
| 检查点 | 每轮保存，保留最近两个；另保存 VOC mAP 与 COCO AP50:95 各自最佳模型 |

标注在评估图像缩放后加载，GT 保持原图坐标，匹配模型还原到原图的预测框。640 可被 32 整除，关闭额外的 batch shape padding。

## 启动

```bash
conda activate rtmdet
cd /home/pwt/Paper1
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
nohup python -u mmyolo/tools/train.py configs/rtmdet_tiny_dior_300e.py \
  --work-dir work_dirs/rtmdet_tiny_dior_640_300e_gpu0 \
  > logs/rtmdet_tiny_dior_640_300e_gpu0.log 2>&1 < /dev/null &
```

## 查看进度

```bash
tail -f /home/pwt/Paper1/logs/rtmdet_tiny_dior_640_300e_gpu0.log
nvidia-smi
```

检查点、框架日志和标量输出位于 `work_dirs/rtmdet_tiny_dior_640_300e_gpu0/`。指标值为 0–1，论文百分数乘以 100。当前实验只评估 val；推理设置为 score=0.001、NMS IoU=0.65、每图最多 300 个检测框。

## 候选方法

三套方法分别位于 `custom_models/{state_guided,magnitude_detail,frequency_guided}`。每个模块单独一个文件，`neck.py` 仅负责组装本方法并注册独立的颈部类型，方法间不互相导入。配置目录与方法目录同名，各含两个单模块配置、一个 `full.py` 和一个机制对照，共12个配置。输出目录均带 `_640` 后缀。所有正式配置保持基线的数据、初始化、优化器、batch、增强和评估设置，工作目录彼此独立。

### A：状态引导（StateGuidedPAFPN）

- `context.py`：将 C4（192通道、40×40）投影到48维，按空间归一化权重聚合成16个隐状态。在状态上执行门控通道混合，再读回空间并与局部深度卷积融合，残差更新 C4。借鉴 [EfficientViM，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_EfficientViM_Efficient_Vision_Mamba_with_Hidden_State_Mixer_based_State_CVPR_2025_paper.pdf)的 HSM 思路；新增归一化聚合/读回和对外输出状态，未实现原版完整 HSM-SSD。
- `upsample.py`：在最后一次40×40→80×80上采样时，用 C3、粗层特征和上述状态生成9邻域权重。对粗层局部邻居加权重组，与最近邻结果残差混合。借鉴 [UPLiFT，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Walmer_UPLiFT_Efficient_Pixel-Dense_Feature_Upsampling_with_Local_Attenders_CVPR_2026_paper.pdf)的 Local Attender；使用现有特征及跨尺度状态提供引导，不使用原论文的预训练编码器或重建训练。

两模块通过隐状态显式连接，研究压缩的上下文能否指导细尺度信息传递。仅上采样配置不创建状态分支；仅上下文配置使用最近邻上采样。`no_guidance.py` 保留两个模块，关闭状态到上采样的直接连接，用于检验显式引导分支的价值。空间 softmax 的状态分配层不使用会被归一化抵消的偏置。

### B：幅值与细节（MagnitudeDetailPAFPN）

- `context.py`：将 C4 投影到48维、4个头，使用配套的幅值缩放与均值补偿计算线性注意力；保留3×3深度卷积局部分支，残差更新 C4。借鉴 [MALA，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.pdf)，在 FP32 中计算关键归约，未移植原模型的旋转位置编码或整套骨干。
- `detail.py`：额外取出骨干 C2（48通道、160×160），投影到24通道，再通过 space-to-depth 将相邻2×2像素重排为96通道的80×80特征。随后用深度卷积、通道投影及 C3 条件门控筛选并注入细节。借鉴 [DM-EFS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.pdf)的浅层扩展特征思路；未实现其尺寸码本和动态路由，也未增加 P2 检测头。

两个模块分别修改 C4 和 C3，通过原有 PAN 融合。是否存在直接引导不作为筛选条件，应分别检验精度收益和开销。DM-EFS 仅提供浅层特征动机；当前细节分支按独立设计评价。`linear_context_only.py` 与 `context_only.py` 的参数、投影、局部分支完全一致，仅改成普通归一化线性注意力，用于检验幅值机制。各配置均返回 C2–C5，仅上下文配置不消费 C2；最终仍为 stride=8/16/32 三层检测。

### C：频率引导（FrequencyGuidedPAFPN）

- `frequency.py`：对 C3 做 FP32 FFT，以归一化频率1/16、1/8、1/4分成四个互补频带；预测各位置、各通道组的频带权重并残差更新特征，同时输出各频带通道平均绝对值的归一化统计（非平方能量）。借鉴 [FDConv，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Frequency_Dynamic_Convolution_for_Dense_Image_Prediction_CVPR_2025_paper.pdf)的 FBM；仅实现特征域适配，不包含 FDW/KSM。权重初始为1，各频带之和还原输入。
- `upsample.py`：用 C3、粗层特征及四个频带的局部幅值统计预测9邻域重组权重，在40×40→80×80位置进行条件上采样。借鉴 UPLiFT，独立实现频率条件支路，不引用 A 的模块。频带幅值提供纹理信息，由训练决定如何利用，不预设高频就是目标。

两模块通过频带幅值显式连接。`no_guidance.py` 保留频带调制和局部上采样，仅关闭幅值统计到上采样的直接连接；调制后的 C3 仍参与融合。仅上采样配置去掉频率支路；仅频带调制配置不计算幅值统计。FFT 固定 FP32；硬频带截断可能产生振铃，频域边界假设和实际开销均需结合结果判断。

### 实验判断与对照

按未增强、非 difficult 标注框的面积小于32²统计，train 为13,752/32,591（42.2%），val 为15,339/35,434（43.3%）；val 小框中船和车辆合计占87.9%。细节增强有数据依据，但应同时检查船/车辆 AP、小目标 AP、总体 mAP 和大目标表现，不能把总分变化直接解释成小目标改进。

| 要检验的说法 | 需要比较的配置 |
|---|---|
| A/C 的显式引导分支有贡献 | 同方法 `full.py` 与 `no_guidance.py` |
| 两个模块都有贡献 | 基线、两个单模块、完整组合 |
| B 的幅值机制优于普通上下文聚合 | `context_only.py` 与 `linear_context_only.py`（同参数量） |

A/C 的关闭引导对照会移除相应投影参数，只能评估整个引导分支，不能单独排除容量变化的影响。A/C 的单上采样还分别使用48/32维引导，跨方案分数差不能直接归因于状态/频率机制。局部上采样仅重组粗特征，细节仍依赖 C3 横向输入；高频也不等同于前景。

以单模块的精度收益、参数和计算开销为筛选依据，不要求两个模块直接关联，也不固定按 A/B/C 成组筛选。优先考虑 MALA 上下文、浅层细节和频带选择；压缩状态可作为 MALA 的上下文替代候选，局部上采样先选择一个宽度验证。有效模块再组合，只有需要解释引导机制时才追加对应对照，避免全部配置盲跑300轮。所有正式比较保持相同训练预算、batch 和指标；若显存要求降低 batch，应对该比较组统一调整。短训练只能排除实现问题，不能作为精度证据。

当前模块均增加参数与计算，属于精度增强候选，尚无真正减参方案。640×640、batch=1、检测头原始输出的计算统计如下；基线为4,878,708参数、7.987703 GMAC。MAC仅统计卷积、线性层和矩阵乘，不包含 FFT、逐元素操作或数据搬运，不等同于完整 FLOPs 或实际延迟。

| 单模块配置 | 新增参数 | 新增上述计算（GMAC） |
|---|---:|---:|
| A 压缩状态 | 27,857 | 0.035208 |
| A 局部上采样 | 9,706 | 0.061747 |
| B MALA 上下文 | 26,305 | 0.043162 |
| B 浅层细节 | 30,073 | 0.211968 |
| C 频带选择 | 2,513 | 0.015360（另有 FFT） |
| C 局部上采样 | 6,474 | 0.041165 |

### 启动候选训练

继续使用官方入口，从项目根目录将自定义包加入 `PYTHONPATH`。例如下面命令会启动 A 的完整300轮训练；替换配置路径可选择其他组合或消融。

```bash
conda activate rtmdet
cd /home/pwt/Paper1
PYTHONPATH=/home/pwt/Paper1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python mmyolo/tools/train.py configs/state_guided/full.py
```

每个配置的 `work_dir` 独立，默认从零开始。训练前按 GPU 实际余量选择卡与 batch；变更 batch 时同步调整学习率。

### 验证范围

基线及12个候选配置均已核对640训练、后期增强和验证/测试尺寸；候选配置通过训练协议一致性检查。基线和6个单模块完成640输入的CPU前向与计算量统计。GPU1 的模块复核覆盖3个完整组合、3个机制对照及 A 的单上下文配置，共7组：800×800、batch=2，16张训练图像运行两轮（16次迭代），第二轮切换弱增强，8张 val 图像运行 VOC/COCO 评估。短测关闭 warmup，检查新增参数全部进入优化器、收到有限非零梯度并发生更新；loss 和预测框/分数有限。极小验证子集缺失类别的 AP 为 NaN 属于无有效样本，不能解释为数值发散。

| 当前完整组合 | 参数量 | 短测峰值张量显存（MiB） |
|---|---:|---:|
| A | 4,919,407 | 850 |
| B | 4,935,086 | 840 |
| C | 4,887,855 | 930 |

基线参数量为4,878,708。显存为共享 GPU 上 batch=2、关闭 cuDNN benchmark 时的 `max_memory_allocated`，不含 CUDA 上下文、其他进程或缓存保留量；不能据此推断正式 batch=12 的余量或速度。检查只证明短程可训练，不证明精度收益或长期稳定性。正式配置为640×640、300轮、batch=12/8、原始 warmup；上述800短测用于验证模块实现，不代表640训练结果。测试不产生检查点，临时产物清理。

`tests/test_candidate_modules.py` 的13项检查覆盖频带重建/频率区分/框架初始化、非恒定特征的邻域取值、引导梯度、线性注意力显式矩阵和幅值对照、配置协议一致性、对照开关、关闭模块时与原版颈部一致，以及奇数/矩形尺寸。
