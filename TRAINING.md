# RTMDet-tiny / DIOR 实验

## 当前配置

GPU1当前训练标准DSConv，后续依次为P3对照A（密集空间卷积）→ P3对照B（去中间SiLU）→ G8 → G16 → 四处SCDown＋G2；四处版＋G4暂不接续。SCDown实验1/2/3与G4已完成300轮，VOC mAP/COCO AP分别为75.74%/56.4%、76.11%/56.7%、76.12%/56.4%、75.58%/56.2%；仅匹配GT与随机等数量GT的ROI对照为75.68%/56.5%、75.99%/56.4%。其余已完成方法的结构和结果见下文。LSConv neck内部块版已取消；细小目标匹配、轻量检测头、`lsconv_csp`、`lsblock_csp`暂不接续。使用Conda `rtmdet`与官方 `mmyolo/tools/train.py`，中断恢复只使用本实验的完整状态。

| 项目 | 设置 |
|---|---|
| GPU | GPU1，RTX4090，CUDA 可用容量约47.37 GiB |
| 输入 | 640×640；训练裁剪/填充，评估保持比例缩放并填充 |
| 数据 | 官方 train 5862 → val 5863；test 不参与 |
| 初始化 | 实验从零训练；中断后仅恢复本实验的完整检查点 |
| 主指标 | VOC2007 mAP@0.5，11点插值 |
| 其他指标 | COCO AP50:95、AP50、AP75、各尺度与各类别 AP |
| 训练 batch | 当前及后续候选真实batch36、不累积 |
| 验证 batch | 8 |
| 精度 | 训练 AMP，验证 FP32 |
| 预算 | 300轮，每10轮验证，包括第300轮 |
| 优化器 | AdamW；lr=0.0005625，weight decay=0.05 |
| 调度 | warmup1000迭代；第150–300轮 cosine，最低lr为初始值的5% |
| 增强 | 前280轮 Mosaic/MixUp，最后20轮弱增强 |
| EMA | interval1/gamma2000；momentum=0.0002，更新缓冲区 |
| 随机种子 | 42 |
| 检查点 | 每轮保存，保留最近两个；另保存 VOC/COCO 主指标各自最佳权重 |
| 推理 | score=0.001，NMS IoU=0.65，每图最多300框 |

同分辨率对照使用已完成的 `work_dirs/rtmdet_tiny_dior_640_gpu1_bs36/`；骨干三处LSConv、FDW及新四项沿用其真实batch36和训练协议。EFS两版为9×4，LSConv扩展为12×3；等效batch都为36，但BN统计及按微batch归一化的损失不完全等价。接续不会自动改变batch或梯度累积次数。

GT在评估图像缩放后加载，保持原图坐标，预测框还原到原图后评价。COCO JSON 仅为标注格式，不代表 COCO 预训练或 COCO 数据。

## EFS 的作用与接入

借鉴 [DM-EFS，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.pdf) 的 Expanded Features Set：增加浅层高分辨率特征，并贯通颈部和检测头。原论文 EFS 单独在 YOLOv7/VisDrone 上将 AP50 从47.81提高到52.01；该数字不作为 DIOR 的预期收益。

| 位置 | 基线 | EFS |
|---|---|---|
| 骨干输出 | C3/C4/C5 | C2/C3/C4/C5，通道48/96/192/384 |
| 颈部 | 三尺度 CSPNeXtPAFPN | 四尺度 CSPNeXtPAFPN，各输出96通道 |
| 检测步长 | 8/16/32 | 4/8/16/32，头部和先验同时修改 |
| 640下特征尺寸 | 80/40/20 | 160/80/40/20 |
| 预测点 | 8400 | 34000 |
| 参数量 | 4,878,708 | 5,005,956（增加127,248） |
| 卷积/线性计算，640、batch1 | 7.9877 GMAC | 18.3769 GMAC（约2.30倍） |

原论文使用 YOLOv7；本项目使用 CSPNeXt 与 RTMDet 的颈部、检测头、损失和动态标签分配。保留 EFS 的高分辨率特征与联合预测机制，不启用 DFM 动态控制器。区别于浅层细节适配器，C2 不再仅降采样注入 C3，而是参与完整四尺度融合，并在 stride4 上直接分类、回归和分配正样本。

预期改善船、车辆等小目标的特征表达和训练覆盖；同时检查整体 mAP、船/车辆类别 AP、小目标 AP，以及中大目标表现。P2显著增加头部计算和匹配显存，不能从小幅参数增长推断训练开销很小。

EFS已完成300轮：VOC mAP75.55%、COCO AP50:95为56.3%、小目标AP9.1%；基线分别为75.65%、56.4%、7.5%。该实验使用微batch9、累积4、warmup4000和EMA interval4/gamma400，BN按9张图计算，训练设置与真实batch36有差异。

EFS源配置已将EMA gamma校正为8000，以匹配MMYOLO实际默认gamma2000及四次梯度累积；测试通过实际EMA类核对更新系数。上述已完成结果仍来自gamma400的原实验快照，不代表修正后训练结果。

第300轮原始权重也已在完整5863张val上复评：基线/EFS的VOC mAP为75.95/75.99，COCO AP为56.7/56.7，小/中/大目标AP为8.3/35.7/70.0与9.3/34.6/69.0。两种权重口径均显示小目标收益、整体基本持平。完整分析见 `work_dirs/diagnostics/efs_final_analysis.json`，曲线见同目录 `efs_analysis.png`；复评不重新训练，不替换原EMA实验记录。

上述 MAC 统计原始检测头输出的卷积和线性层乘加，不包含归一化、激活、标签分配或NMS，不等同于完整FLOPs或实际延迟。

## EFS适配：保留P3侧向特征与独立P2检测塔

[configs/efs_adapt/rtmdet_tiny_dior_640.py](configs/efs_adapt/rtmdet_tiny_dior_640.py) 独立导入 `custom_models.efs_adapt`，继承修正EMA后的EFS协议，保持640、官方train/val、从零训练300轮、每10轮验证。微batch9、累积4，有效batch36；lr=0.0005625，warmup4000微迭代，EMA interval4/gamma8000。

`neck.py`保留融合后的P3侧向特征为96通道，另经96→48投影和上采样生成P2。第一处PAN拼接48通道下采样P2与96通道P3，CSP输入144、输出96；后续PAN与四个96通道输出沿用官方实现。`head.py`仅为P2分类、回归检测塔分离中间卷积，P3–P5继续共享卷积；BN和最终预测层仍按尺度独立。RTMDet损失、联合分配、34000个预测点保持一致，不包含DFM控制器。

全模型5,342,340参数，640、batch1的卷积/线性计算约18.4064 GMAC；分离检测塔增加参数，不减少P2计算。两项修改不隔离共享骨干和PAN路径的训练信号。

检查见 `work_dirs/efs_adapt_validation/checks.json`。六项结构检查包含与基线FPN的P3数值一致性、144通道实际拼接、检测塔参数和梯度归属、四尺度先验、真实检测损失更新与EMA/检查点共享关系。GPU短测覆盖强弱增强、真实更新、VOC/COCO验证及EMA交换恢复。GPU1正式batch9、累积4、每图1200框的三次压力更新已通过，最新峰值张量32.32 GiB、保留36.80 GiB。

EFS适配已完成300轮：EMA的VOC mAP75.62%、COCO AP56.4%、小/中/大目标AP为9.7/34.6/68.9；基线为75.65%、56.4%、7.5/34.9/69.8。小目标收益保留，整体基本持平。实际配置为 `work_dirs/efs_adapt_queue/efs_adapt_640_bs9x4.py`，输出为 `work_dirs/efs_adapt_rtmdet_tiny_dior_640_gpu1_bs9x4/`。

## 启动与状态

当前运行DSConv，随后为P3对照A、P3对照B、G8、G16、四处SCDown＋G2；均从零训练。状态与最近训练日志查询：

```bash
tail -n 20 /home/pwt/Paper1/logs/scdown_dsconv_rtmdet_tiny_dior_640_gpu1_bs36.log
cat /home/pwt/Paper1/work_dirs/scdown_dsconv_queue/status.json
cat /home/pwt/Paper1/work_dirs/scdown_p3_dense_queue/status.json
cat /home/pwt/Paper1/work_dirs/scdown_p3_noact_queue/status.json
cat /home/pwt/Paper1/work_dirs/scdown_g8_queue/status.json
cat /home/pwt/Paper1/work_dirs/scdown_g16_queue/status.json
cat /home/pwt/Paper1/work_dirs/scdown_p3_g2_queue/status.json
nvidia-smi
```

各队列的PID及启动身份在其同目录`queue_identity.json`。接续须核对前序进程身份、成功完成300轮、最终有限验证及检查点保存。源码、容量检查或训练失败就停止，不自动改batch；细小目标匹配修正和轻量检测头不接续启动。

当前待运行队列启动前等待GPU1空闲至少46000 MiB，用正式batch36、不累积、每图1200框完成三次实际更新，分配器上限88%。压力数据含真实图像、重复标注和模拟框，不代表训练精度或速度；报告位于 `work_dirs/<方法>_validation/capacity_gpu1_bs36.json`。正式训练沿用原来的AMP动态缩放和warmup调度，检查不写检查点。

队列默认拒绝已有输出；显式断点恢复须核对同实验完整状态，并以官方`--resume`恢复。服务器重启后需重新检查进程和最近完整检查点再恢复，磁盘上的`training`或`waiting`不证明进程仍在运行。指标为0–1，论文百分数乘100。

## LSConv：独立骨干实验

[configs/lsconv/rtmdet_tiny_dior_640.py](configs/lsconv/rtmdet_tiny_dior_640.py) 继承三尺度基线，独立导入 `custom_models.lsconv`。使用真实batch36、验证batch8、不累积梯度；lr=0.0005625，warmup1000迭代，EMA沿用基线。其余数据、300轮预算、每10轮验证、增强和评价与已完成的batch36基线一致。

借鉴 [LSNet，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.pdf) 的完整LSConv。1×1投影到半宽，7×7深度卷积与通道混合生成逐位置3×3动态核；GroupNorm保留有符号权重，按通道索引取模进行交错共享，最后执行局部聚合、BN和残差。每组动态核由8个通道共享。保留原CSP块的层后激活和后续5×5深度可分离卷积。

| 替换位置 | 内部通道 | 特征尺寸 |
|---|---:|---:|
| `stage2.1.blocks.0.conv1` | 48 | 80×80 |
| `stage3.1.blocks.0.conv1` | 96 | 40×40 |
| `stage4.2.blocks.0.conv1` | 192 | 20×20 |

骨干仍输出C3/C4/C5的96/192/384通道；颈部、检测头及先验保持stride8/16/32，共8400个预测点。参数量4,517,130，比基线减少7.41%；普通卷积为7.660804 GMAC，另有局部加权约0.004838 GMAC。官方算子参考版本为 `cbe737c92b7c43ecf02d08545a07f03f1010177c`；Triton前后向已适配现有Torch2.0/Triton2.0，不升级环境。

GPU0共享环境、AMP、batch2的10步短测中，基线约0.144秒/步、LSConv约0.196秒/步，不含数据增强；小批次下额外算子开销明显，减参不保证训练加速，也不能据此推算正式batch36速度。

LSConv已完成300轮：EMA的VOC mAP75.57%、COCO AP50:95为55.9%、小目标AP7.8%；同轮基线分别为75.65%、56.4%、7.5%。VOC差约−0.08个百分点；参数减少7.41%，实际推理速度收益尚未验证。实际配置快照位于 `work_dirs/lsconv_queue/lsconv_640_bs36.py`，训练输出为 `work_dirs/lsconv_rtmdet_tiny_dior_640_gpu1_bs36/`。

单独手动启动命令（已有接续队列时不要重复启动）：

```bash
PYTHONPATH=/home/pwt/Paper1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python mmyolo/tools/train.py configs/lsconv/rtmdet_tiny_dior_640.py
```

## LSConv扩展：骨干、颈部和检测塔

[configs/lsconv_full/rtmdet_tiny_dior_640.py](configs/lsconv_full/rtmdet_tiny_dior_640.py) 使用独立的 `custom_models/lsconv_full/`。完整LSConv的LKP、GroupNorm、SKA、输出BN和残差保持一致，不修改已完成的三处替换实验。

骨干与颈部共8个CSPNeXt块将`conv1`替换为LSConv，保留后续卷积和CSP连接。颈部三个输出使用`1×1 Conv/BN → LSConv → SiLU`；检测塔中间层使用`LSConv → 1×1 Conv/BN → SiLU`。检测塔四组卷积参数继续跨三个尺度共享，BN/GN按尺度独立，分类与回归分开；最终预测层、三尺度8400个预测点、损失和分配器沿用基线。一张图实际执行23次LSConv，其中检测塔12次。

参数3,385,305，较基线减少30.61%；640、batch1卷积乘加4.429809 GMAC，加上局部加权约4.482685 GMAC，较基线所统计MAC减少43.88%。MAC不包含归一化、激活和调度。共享GPU0上，batch1、AMP、逐用途融合Conv/BN的普通前向为基线11.06 ms、扩展版26.12 ms；CUDA Graph对照为1.27/2.10 ms，仅用于定位调度开销。当前不能将减参、减少MAC表述为推理加速。

正式配置640、从零训练300轮、每10轮验证，微batch12、累积3、有效batch36、lr=0.0005625；warmup3000微迭代，EMA interval3/gamma6000。BN统计基于12张图，与基线真实batch36有差异。检查摘要见 `work_dirs/lsconv_full_validation/checks.json`：五项集成测试覆盖尺寸、实际调用次数、跨尺度梯度求和、所有新增参数更新及检查点读写。GPU0官方入口短测18次微迭代、六次实际更新，342个新增参数张量均有有限非零梯度并更新；强弱增强、两次32图验证和两次EMA交换恢复通过，未保存检查点。

LSConv扩展已完成300轮：EMA的VOC mAP71.22%、COCO AP51.3%、小/中/大目标AP为6.2/31.4/62.3。基线对应75.65%、56.4%、7.5/34.9/69.8。GPU1的12×3压力检查峰值张量12.93 GiB、保留14.54 GiB；未验证更大的batch。

队列依赖EFS适配成功完成300轮及有限验证。启动前检查官方入口、配置和源文件校验值；等待GPU1空闲至少45000 MiB，使用正式batch12、累积3、每图1200框、88%分配器上限完成三次压力更新后再训练。压力数据使用不同真实图像、重复标注和明确的多尺度模拟框，只检验运行与容量。配置快照为 `work_dirs/lsconv_full_queue/lsconv_full_640_bs12x3.py`，输出为 `work_dirs/lsconv_full_rtmdet_tiny_dior_640_gpu1_bs12x3/`。

## LSConv的两项CSP内部变体

两版使用独立目录和配置，都只改变骨干4处及颈部内部4处。CSP分流/合流、颈部输出层、检测塔、三尺度先验和损失沿用基线；没有新增P2，没有颈部输出投影或动态检测塔。640、官方train/val、从零300轮、每10轮验证、真实batch36、不累积、lr=0.0005625、warmup1000和基线EMA完全一致。两项当前均未排入完整训练。

| 配置目录 | 内部块的接法 | 参数 | 卷积＋局部聚合GMAC，640/batch1 |
|---|---|---:|---:|
| 基线 | 原CSPNeXtBlock | 4,878,708 | 7.9877 |
| `configs/lsconv_csp/` | conv1换为LSConv→SiLU，保留原conv2和该块原外层连接 | 4,083,981 | 7.1426 |
| `configs/lsblock_csp/` | 整个内部块换成LSConv＋残差FFN，移除原conv2和该块原外层连接 | 4,385,445 | 7.4504 |

普通扩展版`lsconv_csp`严格沿用旧三处版的层内接法，只扩大到8处。完整块版`lsblock_csp`为`z=LSConv(x); y=z+FFN(z)`；FFN采用作者的`1×1:C→2C/BN → ReLU → 1×1:2C→C/BN`，末端BN缩放为0，不在完整块外再加SiLU或残差。它对应作者LSConv所在Block的主体，外侧仍是本项目CSP结构，不是完整LSNet复现。所统计MAC不含归一化、激活和核调度，未据此判断实际加速。

两版分别保存经过验证的LSConv核心，不互相导入。新增颈部模块在父级递归初始化期间被暂时移出初始化遍历，并在`finally`恢复到原槽位，保留作者核生成器的构造初始化及FFN零缩放；周围原有层继续使用官方初始化。测试核对了全模型初始化前后新增状态完全一致、重复初始化无改动，以及初始化异常时模块和顺序完整恢复。

`tests/test_ls_csp_variants.py`的8项测试通过：包括普通版缩回三处后与旧版全部初始状态及输出逐项完全一致、8处真实调用和通道、原检测头共享关系、初始化、零FFN第一步梯度与随后激活、真实检测损失更新和状态读写。作者参考提交为`cbe737c92b7c43ecf02d08545a07f03f1010177c`；从固定源码执行原LSConv/Block类，SKA用独立显式坐标移位参考。24/48/96/192通道、训练/评估、FP32/AMP共32组前向与梯度对照通过；对照时关闭TF32并固定cuDNN算法，正式训练设置不变。

GPU0两版均以官方入口完成真实16图、batch2、两阶段共16次实际更新、两次32图VOC/COCO验证、原始预测、EMA交换恢复及模型/优化器/AMP状态读写。普通版120个、完整块版168个相关参数张量均有有限非零梯度并更新；完整块允许零初始化FFN在第一步之后逐渐激活。短测关闭warmup、loss scale32，只检查执行，不用于判断精度，不保存检查点。

GPU0另以batch12、每图1200框完成各三次压力更新，峰值张量分别12.06/12.03 GiB，无AMP跳步。这个较小batch仅用于共享GPU容量检查，正式batch36配置没有改变；正式batch36容量尚未验证。汇总位于 `work_dirs/lsconv_csp_validation/checks.json`、`work_dirs/lsblock_csp_validation/checks.json`。

```bash
python -m unittest discover -s tests -p 'test_ls_csp_variants.py' -v
PYTHONPATH=/home/pwt/Paper1 python scripts/lsconv_csp_probe.py \
  --method lsblock_csp --mode preflight \
  --report work_dirs/lsblock_csp_validation/preflight.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/pwt/Paper1 python scripts/lsconv_csp_probe.py \
  --method lsblock_csp --mode smoke --allocator-fraction .25 \
  --report /home/pwt/Paper1/work_dirs/lsblock_csp_validation/smoke.json
```

普通版将`lsblock_csp`替换为`lsconv_csp`。运行GPU检查前核对该卡空闲显存；`fidelity`模式通过`--reference`指定固定版本`model/lsnet.py`，`capacity`模式默认检查正式batch36，`--batch`只调整独立压力检查。

## LSConv：neck与检测头的两组独立实验

共同基础为 `configs/lsconv/rtmdet_tiny_dior_640.py` 的三处骨干接法，已完成VOC mAP75.57%、COCO AP55.9%。新两组使用其结构，不加载已有权重；分辨率、数据、真实batch36、优化器、增强、EMA及300轮预算完全一致。

| 配置 | 相对三处版的改动 | 总参数 | MAC，640/batch1 |
|---|---|---:|---:|
| 三处版对照 | 无 | 4,517,130 | 7.6656 G |
| `configs/lsconv_neck/rtmdet_tiny_dior_640.py` | neck内部四个CSPNeXtBlock整体换为LSBlock | 4,250,532 | 7.3950 G |
| `configs/lsconv_head/rtmdet_tiny_dior_640.py` | 分类/回归检测塔仅第二层换为LSBlock | 4,460,562 | 7.1487 G |

LSBlock对应作者LSConv所在Block的主体：`z=LSConv(x); y=z+FFN(z)`，FFN为两层1×1、扩展比2、末端BN缩放初始化为0。没有额外外层残差或SiLU。neck版替换宽度为96/48/96/192，保留CSP分流/拼接、降采样、所有原输出3×3卷积和检测头。检测头版保留原neck及检测塔第一层3×3；第二层LSBlock的全部卷积跨P3–P5共享，BN/GN及BN统计逐尺度独立，分类/回归分开，最终预测层不变。两组新增代码分别在同名独立目录，原三处骨干保持不变。

初始化保留作者LKP偏置和FFN零缩放，未替换参数及全局随机状态与三处对照完全一致。8项集成测试覆盖范围、初始化、共享梯度求和、保存加载和异常恢复；作者代码GPU FP32/AMP、训练/评估前向与梯度对照共16组通过。两组均经官方入口完成16次实际更新、强弱增强、两次32图验证、EMA及模型/优化器/AMP状态读写，相关参数均更新，无AMP跳步。GPU0另以batch12、每图1200框完成三次压力更新，峰值张量分别11.95/12.30 GiB；这些检查关闭warmup、使用loss scale32，不改正式配置，不保存检查点。

neck版已取消；检测头版已完成300轮：VOC mAP75.27%、COCO AP54.7%，三处骨干对照为75.57%/55.9%。汇总在 `work_dirs/{lsconv_neck,lsconv_head}_validation/checks.json`，复核入口为 `scripts/lsconv_neck_head_probe.py --method lsconv_neck --mode preflight --report <路径>`。MAC包括卷积与9点SKA，不含归一化、激活、残差加法或后处理，不据此推断实际延迟。

## LSConv：neck输出先聚合后投影

[configs/lsconv_neck_out/rtmdet_tiny_dior_640.py](configs/lsconv_neck_out/rtmdet_tiny_dior_640.py) 导入独立的 `custom_models/lsconv_neck_out`，共用已验证的三处LSConv骨干结构，不加载权重。仅将最终三个输出密集3×3改为 `LSConv(Cin) → 1×1(Cin→96)/BN/SiLU`，Cin分别96、192、384。动态核生成和值聚合均发生在投影之前，LSConv自带BN和残差，外部不增加FFN或额外残差。FPN/PAN内部融合块、上下采样、检测塔与8400个预测点保留。正式协议为640、官方train/val、从零训练、真实batch36、不累积、300轮、每10轮验证，与三处版一致。

总参数4,277,142；普通卷积7.093636 GMAC、9点SKA为0.014515 GMAC，合计7.108151 GMAC。相对三处版减少5.31%参数、7.27% MAC；相对原始基线减少12.33%、11.01%。MAC不含归一化、激活、残差加法和后处理，不代表实测延迟。

已完成300轮，EMA的VOC mAP75.12%、COCO AP55.6%、AP75为61.3%，小/中/大目标AP为8.0/33.6/68.7；三处骨干版为75.57%、55.9%、61.4%和7.8/33.9/69.1。30次同轮数VOC验证均低于三处版；最终与原始/EMA权重均有限，48900次训练迭代、48878次优化器更新。当前结果支持轻量化折中，未显示精度增强收益；未单独分离空间/通道分解及各尺度的贡献。完整对照与曲线在 `work_dirs/diagnostics/lsconv_neck_out_analysis/`。

初始化保留作者LKP状态；新1×1采用官方neck的Kaiming配置，未替换参数与随机状态逐项匹配三处对照。5项测试覆盖实际输入路径、聚合前通道宽度、三尺度输出、独立参数、初始化及异常重试；96/192/384通道、FP32/AMP、训练/评估共12组作者源码前向与梯度对照通过。官方训练入口完成16次真实更新、强弱增强、两次32图验证、原始/EMA预测及模型/优化器/AMP状态读写；99个相关参数张量均得到有限非零梯度并更新，无AMP跳步，不保存检查点。

GPU0另以batch12、每图1200框完成三次压力更新，峰值张量12.08 GiB、保留13.77 GiB。短测和压力检查关闭warmup、使用loss scale32，仅验证执行与容量；正式配置不变。GPU1正式batch36、每图1200框也已完成三次压力更新，无AMP跳步，峰值张量36.10 GiB、保留40.77 GiB；检查通过后从零启动完整训练，正式AMP缩放与warmup保持原配置。检查汇总在 `work_dirs/lsconv_neck_out_validation/checks.json`；复核入口为 `scripts/lsconv_neck_out_probe.py`，提供`preflight`、`fidelity`、`smoke`、`capacity`，其中`capacity --batch 12`仅调整独立压力测试，省略`--batch`检查正式batch36。

## LSConv 20处：保留原颈部输出与作者初始化

[configs/lsconv_20/rtmdet_tiny_dior_640.py](configs/lsconv_20/rtmdet_tiny_dior_640.py) 独立导入 `custom_models.lsconv_20`。骨干4个、颈部内部4个CSP块仅替换conv1，保留原conv2与连接；检测塔12次执行 `LSConv → 1×1 Conv/BN → SiLU`，卷积跨尺度共享、BN/GN按尺度独立、分类回归分开。三个颈部输出保留官方3×3，通道96/192/384→96。三尺度、8400预测点、预测层、损失与分配器沿用基线。

初始化时仅暂时移出本变体的LSConv子模块，并在`finally`恢复原槽位；其构造参数、cv4偏置及归一化缓冲区全部保留。颈部其余层继续官方Kaiming初始化；检测塔1×1投影和预测层继续RTMDet初始化，分类先验为0.01。测试覆盖根模型初始化、重复初始化、异常恢复、共享参数身份及共享梯度求和。

参数3,855,645，较基线减少20.97%；640、batch1卷积5.134199 GMAC，加局部聚合0.045619 GMAC，共5.179818 GMAC，减少35.15%。按1 MAC=2 FLOPs换算为10.359636 GFLOPs；未统计归一化、激活、残差加法和后处理。正式协议与已完成真实batch36基线一致：640、从零训练、官方train/val、300轮、每10轮验证、不累积、lr=0.0005625、warmup1000、EMA interval1/gamma2000。

9项集成检查通过；统一权重后，全模型输出与“23处版恢复三个原颈部输出层”逐项完全一致。20组固定作者源码对照覆盖24/48/96/192通道、训练/评估、FP32/AMP及实际特征尺寸。GPU0官方入口真实16图完成16次更新、两次32图验证、两次EMA恢复及模型/优化器/AMP状态读写；288个相关参数张量均有有限非零梯度并更新，预测偏置未为检查额外修改。短测关闭warmup、loss scale32，仅验证运行。

GPU0的batch12、每图1200框压力检查完成三次更新，无AMP跳步；峰值张量12.76 GiB、保留14.39 GiB。GPU1正式batch36、每图500框的三次压力更新已通过，峰值张量20.97 GiB、保留23.00 GiB；每图1200框测试曾在匹配阶段OOM，500框检查不代表任意密度均能运行。正式训练仍为真实batch36、不累积，使用冻结配置 `work_dirs/lsconv_20_queue/lsconv_20_640_bs36.py`。检查摘要在 `work_dirs/lsconv_20_validation/checks.json`。

已完成300轮：EMA的VOC mAP73.49%、COCO AP53.3%、小/中/大目标AP6.3/32.3/66.2；基线为75.65%、56.4%、7.5/34.9/69.8。第290轮20类COCO AP均低于基线。检测头参数341,064→112,728（−66.95%），该部分MAC下降69.94%；主要结构疑点是把密集空间/通道联合映射改为动态逐通道聚合再单次1×1混合。第296轮的同一256图诊断中，EMA/原始权重VOC为72.65/73.20、COCO AP为53.3/53.5；EMA权重单换原始BN缓存后为69.59/51.1，不支持靠单换BN缓存修复。该子集结果不替代正式val，检测塔影响尚未与8处CSP内部替换隔离。分析见 `work_dirs/diagnostics/lsconv20_analysis/report.json`。

```bash
python -m unittest discover -s tests -p 'test_lsconv_20.py' -v
PYTHONPATH=/home/pwt/Paper1 python scripts/lsconv_20_probe.py \
  --mode preflight --report work_dirs/lsconv_20_validation/preflight.json
# GPU1空闲后复核正式batch36；省略--batch即使用正式配置。
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/pwt/Paper1 python scripts/lsconv_20_probe.py \
  --mode capacity --allocator-fraction .88 --gt-per-image 500 \
  --report work_dirs/lsconv_20_validation/capacity_gpu1_bs36.json
```

## FDW：独立频域动态卷积

[configs/fdw/rtmdet_tiny_dior_640.py](configs/fdw/rtmdet_tiny_dior_640.py) 导入独立的 `custom_models.fdw`，与LSConv一样使用真实batch36、验证batch8、lr=0.0005625、warmup1000和基线EMA，保持640、300轮、官方train/val与相同增强和评价。

保留 [FDConv，CVPR2025](https://github.com/Linwei-Chen/FDConv) 的FDW、输入条件核混合与原作者的缩放；关闭额外KSM调制和FBM。将权重重排矩阵的半谱按频率距离等分为16组，用GAP→1×1→BN→StarReLU→1×1→Sigmoid生成每个样本的混合系数，混合频谱后逆FFT合成卷积核。该频率包含通道维度，不直接等同于图像高低频；合成核在样本内共享，不是逐像素动态核。

替换 `stage2.1.blocks.0.conv1.conv`、`stage3.1.blocks.0.conv1.conv`、`stage4.2.blocks.0.conv1.conv`，内部通道48/96/192，特征80/40/20；保留原BN、SiLU、后续5×5卷积和CSP连接。三尺度输出及8400个预测点沿用基线。16组能等分三个半谱，原论文常用的64组无法直接等分前两个位置。

由相同种子的随机卷积权重转换频谱；路由器采用原作者的小权重初始化，未替换层的初始权重与基线一致。全模型4,887,018参数，增加8310（0.17%）。卷积统计约7.987709 GMAC，另有FFT与核合成开销。共享GPU0、AMP、batch2短测约0.209秒/步，同次基线约0.177秒/步，不含增强，不用于推算正式batch36速度。

检查摘要位于 `work_dirs/fdw_validation/checks.json`。原作者参考版本为 `bf7a33d4f39ac7ea8172158f5628107fcaaba469`；三种宽度与关闭额外调制后的官方算子输出一致，梯度误差在浮点容差内。真实16图、batch2、两轮共16次优化器更新，24个新增参数张量均收到有限非零梯度并更新；覆盖强弱增强、两次32图VOC/COCO验证、原始权重预测及EMA交换恢复，不保存检查点。

FDW已完成300轮：VOC mAP75.44%、COCO AP56.2%、小/中/大目标AP为7.6/34.2/69.5；基线分别为75.65%、56.4%、7.5/34.9/69.8。正式batch36压力检查通过，峰值张量35.18 GiB、保留39.71 GiB。实际配置为 `work_dirs/fdw_queue/fdw_640_bs36.py`，输出为 `work_dirs/fdw_rtmdet_tiny_dior_640_gpu1_bs36/`。

完整5863张val的配对诊断中，动态路由VOC为75.438%，将三处路由固定为64张无标签val图像的平均值后为75.340%，COCO AP均为56.2%；权重和评估设置相同。该结果说明这份权重对输入条件路由的精度依赖较小，不等同于另训静态模型。记录见 `work_dirs/diagnostics/fdw_routing_eval.json`。据此增加下面的局部KSM分支，检验更细的权重调制能否改善现有FDW。

## FDW＋局部KSM

[configs/fdw_ksm/rtmdet_tiny_dior_640.py](configs/fdw_ksm/rtmdet_tiny_dior_640.py) 使用独立的 `custom_models/fdw_ksm/`：`frequency.py`保存FDW核心，`local_modulation.py`实现原作者局部KSM，`operator.py`组合两者，`backbone.py`接入原来三处48/96/192通道卷积。邻接通道卷积的核长分别为3/5/5，颈部、检测头和预测点不变。

新增路径为 `GAP → LayerNorm → Conv1d → 重排为[Cout,Cin,3,3] → 2×Sigmoid`，生成的调制张量逐元素乘到FDW合成核上，再卷积输入。局部指池化描述符的通道邻域；一个样本仍使用一组空间共享的卷积核。保留原作者1e-6的Conv1d初始化，使初始调制接近1；未启用全局KSM的通道/滤波器/空间三轴调制及FBM。

全模型4,901,946参数，比FDW增加14,928（0.31%），比基线增加23,238（0.48%）。统计卷积乘加为7.989845 GMAC，其中新增Conv1d约0.002136 GMAC；不包括FFT、核合成和逐元素运算。训练协议与已完成FDW一致：640、官方train/val、从零训练300轮、每10轮验证、真实batch36、lr=0.0005625、warmup1000、EMA interval1/gamma2000。相同种子下，所有原有FDW和其他层的初始状态一致，仅增加9个局部分支参数张量。

检查汇总见 `work_dirs/fdw_ksm_validation/checks.json`。7项测试覆盖非均匀调制的通道顺序、非连续输入、前向/梯度、GPU FP32/AMP、原FDW退化特例、检测梯度、EMA/检查点和训练协议。三种实际宽度及特征尺寸与作者算子在训练/评估模式下对照，前向误差均为0，梯度最大误差1.35e-7以内。GPU0官方入口短测16次更新，33个相关参数张量全部有有限非零梯度并更新，两次32图VOC/COCO验证及EMA交换恢复通过，峰值张量1.95 GiB；短测关闭warmup、使用loss scale32，未保存检查点。

GPU0的batch12、每图1200框压力检查完成三次更新，峰值张量11.81 GiB、保留13.44 GiB，无AMP跳步。压力数据含重复标注和模拟框，仅用于容量检查。正式batch36容量尚未验证，FDW＋局部KSM当前未排入完整训练。复核脚本为 `scripts/fdw_ksm_probe.py`，支持`preflight`、`fidelity`、`smoke`、`capacity`；作者对照模式通过`--reference`指定原始算子文件。

## 四项独立候选

四个目录互不导入，各自继承三尺度基线：640、官方train/val、从零训练300轮、每10轮验证，真实batch36、lr=0.0005625、warmup1000、EMA interval1/gamma2000。独立配置均为 `configs/<目录>/rtmdet_tiny_dior_640.py`。FreqFusion和实例语义监督已完成，匹配修正和轻量头暂未启动。FreqFusion第300轮：VOC mAP 76.22%、COCO AP 57.1%，同配置基线分别为75.65%、56.4%。

正式batch36容量在各自启动前检查；GPU0已分别以batch12、每图1200框验证容量脚本和三次真实更新，峰值张量约12.41/11.78/10.52/12.63 GiB，未跳过AMP更新、不保留检查点。这个较小batch只用于共享GPU检查，四项正式配置不变；它不代替GPU1上的batch36检查。

| 目录 | 训练参数 | 推理Conv/Linear GMAC | 目的 |
|---|---:|---:|---|
| 基线 | 4,878,708 | 7.9877 | 对照 |
| `freq_fusion` | 4,987,800 | 8.1975 | 改善跨尺度融合的内容一致性与边界 |
| `instance_semantic` | 4,880,648 | 7.9877 | 给每个实例提供不依赖匹配点数、IoU的辅助类别监督 |
| `tiny_assignment` | 4,878,708 | 7.9877 | 补充失去有效匹配的细小目标，并提供低IoU时的回归梯度 |
| `factorized_head` | 4,710,516 | 6.4975 | 降低检测塔卷积乘加，保留分类/回归独立性 |

GMAC口径不含动态滤波、重采样、归一化或后处理。语义分支的1940个参数保存在训练模型中，推理不执行该分支。轻量头全模型参数减少3.45%、所统计MAC减少18.66%，不能据此宣称延迟下降。

共享GPU0、batch1、640、AMP、未融合Conv/BN的普通前向测得中位耗时：基线12.16 ms、FreqFusion18.54 ms、轻量头14.36 ms（预热20次，三组各80次，不含预处理/NMS）。轻量头将检测塔卷积执行从12次增至36次；当前执行方式没有加速收益。此处不代表部署融合后或正式大batch训练的速度。记录见 `work_dirs/diagnostics/candidate_forward_latency_20261015.json`。

**完整FreqFusion**：对应 [TPAMI论文](https://arxiv.org/abs/2408.12879) 的作者检测配置，参考提交 `3fb0c70637a3c194fb74294d3ce4681958b26241`。`operator.py`保留5×5低通、3×3高通和增强初始融合；`resample.py`保留膨胀率2的余弦邻域、四组偏移、半像素采样和边界处理；`neck.py`接入C5→C4、C4→C3两次融合，原值分支为192/96通道，引导分支按作者压缩比8得到48/24通道。原CSP拼接、PAN侧向来源和三尺度检测头不变。CUDA使用MMCV CARAFE零填充；CPU参考与之数值一致。模块内用FP32避免CARAFE在AMP下混合不同数据类型，输出值恢复输入类型。作者外层Xavier初始化会覆盖偏移分支构造时的小/零初始化，本实现保留其最终行为；新增模块不改变其余基线权重的随机初始化。

**实例语义辅助监督**：`head.py`一次计算分类塔特征与原预测，`roi.py`对增强后的每个正面积GT按尺寸选P3–P5，RoIAlign 3×3、每格2×2采样后平均、Linear到20类，CE权重0.1。尺度映射为`floor(log2(sqrt(area)/56))`并限制在三层；不按检测正样本数重复实例，也不截断实例数量。RTMDet点坐标offset=0，取样明确抵消RoIAlign的半像素偏移，小于stride的框不扩大到一格。仅辅助损失排除增强后无有效面积的框，并按有效实例数归一化；所有正面积小框保留，原检测标注及QFL/GIoU不变。5项测试覆盖退化框、坐标、基线预测与梯度范围，修复后GPU短测16次更新和两次验证通过。已完成300轮，VOC mAP76.31%、COCO AP56.9%，基线为75.65%、56.4%。无教师、无额外预训练；这是直接语义监督的项目设计。

**细小目标匹配与回归**：`assigner.py`保留DSL原代价、dynamic-k及跨GT候选规则，显式传递GT索引。只对输入框短边小于8像素的目标处理：先清除被全INF代价勉强分配的无效点，再为未匹配GT从最细层、中心周围1.5个stride内的空闲点补匹配。增广路径处理新增匹配间的竞争，不抢已有有效正样本；局部候选不足时保留未匹配状态，不从远处硬补。`head.py`保留IoU加权GIoU，补足权重`max(0.05-IoU,0)`用于stride归一化的四边SmoothL1，回归分母按两者权重和计算；QFL质量标签仍为真实IoU，分类分母仍按IoU和计算。补足项不能直接乘GIoU：线性ltrb初始可能产生倒置框，GIoU对其可给出负损失和错误梯度。边界损失无需对预测排序或截断，并保留框外近邻点表达GT所需的有符号ltrb。阈值是本项目初始设计参数，不是论文已验证的最优值。

**轻量检测头**：`head.py`将每个96→96的3×3检测塔层替换为`1×1:96→192 → DW3×3:192 → 1×1:192→96`，每段均有独立BN与SiLU。分类与回归保持分开；三段卷积分别跨P3–P5共享，BN按尺度独立，最终预测层保持基线结构。没有新增残差或分类/回归共享。这是利用通道/空间分解的检测头设计，并非照搬InceptionNeXt的多分支结构。

检查覆盖核权重/边界、已知坐标平移、原颈部旁路数值、初始化、ROI坐标、重复GT索引、候选冲突和耗尽、回归梯度与归一化、共享梯度求和、EMA/状态读写。FreqFusion六组GPU FP32/AMP对照作者代码，前向误差为0，梯度最大绝对误差2.40e-5。四项均用官方训练入口完成真实16图、batch2、强弱增强共16次实际更新，以及两次32图VOC/COCO验证；相关参数全部获得有限非零梯度并更新。短测使用loss scale32并关闭warmup，仅验证执行，不用于估计精度，不保存检查点。报告位于 `work_dirs/<目录>_validation/`。

```bash
python -m unittest discover -s tests -p 'test_freq_fusion.py' -v
python scripts/candidate_probe.py --method freq_fusion --mode preflight \
  --report work_dirs/freq_fusion_validation/preflight.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/pwt/Paper1 python scripts/candidate_probe.py \
  --method freq_fusion --mode smoke --allocator-fraction 0.35 \
  --report /home/pwt/Paper1/work_dirs/freq_fusion_validation/smoke.json
```

其余三项将`freq_fusion`替换为相应目录名；GPU短测前先检查该卡可用显存。`fidelity`模式另需`--reference`指定固定版本作者检测目录下的`FreqFusion.py`。

## ROI语义监督：机理对照

三组各用独立的 `custom_models/instance_semantic_<名称>/` 和同名配置目录，继承原ROI版的640、官方train/val、真实batch36、不累积、从零300轮、每10轮验证及优化/EMA协议。辅助分类器仍为96→20、CE权重0.1；全模型4,880,648参数，推理Conv/Linear 7.987703 GMAC。

| 名称 | 唯一目的性变化 |
|---|---|
| `center` | 全部有效GT沿用原尺度选择，只在GT中心取一次双线性特征；无ROI区域平均 |
| `matched` | 原ROI取样与分类器保留，仅选择本步DSL已匹配的有效GT |
| `random` | 原ROI取样与分类器保留，从全部有效GT均匀随机选M个；M为本模型本步已匹配有效GT数 |

后两组分母始终使用全部有效GT数G，未改成M；同类、相同框的不同GT用真实索引区分，增强后退化框仅从辅助任务排除。随机数生成器独立于数据增强，状态随模型和EMA保存、恢复。D/E各自训练后的匹配数可以分化，并非逐步锁定成相同数量；正式日志记录 `semantic_all_gt`、`semantic_matched_gt`、`semantic_selected_gt` 供比较。公共目录 `roi_supervision_common` 只提供原DSL索引输出、原检测损失和GT映射；两个选择策略各自独立，没有修改旧方法或叠加类别排序损失。

15项新测试和原ROI的5项测试通过：覆盖单点取样独立参考及梯度、亚stride框与边界、精确GT身份、全GT分母、随机抽样与状态恢复、原检测损失、辅助梯度范围及单次BN更新。初始化和推理与原ROI版逐项一致；中心点零权重分支的梯度按FP32容差核对，不宣称不同计算图的逐位一致。三组各用官方入口完成16次实际AdamW更新、两次32图VOC/COCO验证、EMA交换及模型/优化器/AMP状态读写，22个相关参数张量均有有限非零梯度并更新，无AMP跳步，不保存短测检查点。

中心点版已通过GPU1正式batch36、每图1200框的三次压力更新，峰值张量35.18 GiB、保留39.86 GiB，随后从零启动；后两组等待各自启动前检查容量。接续顺序为 `center → matched → random`，各自状态在 `work_dirs/instance_semantic_<名称>_queue/status.json`。每组启动前等待GPU1空闲至少46000 MiB，以88%分配器上限、正式batch36、每图1200框做三次压力更新，通过后启动正式训练。检查汇总在各自 `work_dirs/instance_semantic_<名称>_validation/checks.json`；复核入口 `scripts/roi_semantic_ablation_probe.py --method instance_semantic_<名称> --mode preflight|smoke|capacity --report <路径>`。短测关闭warmup、loss scale32，仅检验执行；正式训练保留原AMP与warmup设置。

## 实例语义监督：检测类别排序

[configs/instance_semantic_rank/rtmdet_tiny_dior_640.py](configs/instance_semantic_rank/rtmdet_tiny_dior_640.py) 独立导入 `custom_models.instance_semantic_rank`，复用原整图GT语义监督，不使用裁剪分支。640、官方train/val、从零训练300轮、每10轮验证、真实batch36、不累积及优化/EMA协议均沿用原实例监督版。

`assigner.py`只在原DSL结果中返回实际GT索引，代价、dynamic-k、正样本和IoU质量标签保持原行为。`loss.py`按“图像索引、GT索引”归组所有尺度的正样本，先平均真实检测层 `rtm_cls` 的20维原始logits，再对每个匹配GT算一次softmax CE，最后按GT数平均；AMP下以FP32聚合。未匹配GT仍参与原ROI语义监督，不额外补匹配。

总损失为原QFL/GIoU＋0.1×原ROI类别CE＋0.1×实例排序CE。新增项约束类别之间的相对关系，不将检测真实类别的sigmoid分数强制设为1；它仍可能改变检测分数校准，完整训练需同时观察VOC、COCO及误检。没有增加可学习参数、分类器、图像前向或BN更新；保留原训练模型的4,880,648参数和推理Conv/Linear 7.987703 GMAC。新增梯度直接经过实际分类预测层、分类塔及共享特征，独立于旧辅助分类器。

9项新测试及原实例监督5项测试通过，覆盖官方分配/GT索引逐项对照、同类多实例、空GT、跨图跨层归组、实例权重、相对logit平移、空DDP分片归一化、原ROI保留及梯度范围。新增权重为0时，初始化、预测、原损失、全部梯度和一次AdamW更新与原版完全一致。GPU0官方入口短测完成16次实际更新、两次32图VOC/COCO验证、EMA交换及模型/优化器/AMP状态读写；22个相关参数张量均有有限非零梯度并更新，无AMP跳步。

GPU0 batch12、每图1200框完成三次压力更新，峰值张量11.78 GiB、保留13.42 GiB。GPU1正式batch36、每图1200框也已通过三次压力更新，峰值张量35.18 GiB、保留39.86 GiB。短测关闭warmup、loss scale32，不保留检查点，不代表精度收益。正式训练已完成300轮，EMA的VOC mAP75.56%、COCO AP55.7%、AP75为60.4%，小/中/大目标AP为8.9/35.0/68.7；原实例监督版对应76.31%、56.9%、62.1%、7.5/34.5/70.3。新增排序损失后VOC降低0.74个百分点、COCO AP降低1.2个百分点，16/20类的COCO AP下降；小目标收益未转化为整体提升。第300轮也是本次VOC与COCO最佳轮。

最终分析见 `work_dirs/diagnostics/instance_semantic_rank_analysis/final/analysis.json`。训练协议和源码核对一致，无非有限权重；平均logits的CE给同一GT的所有匹配点相同类别梯度，不区分框质量。最终权重在108张强增强训练图上的检查发现：IoU<0.1的134个点中，新增项真实类别梯度范数约为QFL的6.9倍，53.7%的点逆转了QFL要求的降分方向。同批512张val图的GT内部候选诊断中，最高分点类别正确率97.80%→98.26%，对应框平均IoU为0.752→0.745；该GT限定诊断不是检测AP，定位和分数选择均有变化，不能把完整掉点全部归给校准。

冻结配置为 `work_dirs/instance_semantic_rank_queue/instance_semantic_rank_640_bs36.py`；状态与进程身份在同目录 `status.json`、`queue_identity.json`，正式输出为 `work_dirs/instance_semantic_rank_rtmdet_tiny_dior_640_gpu1_bs36/`。检查汇总位于 `work_dirs/instance_semantic_rank_validation/checks.json`；复核入口为 `scripts/instance_semantic_rank_probe.py`，支持 `preflight`、`smoke`、`capacity`，`capacity --batch 12`仅调整独立压力检查。

## 实例语义监督：干净目标裁剪与多尺度训练

[configs/instance_semantic_multiscale/rtmdet_tiny_dior_640.py](configs/instance_semantic_multiscale/rtmdet_tiny_dior_640.py) 和 `custom_models/instance_semantic_multiscale/` 独立实现，继承旧版的从零训练、640主输入、官方train/val、真实batch36、不累积及300轮协议。后台队列等待LSConv检测头版成功完成，再做主图batch36、裁剪800×800、每图1200框的三次联合更新容量检查；检查通过后从零训练300轮，每10轮验证。冻结配置为 `work_dirs/instance_semantic_multiscale_queue/instance_semantic_multiscale_640_bs36.py`，输出为 `work_dirs/instance_semantic_multiscale_rtmdet_tiny_dior_640_gpu1_bs36/`。

借鉴 [MPSR](https://github.com/jiaxi-wu/MPSR) 的目标裁剪金字塔：在Mosaic/MixUp之前，每张主图随机选一个有效GT，以方形裁剪保留目标长宽比，越界填114；每次迭代均匀随机选择32/64/128/256/512/800中的一个尺度，全批裁剪统一缩放。主图仍按原流程增强和检测。随机单尺度是本项目的计算预算适配，没有同时执行原论文的全部尺度或额外RPN监督。

裁剪图像共用实际的骨干、neck和原分类塔；按缩放后的真实矩形GT面积选P3–P5，只执行所选层的分类塔，RoIAlign 3×3后平均，接原规格的96→20分类器，CE权重0.1。尺度选择和取样都使用GT坐标，填充正方形不充当GT。旧版的整图GT辅助损失被替代，不叠加；回归塔和检测预测层不在裁剪分支执行。总参数4,880,648，与旧实例语义版相同；推理卷积计算仍为7.9877 GMAC、8400个预测点。

裁剪BN使用自己的批统计与共享仿射参数，不修改主图的running mean/var或计数；单样本1×1退化情况使用主统计。裁剪先反传释放计算图，再计算主检测损失，`ObjectCropAmpOptimWrapper`只在主损失反传后执行一次参数更新、AMP缩放更新和迭代计数。每次迭代EMA更新一次，尺度采样状态随模型保存，并经过实际EMA检查点交换/恢复核对。该训练接口支持单GPU。

检查入口为 `scripts/instance_semantic_multiscale_probe.py`，报告位于 `work_dirs/instance_semantic_multiscale_validation/`。11项测试覆盖裁剪像素/坐标、强弱增强主路径逐项一致、初始化与推理一致、BN隔离、辅助梯度范围、分两次反传与合并损失的一次AdamW更新等价、尺度状态恢复。GPU AMP梯度及更新对照完全一致；官方入口真实batch4完成24次更新、两次32图VOC/COCO验证，覆盖六个尺度、两次EMA交换及模型/优化器/AMP恢复，无跳步。batch2在32像素裁剪时出现BN梯度放大及AMP溢出，短测改用batch4；正式配置保持batch36。短测不保存检查点，不用于判断涨点。

GPU0的主图batch12、每图1200框、裁剪800×800通过三次联合更新，峰值张量11.81 GiB；单独的36张裁剪覆盖六个尺度及三次800×800更新，峰值8.01 GiB。裁剪容量检查不等于主图batch36的联合压力检查；后者待独占GPU容量充足时执行，正式batch36配置没有改变。复核可运行 `python -m unittest tests.test_instance_semantic_multiscale -v`；探针支持`preflight`、`equivalence`、`smoke`、`capacity`、`crop_capacity`，其中`capacity`默认batch12，`--batch 36`显式检查正式batch。

## FreqFusion：CSP融合后注入高频残差

[configs/freq_fusion_residual/rtmdet_tiny_dior_640.py](configs/freq_fusion_residual/rtmdet_tiny_dior_640.py) 导入独立的 `custom_models.freq_fusion_residual`，不修改原FreqFusion。640、从零训练、官方train/val、300轮、每10轮验证、真实batch36及优化协议均与原FreqFusion一致。已完成300轮，VOC mAP76.29%、COCO AP56.9%。冻结配置为 `work_dirs/freq_fusion_residual_queue/freq_fusion_residual_640_bs36.py`，输出为 `work_dirs/freq_fusion_residual_rtmdet_tiny_dior_640_gpu1_bs36/`。

原接法为 `F(concat(U, H + D))`；新版为 `F(concat(U, H)) + P(D)`。`U`沿用完整低通与偏移重采样，`D`为FP32中直接计算的 `H - smooth(H)`，增强初始融合保留；该算子等价于作者的 `hr_residual=False` 设置，结构改动在外部CSP路径。`F`包含整个原top-down块，第一处还包括原192→96投影；`P`为独立、无偏置、无BN/激活的1×1卷积，分别192→96@40×40、96→96@80×80。残差只注入一次，融合结果继续参与下一层top-down和原PAN侧向连接。

总参数5,015,448，Conv/Linear 8.285956 GMAC；比原FreqFusion增加27,648参数、0.088474 GMAC，统计不含动态滤波、重采样、归一化和后处理。6项测试验证独立空间滤波、符号与梯度、完整块后的实际接线、原有初始化与随机状态、异常恢复及旁路精确退化为基线。6组GPU FP32/AMP与固定作者源码对照，前向误差0、最大梯度误差7.63e-6；大恒等项与小残差的用例确认没有从已舍入的 `H+D` 反推细节。

GPU0官方入口短测覆盖强弱增强、16次实际更新、两次32图验证、原始/EMA预测及模型/优化器/AMP状态读写；42个相关参数张量均有有限非零梯度并更新，无AMP跳步，不保存检查点。另以batch12、每图1200框完成三次压力更新，峰值张量12.43 GiB、保留14.06 GiB。GPU1正式batch36、每图1200框检查通过，峰值张量37.13 GiB、保留41.69 GiB。短测及压力检查关闭warmup、loss scale32，仅验证执行与容量；正式配置不变。汇总位于 `work_dirs/freq_fusion_residual_validation/checks.json`。

```bash
PYTHONPATH=/home/pwt/Paper1 python scripts/freq_fusion_residual_probe.py \
  --mode preflight --report work_dirs/freq_fusion_residual_validation/preflight.json
```

同一脚本提供`fidelity`、`smoke`、`capacity`；GPU检查前核对空闲显存并设置`CUDA_VISIBLE_DEVICES`。`capacity --batch 12`仅为独立小batch压力检查，不修改正式batch；省略`--batch`检查正式batch36。

## SCDown：三处深层降采样

[configs/scdown/rtmdet_tiny_dior_640.py](configs/scdown/rtmdet_tiny_dior_640.py) 独立导入 `custom_models.scdown`，以原始RTMDet-tiny为对照，不叠加其他改进。仅替换 `backbone.stage3.0`、`backbone.stage4.0`、`neck.downsample_layers.1`；实际通道分别96→192、192→384、192→192。结构为1×1/BN/SiLU，再DW3×3、stride2、padding1及BN，末端不加激活。第一处PAN降采样、CSP块、三个输出3×3、检测塔与8400个预测点保留。

算子对照YOLOv10作者提交 `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`，归一化与初始化使用RTMDet设置。独立初始化处理保留其余层的原始参数和随机状态。正式协议沿用640、官方train/val、从零300轮、每10轮验证、真实batch36、不累积、lr0.0005625、warmup1000及原EMA。实际构建为3,854,964参数、卷积7.623901 GMAC，较基线减少20.98%和4.55%；MAC逐次统计共享检测塔的实际执行，不包含归一化、激活或后处理。

9项测试覆盖三处接线、奇数尺寸和采样中心、无末端激活、初始化与随机状态、恢复原三层后整模型预测/损失/梯度完全一致、真实梯度更新、状态读写及初始化异常恢复。24组作者源码GPU对照覆盖三种通道、实际/奇数尺寸、训练/评估和FP32/AMP，前向及梯度最大误差均为0。官方入口真实数据完成16次实际更新、两次32图VOC/COCO验证、EMA交换和模型/优化器/AMP状态读写，18个相关参数张量均有有限非零梯度并更新，无AMP跳步。

GPU0 batch12、每图1200框的三次压力更新通过，峰值张量11.88 GiB、保留13.55 GiB；GPU1正式batch36、每图1200框也已通过三次更新，峰值张量35.55 GiB、保留40.24 GiB，分配器上限88%。短测关闭warmup、loss scale32，不保留检查点，不代表精度收益。

SCDown已在实例类别排序监督成功完成后自动接续并完成300轮。EMA的VOC mAP75.87%、COCO AP56.5%、AP75为61.9%，小/中/大目标AP为7.6/35.6/69.7；基线对应75.65%、56.4%、61.9%、7.5/34.9/69.8。VOC高0.22个百分点、COCO AP高0.1个百分点，第300轮也是本次两项最佳轮。单次结果为精度基本持平，明确收益是参数减少20.98%，所统计MAC减少4.55%。冻结配置为 `work_dirs/scdown_queue/scdown_640_bs36.py`，完成状态与进程身份位于同目录 `status.json`、`queue_identity.json`，训练输出为 `work_dirs/scdown_rtmdet_tiny_dior_640_gpu1_bs36/`，日志为 `logs/scdown_rtmdet_tiny_dior_640_gpu1_bs36.log`。

共享GPU0、batch1、640、AMP、随机权重前向计时：基线/SCDown未融合BN为10.77/10.33 ms，融合后8.29/8.20 ms。数值为三组各80次均值的中位数，预热20次，不含预处理和NMS；组间存在波动，融合后耗时基本相当。融合前先复制跨尺度共享卷积，再分别折叠各尺度BN，并用不同BN统计的用例核对输出，避免重复修改共享权重。此结果不代表正式训练速度或独占GPU部署延迟。

复核入口 `scripts/scdown_probe.py` 支持 `preflight`、`fidelity`、`smoke`、`capacity`、`latency`；作者对照使用 `--reference-dir` 指定固定提交的 `conv.py`、`block.py`。`capacity --batch 12`只调整独立压力检查，省略该参数检查正式batch36。汇总位于 `work_dirs/scdown_validation/checks.json`。

## SCDown实验1、2、3

三版以已完成的三处SCDown为结构对照，分别使用独立代码和配置；不叠加实例语义监督。640、真实batch36、从零300轮、每10轮验证、优化器与增强沿用对照，检测头仍为P3/P4/P5、8400个预测点。

| 实验 | 配置目录 | 相对三处版新增的改动 | 参数 | 卷积GMAC |
|---|---|---|---:|---:|
| 对照 | `configs/scdown/` | 无 | 3,854,964 | 7.623901 |
| 1 | `configs/scdown_g2/` | 七个CSP内部3×3、stride1改为groups=2 | 3,378,036 | 7.159415 |
| 2 | `configs/scdown_p3/` | 骨干P3入口48→96改为SCDown，共四处 | 3,819,156 | 7.481975 |
| 3 | `configs/scdown_p2p3/` | 骨干P3及P2入口改为SCDown，共五处 | 3,810,468 | 7.345578 |

实验1分组位置为骨干 `stage2.1`、`stage3.1`、`stage4.2`，以及neck的两处top-down和两处bottom-up CSP内部 `blocks.0.conv1`。通道48/96/192各分成连续的两组，每组24/48/96通道；保留原BN、SiLU、后续DW5×5和完整1×1通道混合。不额外加shuffle或投影，不改P2内部块、neck输出和检测塔。实验2、3只扩展降采样，P2入口为24→48，不新增P2预测尺度，也不包含实验1的G2改动。

实验1、2、3已完成300轮：EMA的VOC mAP/COCO AP分别为75.74%/56.4%、76.11%/56.7%、76.12%/56.4%；三处SCDown为75.87%/56.5%。实验2相对三处版仅新增P3入口替换，后期曲线支持小幅收益。实验3比实验2少8,688参数（0.23%）和1.82% MAC，VOC基本相同，COCO AP低0.3个百分点。

14项测试通过，覆盖原三处版及新增作用范围、分组与块对角稠密卷积的前向/梯度对照、奇数尺寸采样中心、初始化随机状态及还原新增替换后的完整预测/损失/梯度一致性。三版各自通过官方入口真实数据16次AMP更新、两次32图VOC/COCO验证、EMA交换及模型/优化器/scaler状态读写；分别核对39/24/30个参数张量有有限非零梯度并实际更新。

GPU0共享环境batch8、每图1200框、三次更新的压力峰值张量分别7.93/8.04/8.24 GiB，保留显存9.16/9.26/9.35 GiB；这不是正式batch36容量结论。正式GPU1检查将在各自前序成功完成后执行。短测关闭warmup并将loss scale设为32，仅用于核对实际更新；正式配置不变，不保留短测检查点。检查摘要在各自 `work_dirs/<方法>_validation/checks.json`，冻结配置与接续身份在 `work_dirs/<方法>_queue/`。

## SCDown的G4与标准DSConv对照

两版继承三处SCDown的640、官方train/val、真实batch36、不累积、从零300轮及每10轮验证协议，使用独立模块、配置和 `scripts/scdown_compression_probe.py`；已按G4→DSConv接在SCDown实验3后。替换位置与G2完全相同：骨干P3–P5的三个CSP及neck内四个CSP的 `blocks.0.conv1`；原三处SCDown、P2内部块、neck输出层与检测塔保留，仍为8400个预测点。

| 对照 | 配置 | 参数 | 卷积GMAC（640、batch1） |
|---|---|---:|---:|
| G4 | `configs/scdown_g4/rtmdet_tiny_dior_640.py` | 3,139,572 | 6.927172 |
| DSConv | `configs/scdown_dsconv/rtmdet_tiny_dior_640.py` | 3,015,540 | 6.809207 |

G4将48/96/192通道各分为四组，每组12/24/48通道，保持原BN和SiLU。DSConv使用MMCV的标准深度可分离卷积：DW3×3/BN/SiLU→PW1×1/BN/SiLU，沿用RTMDet归一化和激活；对应MobileNet的DW→PW次序，激活为SiLU而非原论文ReLU。原CSP的后续DW5×5→PW1×1仍保留，因此此分支成为DW3×3→PW1×1→DW5×5→PW1×1，各步骤均含原定的BN/SiLU。

G1指三处SCDown对照中这七处仍采用普通卷积，总参数3,854,964；G2/G4仅减少这些3×3的跨通道连接，后续原有1×1仍混合全部通道。G4已完成300轮，VOC mAP75.58%、COCO AP56.2%；DSConv仍在训练。

8项测试通过，覆盖作用范围、通道与共享关系、未改动参数及随机状态、G4与块对角稠密卷积的前向/梯度对照、DSConv与独立PyTorch参考的前向/梯度对照、还原七处后的整模型预测/损失/梯度一致性、实际检测梯度更新、单次BN更新及状态读写。两版各自通过官方入口16次真实数据AMP更新、两次32图VOC/COCO验证、EMA交换与模型/优化器/scaler读写，39/60个相关参数张量均有有限非零梯度并实际更新。GPU0共享环境、batch8、每图1200框的三次压力更新通过，G4/DSConv峰值张量7.93/7.98 GiB、保留显存9.16/9.12 GiB；正式GPU1 batch36容量检查将在各自前序成功完成后执行，通过后才启动训练。短测关闭warmup、loss scale设为32，不保留检查点；正式配置保持原调度。检查入口支持 `preflight`、`smoke`、`capacity`；摘要及源码校验值在各自 `work_dirs/<方法>_validation/checks.json`。

## G8/G16分组数对照

两版沿用三处SCDown和原G2/G4的七处CSP `conv1`，分别使用独立的 `custom_models/scdown_g8/`、`custom_models/scdown_g16/` 与同名配置目录，只改变分组数，保留各层原BN/SiLU和后续DW5×5/PW1×1。骨干P3入口仍为原密集卷积，未叠加实验2的第四处SCDown。

| 分组 | 48/96/192通道的每组宽度 | 参数 | 卷积GMAC，640/batch1 | 相对原始基线减参/MAC |
|---|---|---:|---:|---|
| G8 | 6/12/24 | 3,020,340 | 6.811050 | 38.09%/14.73% |
| G16 | 3/6/12 | 2,960,724 | 6.752989 | 39.31%/15.46% |

6项测试通过，覆盖实际范围与通道、原有协议、初始化随机状态、恢复七处后的整模型预测/损失/梯度一致性、分组与块对角密集卷积前后向对照，以及参数更新、单次BN更新和状态读写。GPU0两版各完成16次官方AMP更新、两次32图验证与EMA/优化器/scaler检查，39个相关参数张量均更新，无AMP跳步；batch8、每图1200框的三次压力更新均通过，峰值张量约7.93 GiB、保留9.16 GiB。

队列为P3对照B → G8 → G16 → 四处版＋G2。仅重新安排尚未启动的组合实验，保留DSConv训练及A/B进程。GPU1正式启动前分别检查真实batch36容量；报告位于同名 `work_dirs/*_validation/checks.json`，复核入口 `scripts/scdown_group_sweep_probe.py`。四处版＋G4仅完成CPU结构测试，未安排GPU检查和正式训练。

## 四处SCDown＋G2

[configs/scdown_p3_g2/rtmdet_tiny_dior_640.py](configs/scdown_p3_g2/rtmdet_tiny_dior_640.py) 以实验2的四处SCDown为基础，增加与实验1完全相同的七处CSP `conv1`分组，两组连续通道，原BN/SiLU及后续卷积保留。独立骨干继承四处版，neck直接复用已验证的G2版；P2入口、neck输出、检测头、8400个预测点、数据和训练协议保持不变。参数3,342,228、640/batch1卷积7.017489 GMAC，比原始基线减少31.49%/12.15%，比实验2减少12.49%/6.21%。与实验2比较G2在四处版上的效果；结合三处版与原G2结果，检查P3替换后的组合收益。

5项测试通过，核对范围/共享关系、两组对照协议、初始化与随机状态、分别还原P3或G2后的整模型预测/损失/梯度一致性，以及实际优化器更新、单次BN更新和状态读写。GPU0官方入口完成16次AMP更新、两次32图验证、EMA交换与模型/优化器/scaler读写，45个相关参数张量均有有限非零梯度并更新，无AMP跳步。另以batch8、每图1200框完成三次压力更新，峰值张量8.03 GiB、保留9.26 GiB；正式batch36未在共享GPU0上测试。

已接在G16后。正式GPU1启动前等待空闲至少46000 MiB，以batch36、不累积、每图1200框和88%分配器上限检查三次更新，通过后才开始300轮训练。检查入口 `scripts/scdown_p3_g2_probe.py`；报告、冻结配置及队列状态分别在同名 `work_dirs/*_validation/`、`work_dirs/*_queue/`，前序或检查失败停止接续。

## P3降采样的两组机理对照

两版继承实验2 `scdown_p3` 的640、官方train/val、真实batch36、不累积、从零300轮及每10轮验证协议，仅改变 `backbone.stage2.0`，其余三处SCDown、neck、检测头、8400个预测点与损失保持一致；独立模块和配置，已按A→B接在DSConv之后。冻结配置、进程身份与状态位于各自 `work_dirs/<方法>_queue/`。

| 对照 | 配置目录 | P3入口结构 | 参数 | 卷积GMAC |
|---|---|---|---:|---:|
| 实验2 | `configs/scdown_p3/` | PW48→96/BN/SiLU → DW3×3 stride2/BN | 3,819,156 | 7.481975 |
| A | `configs/scdown_p3_dense/` | 保留上述流程，只把DW换成96→96的密集3×3 | 3,901,236 | 8.007287 |
| B | `configs/scdown_p3_noact/` | 保留两层卷积和BN，只去掉PW后的SiLU | 3,819,156 | 7.481975 |

A检验空间通道连接的约束，B检验降采样前非线性的作用；B不把SiLU移到末端。PW在160×160上输出96通道，空间卷积以padding1、stride2输出80×80。A仅空间卷积权重改变形状，其余初始权重及BN状态与实验2一致；B全部初始权重与实验2一致。顺序反转对照C尚未实现。

8项测试通过：包括与独立PyTorch参考的前向/梯度一致性、密集卷积对角形式复现DW的前向/梯度、还原各自改动后整模型预测/损失/梯度完全一致、作用范围与初始化随机状态、实际检测梯度更新、BN执行次数、共享关系和状态读写。GPU0官方入口两版各完成16次AMP更新、两次32图VOC/COCO验证、EMA交换及模型/优化器/scaler读写，24个相关参数张量均有有限非零梯度并更新，无AMP跳步。batch8、每图1200框的三次压力更新也已通过，A/B峰值张量8.04/8.00 GiB、保留显存9.27/9.22 GiB；各自前序成功完成后，在GPU1先执行正式batch36容量检查，通过后才启动训练。短测关闭warmup、loss scale32，不保留检查点，正式配置不变。检查入口为 `scripts/scdown_p3_controls_probe.py`，支持 `preflight`、`smoke`、`capacity`；摘要与源码校验值在各自 `work_dirs/<方法>_validation/checks.json`。

## 检查

模型检查结果见各方法`work_dirs/*_validation/checks.json`；EFS适配与LSConv扩展已完成真实GPU训练/验证检查。`tests/test_lsconv_full.py`检查扩展范围、尺度共享和梯度、检查点及训练协议；`tests/test_efs_adapt.py`核对EFS两项适配的结构、梯度和EMA。队列六项测试通过，覆盖进程身份、重复启动、成功/失败接续，以及显式断点恢复、日志保留和无效检查点拒绝。`tests/test_fdw.py`覆盖频谱重建、梯度、FP32/AMP、参数更新、初始权重和检查点；`tests/test_lsconv.py`覆盖完整LSConv及Triton前后向，CUDA测试需设置 `PAPER1_RUN_CUDA_TESTS=1`。其他候选见各自测试文件。

GPU检查摘要位于 `work_dirs/efs_validation/checks.json`：每图1200个重复框、12次微迭代与3次实际优化器更新，峰值张量显存32.31 GiB、保留显存36.85 GiB，检查分配器上限为可用容量的88%。真实图像短测覆盖强弱增强、梯度累积、25个P2参数张量更新及两次32图VOC/COCO评估。短测不保存或复用检查点，不用于判断精度收益。压力测试仅模拟匹配显存，不代表真实训练速度；短测降低AMP初始缩放并关闭warmup以检验更新，正式训练保留原始AMP动态缩放与warmup。

LSConv摘要为 `work_dirs/lsconv_validation/checks.json`。CPU完整算子的三种宽度与官方实现输出及梯度一致；GPU短测使用16图、batch2、两轮，共16次实际优化器更新，45个新增参数张量均收到有限非零梯度并更新；强弱增强及两次32图评估通过。batch36压力检查已通过：每图1200个重复框、三次实际更新，峰值张量35.42 GiB、保留40.09 GiB。小验证子集缺少某类某尺度GT时，该类别尺度AP可为NaN，整体指标和预测均有限。

## 其他独立模块

| 目录 | 模块及借鉴来源 |
|---|---|
| `custom_models/state_guided/` | 压缩状态上下文（EfficientViM/HSM）与9邻域局部上采样（UPLiFT） |
| `custom_models/magnitude_detail/` | 幅值线性上下文（MALA）与C2细节注入；最终保持三尺度预测 |
| `custom_models/frequency_guided/` | 特征频带选择（FDConv/FBM）与频带条件局部上采样 |

这些是机制适配，不是原论文完整复现。模块文件各自独立，`neck.py`只组装所属方法；同名配置目录包含单模块、组合及对应机制对照。它们的源配置默认batch12；已有完整对照的实际运行配置快照在各自输出目录。当前不安排这些方法继续训练。
