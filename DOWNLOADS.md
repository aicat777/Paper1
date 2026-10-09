# 代码、论文与 DIOR 数据

本地目录 `/home/pwt/Paper1`，下载日期 2026-10-09。论文、官方代码、DIOR 全量图像、水平框标注和原始划分已就绪。训练从零初始化，不需要下载预训练模型。

## 论文与代码

| 内容 | 本地目录 | 固定来源 / 版本 |
| --- | --- | --- |
| RTMDet 论文，arXiv v2，15 页 | `papers/RTMDet_2022.pdf` | https://arxiv.org/abs/2212.07784 |
| MMYOLO | `mmyolo/` | https://github.com/open-mmlab/mmyolo ，`8c4d9dc503dc8e327bec8147e8dc97124052f693` |
| MMDetection | `mmdetection/` | https://github.com/open-mmlab/mmdetection ，`cfd5d3a985b0249de009b67d04f37263e11cdf3d` |

当前 DIOR 配置继承官方 tiny 的网络与训练方案，并覆盖数据、类别、单卡 batch、学习率、验证和初始化。官方上层配置的 ImageNet 骨干初始化已关闭，整模型也不加载 COCO 检测权重。

## DIOR 来源

作者主页：https://gcheng-nwpu.github.io/#Datasets 。许可证：CC BY-NC 4.0。

下载镜像：https://huggingface.co/datasets/ObjEarth/ObjEarth-Data/tree/f58d220a2b74b610badebaefa122d1b61628bf73/DIOR 。该镜像不是作者主页的下载站点。

固定 revision：`f58d220a2b74b610badebaefa122d1b61628bf73`。图像压缩包通过同 revision 的 `hf-mirror.com` 直连加速传输，限流后回退至原站。完整压缩包均核对 Hugging Face LFS SHA256 后解压；压缩包和临时安装包已清理。

| 原始划分 | 图像数 |
| --- | ---: |
| train | 5,862 |
| val | 5,863 |
| test | 11,738 |

未重新随机划分。原始列表位于 `datasets/DIOR/ImageSets/Main/`，版本控制副本位于 `data_metadata/DIOR/splits/`。

标注实读为 23,463 个 XML、20 类、192,472 个 object；作者主页写的是 192,518 个实例。保留原始标注，没有补改实例。

| 压缩包 | 大小（字节） | SHA256 |
| --- | ---: | --- |
| Annotations.zip | 10,781,498 | `4e9a007c768fd5aa1168894ae1654ddef54626041d8219f9403e389fda5d2afa` |
| JPEGImages-test.zip | 3,528,597,945 | `9a91b6eb8af70fd1c8052564f4bd504c26b8be14d587fa1fddd1e741a43cf4c7` |
| JPEGImages-trainval.zip | 3,887,687,878 | `f93f4b13ca0b0d235d96d23317a626bb4c03221c96f90b46953a6396dc2e28e0` |

下载和校验脚本为 `scripts/download_dior.py`、`scripts/verify_dior.py`；来源元数据和核对报告保存在 `datasets/DIOR/`，副本位于 `data_metadata/DIOR/`。

## COCO JSON 转换

运行 `python scripts/convert_dior_to_coco.py` 生成 `datasets/DIOR/coco_annotations/`。COCO JSON 仅是标注格式，数据仍全部来自 DIOR。

原始 XML 保留，按原始划分生成 train / val / test JSON。坐标遵循 MMDetection 的 VOC 转换工具：四个 XYXY 坐标均减 1，再转换为 XYWH；零面积框剔除 train 1 个、val 3 个、test 3 个。类别名称统一小写，图像尺寸从实际 JPEG 读取，详情见 `conversion_report.json`。

DIOR 引用：Ke Li, Gang Wan, Gong Cheng, Liqiu Meng, Junwei Han. *Object detection in optical remote sensing images: a survey and a new benchmark*. ISPRS Journal of Photogrammetry and Remote Sensing, 159: 296–307, 2020。
