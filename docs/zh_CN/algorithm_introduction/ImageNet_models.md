<!-- 简体中文 | [English](../../en/algorithm_introduction/ImageNet_models.md) -->


<<<<<<< HEAD
## ImageNet预训练模型库

### 目录

- [模型库概览图](#模型库概览图)
- [SSLD知识蒸馏预训练模型](#模型库概览图)
    - [服务器端知识蒸馏模型](#服务器端知识蒸馏模型)
    - [移动端知识蒸馏模型](#移动端知识蒸馏模型)
    - [Intel CPU端知识蒸馏模型](#Intel-CPU端知识蒸馏模型)
- [PP-LCNet系列](#PP-LCNet系列)
- [ResNet系列](#ResNet系列)
- [移动端系列](#移动端系列)
- [SEResNeXt与Res2Net系列](#SEResNeXt与Res2Net系列)
- [DPN与DenseNet系列](#DPN与DenseNet系列)
- [HRNet系列](#HRNet系列)
- [Inception系列](#Inception系列)
- [EfficientNet与ResNeXt101_wsl系列](#EfficientNet与ResNeXt101_wsl系列)
- [ResNeSt与RegNet系列](#ResNeSt与RegNet系列)
- [ViT_and_DeiT系列](#ViT_and_DeiT系列)
- [RepVGG系列](#RepVGG系列)
- [MixNet系列](#MixNet系列)
- [SwinTransformer系列](#SwinTransformer系列)
- [LeViT系列](#LeViT系列)
- [HarDNet系列](#HarDNet系列)
- [DLA系列](#DLA系列)
- [RedNet系列](#RedNet系列)
- [TNT系列](#TNT系列)
- [其他模型](#其他模型)


<a name="模型库概览图"></a>
### 模型库概览图

基于ImageNet1k分类数据集，PaddleClas支持37个系列分类网络结构以及对应的217个图像分类预训练模型，训练技巧、每个系列网络结构的简单介绍和性能评估将在相应章节展现，下面所有的速度指标评估环境如下：
* Arm CPU的评估环境基于骁龙855（SD855）。
* Intel CPU的评估环境基于Intel(R) Xeon(R) Gold 6148。
* GPU评估环境基于T4机器，在FP32+TensorRT配置下运行500次测得（去除前10次的warmup时间）。
* FLOPs与Params通过`paddle.flops()`计算得到（PaddlePaddle版本为2.2）
# ImageNet 预训练模型库

## 目录

- [1. 模型库概览图](#1)
- [2. SSLD 知识蒸馏预训练模型](#2)
  - [2.1 服务器端知识蒸馏模型](#2.1)
  - [2.2 移动端知识蒸馏模型](#2.2)
  - [2.3 Intel CPU 端知识蒸馏模型](#2.3)
- [3. PP-LCNet 系列](#3)
- [4. ResNet 系列](#4)
- [5. 移动端系列](#5)
- [6. SEResNeXt 与 Res2Net 系列](#6)
- [7. DPN 与 DenseNet 系列](#7)
- [8. HRNet 系列](#8)
- [9. Inception 系列](#9)
- [10. EfficientNet 与 ResNeXt101_wsl 系列](#10)
- [11. ResNeSt 与 RegNet 系列](#11)
- [12. ViT_and_DeiT 系列](#12)
- [13. RepVGG 系列](#13)
- [14. MixNet 系列](#14)
- [15. ReXNet 系列](#15)
- [16. SwinTransformer 系列](#16)
- [17. LeViT 系列](#17)
- [18. Twins 系列](#18)
- [19. HarDNet 系列](#19)
- [20. DLA 系列](#20)
- [21. RedNet 系列](#21)
- [22. TNT 系列](#22)
- [23. 其他模型](#23)

<a name="1"></a>

## 1. 模型库概览图

基于 ImageNet1k 分类数据集，PaddleClas 支持 37 个系列分类网络结构以及对应的 217 个图像分类预训练模型，训练技巧、每个系列网络结构的简单介绍和性能评估将在相应章节展现，下面所有的速度指标评估环境如下：
* Arm CPU 的评估环境基于骁龙 855(SD855)。
* Intel CPU 的评估环境基于 Intel(R) Xeon(R) Gold 6148。
* GPU 评估环境基于 T4 机器，在 FP32+TensorRT 配置下运行 500 次测得（去除前 10 次的 warmup 时间）。
* FLOPs 与 Params 通过 `paddle.flops()` 计算得到（PaddlePaddle 版本为 2.2）

常见服务器端模型的精度指标与其预测耗时的变化曲线如下图所示。

![](../../images/models/T4_benchmark/t4.fp32.bs1.main_fps_top1.png)


常见移动端模型的精度指标与其预测耗时、模型存储大小的变化曲线如下图所示。

![](../../images/models/mobile_arm_storage.png)

![](../../images/models/mobile_arm_top1.png)

<a name="2"></a>

## 2. SSLD 知识蒸馏预训练模型
基于 SSLD 知识蒸馏的预训练模型列表如下所示，更多关于 SSLD 知识蒸馏方案的介绍可以参考：[SSLD 知识蒸馏文档](./knowledge_distillation.md)。

<a name="2.1"></a>

### 2.1 服务器端知识蒸馏模型

| 模型                  | Top-1 Acc | Reference<br>Top-1 Acc | Acc gain | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                         |
|---------------------|-----------|-----------|---------------|----------------|-----------|----------|-----------|-----------------------------------|
| ResNet34_vd_ssld         | 0.797    | 0.760  | 0.037  | 2.434               | 6.222              | 3.93     | 21.84     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_ssld_pretrained.pdparams)         |
| ResNet50_vd_ssld | 0.830    | 0.792    | 0.039 | 3.531               | 8.090              | 4.35     | 25.63     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_ssld_pretrained.pdparams) |
| ResNet101_vd_ssld   | 0.837    | 0.802    | 0.035 |  6.117               | 13.762             | 8.08     | 44.67     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_ssld_pretrained.pdparams)   |
| Res2Net50_vd_26w_4s_ssld | 0.831    | 0.798    | 0.033 |  4.527              | 9.657             | 4.28     | 25.76     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_ssld_pretrained.pdparams) |
| Res2Net101_vd_<br>26w_4s_ssld | 0.839    | 0.806    | 0.033 | 8.087              | 17.312             | 8.35    | 45.35     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_ssld_pretrained.pdparams) |
| Res2Net200_vd_<br>26w_4s_ssld | 0.851    | 0.812    | 0.049 | 14.678              | 32.350             | 15.77    | 76.44     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_ssld_pretrained.pdparams) |
| HRNet_W18_C_ssld | 0.812    | 0.769   | 0.043 | 7.406          | 13.297         | 4.32     | 21.35     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_ssld_pretrained.pdparams) |
| HRNet_W48_C_ssld | 0.836    | 0.790   | 0.046  | 13.707         | 17.34         | 17.34    | 77.57     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_ssld_pretrained.pdparams) |
| SE_HRNet_W64_C_ssld | 0.848    |  -    |  - |  31.697      |     94.995      | 29.00    | 129.12    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SE_HRNet_W64_C_ssld_pretrained.pdparams) |

<a name="2.2"></a>

### 2.2 移动端知识蒸馏模型

| 模型                  | Top-1 Acc | Reference<br>Top-1 Acc | Acc gain | SD855 time(ms)<br>bs=1 | Flops(G) | Params(M) | 模型大小(M) | 下载地址   |
|---------------------|-----------|-----------|---------------|----------------|-----------|----------|-----------|-----------------------------------|
| MobileNetV1_ssld   | 0.779    | 0.710    | 0.069 |  32.523              | 578.88     | 4.25      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_ssld_pretrained.pdparams)                 |
| MobileNetV2_ssld                 | 0.767    | 0.722  | 0.045  | 23.318              | 327.84      | 3.54      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams)                 |
| MobileNetV3_small_x0_35_ssld          | 0.556    | 0.530 | 0.026   | 2.635                 | 14.56    | 1.67      | 6.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_ssld_pretrained.pdparams)          |
| MobileNetV3_large_x1_0_ssld      | 0.790    | 0.753  | 0.036  | 19.308           | 229.66     | 5.50      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)      |
| MobileNetV3_small_x1_0_ssld      | 0.713    | 0.682  |  0.031  | 6.546                 | 63.67    | 2.95      | 12      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)      |
| GhostNet_x1_3_ssld                    | 0.794    | 0.757   | 0.037 | 19.983                | 236.89     | 7.38       | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_ssld_pretrained.pdparams)               |


<a name="Intel-CPU端知识蒸馏模型"></a>
#### Intel CPU端知识蒸馏模型
<a name="2.3"></a>

### 2.3 Intel CPU 端知识蒸馏模型

| 模型                  | Top-1 Acc | Reference<br>Top-1 Acc | Acc gain |  Intel-Xeon-Gold-6148 time(ms)<br>bs=1 | FLOPs(M) | Params(M)  | 下载地址   |
|---------------------|-----------|-----------|---------------|----------------|----------|-----------|-----------------------------------|
| PPLCNet_x0_5_ssld   | 0.661    | 0.631    | 0.030 | 2.05     | 47.28     |   1.89   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_ssld_pretrained.pdparams)                 |
| PPLCNet_x1_0_ssld   | 0.744    | 0.713    | 0.033 | 2.46     | 160.81     |   2.96  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_ssld_pretrained.pdparams)                 |
| PPLCNet_x2_5_ssld   | 0.808    | 0.766    | 0.042 | 5.39     | 906.49     |   9.04  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_ssld_pretrained.pdparams)                 |




* 注: `Reference Top-1 Acc` 表示 PaddleClas 基于 ImageNet1k 数据集训练得到的预训练模型精度。

<a name="3"></a>

## 3. PP-LCNet 系列

PP-LCNet 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[PP-LCNet 系列模型文档](../models/PP-LCNet.md)。

| 模型           | Top-1 Acc | Top-5 Acc | Intel-Xeon-Gold-6148 time(ms)<br>bs=1 | FLOPs(M) | Params(M) | 下载地址 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| PPLCNet_x0_25        |0.5186           | 0.7565   |  1.74      | 18.25    | 1.52  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams) |
| PPLCNet_x0_35        |0.5809           | 0.8083   |  1.92      | 29.46    | 1.65  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams) |
| PPLCNet_x0_5         |0.6314           | 0.8466   |  2.05      | 47.28    | 1.89  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams) |
| PPLCNet_x0_75        |0.6818           | 0.8830   |  2.29      | 98.82    | 2.37  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams) |
| PPLCNet_x1_0         |0.7132           | 0.9003   |  2.46      | 160.81   | 2.96  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams) |
| PPLCNet_x1_5         |0.7371           | 0.9153   |  3.19      | 341.86   | 4.52  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams) |
| PPLCNet_x2_0         |0.7518           | 0.9227   |  4.27      | 590   | 6.54  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams) |
| PPLCNet_x2_5         |0.7660           | 0.9300   |  5.39      | 906   | 9.04  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams) |

<a name="4"></a>

## 4. ResNet 系列

ResNet 及其 Vd 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[ResNet 及其 Vd 系列模型文档](../models/ResNet_and_vd.md)。

| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                         |
|---------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------|
| ResNet18            | 0.7098    | 0.8992    | 1.45606               | 3.56305              | 1.83     | 11.70     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_pretrained.pdparams)            |
| ResNet18_vd         | 0.7226    | 0.9080    | 1.54557               | 3.85363              | 2.07     | 11.72     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_vd_pretrained.pdparams)         |
| ResNet34            | 0.7457    | 0.9214    | 2.34957               | 5.89821              | 3.68     | 21.81     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_pretrained.pdparams)            |
| ResNet34_vd         | 0.7598    | 0.9298    | 2.43427               | 6.22257              | 3.93     | 21.84     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_pretrained.pdparams)         |
| ResNet34_vd_ssld         | 0.7972    | 0.9490    | 2.43427               | 6.22257              | 3.93     | 21.84     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_ssld_pretrained.pdparams)         |
| ResNet50            | 0.7650    | 0.9300    | 3.47712               | 7.84421              | 4.11     | 25.61     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_pretrained.pdparams)            |
| ResNet50_vc         | 0.7835    | 0.9403    | 3.52346               | 8.10725              | 4.35     | 25.63     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vc_pretrained.pdparams)         |
| ResNet50_vd         | 0.7912    | 0.9444    | 3.53131               | 8.09057              | 4.35     | 25.63     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams)         |
| ResNet101           | 0.7756    | 0.9364    | 6.07125               | 13.40573             | 7.83    | 44.65     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_pretrained.pdparams)           |
| ResNet101_vd        | 0.8017    | 0.9497    | 6.11704               | 13.76222             | 8.08     | 44.67     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_pretrained.pdparams)        |
| ResNet152           | 0.7826    | 0.9396    | 8.50198               | 19.17073             | 11.56    | 60.34     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_pretrained.pdparams)           |
| ResNet152_vd        | 0.8059    | 0.9530    | 8.54376               | 19.52157             | 11.80    | 60.36     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_vd_pretrained.pdparams)        |
| ResNet200_vd        | 0.8093    | 0.9533    | 10.80619              | 25.01731             | 15.30    | 74.93     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet200_vd_pretrained.pdparams)        |
| ResNet50_vd_<br>ssld | 0.8300    | 0.9640    | 3.53131               | 8.09057              | 4.35     | 25.63     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_ssld_pretrained.pdparams) |
| ResNet101_vd_<br>ssld   | 0.8373    | 0.9669    | 6.11704               | 13.76222             | 8.08     | 44.67     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_ssld_pretrained.pdparams)   |

<a name="5"></a>

## 5. 移动端系列

移动端系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[移动端系列模型文档](../models/Mobile.md)。

| 模型          | Top-1 Acc | Top-5 Acc | SD855 time(ms)<br>bs=1 | FLOPs(M) | Params(M) | 模型大小(M) | 下载地址   |
|----------------------------------|-----------|-----------|------------------------|----------|-----------|---------|-----------------------------------------------------------------------------------------------------------|
| MobileNetV1_<br>x0_25                | 0.5143    | 0.7546    | 3.21985                | 43.56     | 0.48      | 1.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_25_pretrained.pdparams)                |
| MobileNetV1_<br>x0_5                 | 0.6352    | 0.8473    | 9.579599               | 154.57     | 1.34      | 5.2     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_5_pretrained.pdparams)                 |
| MobileNetV1_<br>x0_75                | 0.6881    | 0.8823    | 19.436399              | 333.00     | 2.60      | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_75_pretrained.pdparams)                |
| MobileNetV1                      | 0.7099    | 0.8968    | 32.523048              | 578.88     | 4.25      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_pretrained.pdparams)                      |
| MobileNetV1_<br>ssld                 | 0.7789    | 0.9394    | 32.523048              | 578.88     | 4.25      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_ssld_pretrained.pdparams)                 |
| MobileNetV2_<br>x0_25                | 0.5321    | 0.7652    | 3.79925                | 34.18     | 1.53       | 6.1     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams)                |
| MobileNetV2_<br>x0_5                 | 0.6503    | 0.8572    | 8.7021                 | 99.48     | 1.98      | 7.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x0_75                | 0.6983    | 0.8901    | 15.531351              | 197.37     | 2.65      | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams)                |
| MobileNetV2                      | 0.7215    | 0.9065    | 23.317699              | 327.84      | 3.54      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams)                      |
| MobileNetV2_<br>x1_5                 | 0.7412    | 0.9167    | 45.623848              | 702.35     | 6.90      | 26      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x2_0                 | 0.7523    | 0.9258    | 74.291649              | 1217.25     | 11.33     | 43      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams)                 |
| MobileNetV2_<br>ssld                 | 0.7674    | 0.9339    | 23.317699              | 327.84      | 3.54      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams)                 |
| MobileNetV3_<br>large_x1_25          | 0.7641    | 0.9295    | 28.217701              | 362.70    | 7.47      | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0           | 0.7532    | 0.9231    | 19.30835               | 229.66     | 5.50      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_75          | 0.7314    | 0.9108    | 13.5646                | 151.70    | 3.93      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>large_x0_5           | 0.6924    | 0.8852    | 7.49315                | 71.83    | 2.69      | 11      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_35          | 0.6432    | 0.8546    | 5.13695                | 40.90    | 2.11       | 8.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_25          | 0.7067    | 0.8951    | 9.2745                 | 100.07    | 3.64      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_0           | 0.6824    | 0.8806    | 6.5463                 | 63.67    | 2.95      | 12      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_75          | 0.6602    | 0.8633    | 5.28435                | 46.02    | 2.38      | 9.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_5           | 0.5921    | 0.8152    | 3.35165                | 22.60    | 1.91       | 7.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_35          | 0.5303    | 0.7637    | 2.6352                 | 14.56    | 1.67      | 6.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_35_ssld          | 0.5555    | 0.7771    | 2.6352                 | 14.56    | 1.67      | 6.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_ssld_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0_ssld      | 0.7896    | 0.9448    | 19.30835               | 229.66     | 5.50      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)      |
| MobileNetV3_small_<br>x1_0_ssld      | 0.7129    | 0.9010    | 6.5463                 | 63.67    | 2.95      | 12      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)      |
| ShuffleNetV2                     | 0.6880    | 0.8845    | 10.941                 | 148.86     | 2.29      | 9       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams)                     |
| ShuffleNetV2_<br>x0_25               | 0.4990    | 0.7379    | 2.329                  | 18.95     | 0.61       | 2.7     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_33               | 0.5373    | 0.7705    | 2.64335                | 24.04     | 0.65      | 2.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_5                | 0.6032    | 0.8226    | 4.2613                 | 42.58     | 1.37      | 5.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x1_5                | 0.7163    | 0.9015    | 19.3522                | 301.35     | 3.53      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x2_0                | 0.7315    | 0.9120    | 34.770149              | 571.70     | 7.40      | 28      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams)                |
| ShuffleNetV2_<br>swish               | 0.7003    | 0.8917    | 16.023151              | 148.86     | 2.29      | 9.1     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams)               |
| GhostNet_<br>x0_5                    | 0.6688    | 0.8695    | 5.7143                 | 46.15    | 2.60       | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams)               |
| GhostNet_<br>x1_0                    | 0.7402    | 0.9165    | 13.5587                | 148.78    | 5.21       | 20      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams)               |
| GhostNet_<br>x1_3                    | 0.7579    | 0.9254    | 19.9825                | 236.89     | 7.38       | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams)               |
| GhostNet_<br>x1_3_ssld                    | 0.7938    | 0.9449    | 19.9825                | 236.89     | 7.38       | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_ssld_pretrained.pdparams)               |
| ESNet_x0_25 | 62.48 | 83.46 || 30.85 | 2.83 | 11 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_25_pretrained.pdparams) |
| ESNet_x0_5 | 68.82 | 88.04 || 67.31 | 3.25 | 13 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_5_pretrained.pdparams)               |
| ESNet_x0_75 | 72.24 | 90.45 || 123.74 | 3.87 | 15 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x0_75_pretrained.pdparams)               |
| ESNet_x1_0 | 73.92 | 91.40 || 197.33 | 4.64 | 18 |[下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ESNet_x1_0_pretrained.pdparams)               |

<a name="6"></a>

## 6. SEResNeXt 与 Res2Net 系列

SEResNeXt 与 Res2Net 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[SEResNeXt 与 Res2Net 系列模型文档](../models/SEResNext_and_Res2Net.md)。


| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                         |
|---------------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| Res2Net50_<br>26w_4s          | 0.7933    | 0.9457    | 4.47188               | 9.65722              | 4.28     | 25.76      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_26w_4s_pretrained.pdparams)          |
| Res2Net50_vd_<br>26w_4s       | 0.7975    | 0.9491    | 4.52712               | 9.93247              | 4.52     | 25.78     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_pretrained.pdparams)       |
| Res2Net50_<br>14w_8s          | 0.7946    | 0.9470    | 5.4026                | 10.60273             | 4.20     | 25.12     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_14w_8s_pretrained.pdparams)          |
| Res2Net101_vd_<br>26w_4s      | 0.8064    | 0.9522    | 8.08729               | 17.31208             | 8.35    | 45.35     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_pretrained.pdparams)      |
| Res2Net200_vd_<br>26w_4s      | 0.8121    | 0.9571    | 14.67806              | 32.35032             | 15.77    | 76.44     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_pretrained.pdparams)      |
| Res2Net200_vd_<br>26w_4s_ssld | 0.8513    | 0.9742    | 14.67806              | 32.35032             | 15.77    | 76.44     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_ssld_pretrained.pdparams) |
| ResNeXt50_<br>32x4d           | 0.7775    | 0.9382    | 7.56327               | 10.6134              | 4.26     | 25.10     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams)           |
| ResNeXt50_vd_<br>32x4d        | 0.7956    | 0.9462    | 7.62044               | 11.03385             | 4.50     | 25.12     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_32x4d_pretrained.pdparams)        |
| ResNeXt50_<br>64x4d           | 0.7843    | 0.9413    | 13.80962              | 18.4712              | 8.02    | 45.29     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams)           |
| ResNeXt50_vd_<br>64x4d        | 0.8012    | 0.9486    | 13.94449              | 18.88759             | 8.26    | 45.31     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_64x4d_pretrained.pdparams)        |
| ResNeXt101_<br>32x4d          | 0.7865    | 0.9419    | 16.21503              | 19.96568             | 8.01    | 44.32     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams)          |
| ResNeXt101_vd_<br>32x4d       | 0.8033    | 0.9512    | 16.28103              | 20.25611             | 8.25    | 44.33     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_32x4d_pretrained.pdparams)       |
| ResNeXt101_<br>64x4d          | 0.7835    | 0.9452    | 30.4788               | 36.29801             | 15.52    | 83.66     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams)          |
| ResNeXt101_vd_<br>64x4d       | 0.8078    | 0.9520    | 30.40456              | 36.77324             | 15.76    | 83.68     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_64x4d_pretrained.pdparams)       |
| ResNeXt152_<br>32x4d          | 0.7898    | 0.9433    | 24.86299              | 29.36764             | 11.76    | 60.15     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams)          |
| ResNeXt152_vd_<br>32x4d       | 0.8072    | 0.9520    | 25.03258              | 30.08987             | 12.01    | 60.17      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_32x4d_pretrained.pdparams)       |
| ResNeXt152_<br>64x4d          | 0.7951    | 0.9471    | 46.7564               | 56.34108             | 23.03    | 115.27    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams)          |
| ResNeXt152_vd_<br>64x4d       | 0.8108    | 0.9534    | 47.18638              | 57.16257             | 23.27    | 115.29   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_64x4d_pretrained.pdparams)       |
| SE_ResNet18_vd            | 0.7333    | 0.9138    | 1.7691                | 4.19877              | 2.07     | 11.81      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet18_vd_pretrained.pdparams)            |
| SE_ResNet34_vd            | 0.7651    | 0.9320    | 2.88559               | 7.03291              | 3.93     | 22.00     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet34_vd_pretrained.pdparams)            |
| SE_ResNet50_vd            | 0.7952    | 0.9475    | 4.28393               | 10.38846             | 4.36     | 28.16     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet50_vd_pretrained.pdparams)            |
| SE_ResNeXt50_<br>32x4d        | 0.7844    | 0.9396    | 8.74121               | 13.563               | 4.27     | 27.63     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_32x4d_pretrained.pdparams)        |
| SE_ResNeXt50_vd_<br>32x4d     | 0.8024    | 0.9489    | 9.17134               | 14.76192             | 5.64    | 27.76     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_vd_32x4d_pretrained.pdparams)     |
| SE_ResNeXt101_<br>32x4d       | 0.7939    | 0.9443    | 18.82604              | 25.31814             | 8.03    | 49.09     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt101_32x4d_pretrained.pdparams)       |
| SENet154_vd               | 0.8140    | 0.9548    | 53.79794              | 66.31684             | 24.45    | 122.03    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SENet154_vd_pretrained.pdparams)               |

<a name="7"></a>

## 7. DPN 与 DenseNet 系列

DPN 与 DenseNet 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[DPN 与 DenseNet 系列模型文档](../models/DPN_DenseNet.md)。


| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                         |
|-------------|-----------|-----------|-----------------------|----------------------|----------|-----------|--------------------------------------------------------------------------------------|
| DenseNet121 | 0.7566    | 0.9258    | 4.40447               | 9.32623              | 2.87     | 8.06      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams) |
| DenseNet161 | 0.7857    | 0.9414    | 10.39152              | 22.15555             | 7.79    | 28.90     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams) |
| DenseNet169 | 0.7681    | 0.9331    | 6.43598               | 12.98832             | 3.40     | 14.31     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams) |
| DenseNet201 | 0.7763    | 0.9366    | 8.20652               | 17.45838             | 4.34     | 20.24     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams) |
| DenseNet264 | 0.7796    | 0.9385    | 12.14722              | 26.27707             | 5.82    | 33.74     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams) |
| DPN68       | 0.7678    | 0.9343    | 11.64915              | 12.82807             | 2.35     | 12.68     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN68_pretrained.pdparams)       |
| DPN92       | 0.7985    | 0.9480    | 18.15746              | 23.87545             | 6.54    | 37.79     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN92_pretrained.pdparams)       |
| DPN98       | 0.8059    | 0.9510    | 21.18196              | 33.23925             | 11.728    | 61.74     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN98_pretrained.pdparams)       |
| DPN107      | 0.8089    | 0.9532    | 27.62046              | 52.65353             | 18.38    | 87.13     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN107_pretrained.pdparams)      |
| DPN131      | 0.8070    | 0.9514    | 28.33119              | 46.19439             | 16.09    | 79.48     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN131_pretrained.pdparams)      |



<a name="8"></a>

## 8. HRNet 系列

HRNet 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[HRNet 系列模型文档](../models/HRNet.md)。


| 模型          | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                 |
|-------------|-----------|-----------|------------------|------------------|----------|-----------|--------------------------------------------------------------------------------------|
| HRNet_W18_C | 0.7692    | 0.9339    | 7.40636          | 13.29752         | 4.32     | 21.35     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_pretrained.pdparams) |
| HRNet_W18_C_ssld | 0.81162    | 0.95804    | 7.40636          | 13.29752         | 4.32     | 21.35     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_ssld_pretrained.pdparams) |
| HRNet_W30_C | 0.7804    | 0.9402    | 9.57594          | 17.35485         | 8.15   | 37.78     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W30_C_pretrained.pdparams) |
| HRNet_W32_C | 0.7828    | 0.9424    | 9.49807          | 17.72921         | 8.97    | 41.30     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams) |
| HRNet_W40_C | 0.7877    | 0.9447    | 12.12202         | 25.68184         | 12.74    | 57.64     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W40_C_pretrained.pdparams) |
| HRNet_W44_C | 0.7900    | 0.9451    | 13.19858         | 32.25202         | 14.94    | 67.16     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W44_C_pretrained.pdparams) |
| HRNet_W48_C | 0.7895    | 0.9442    | 13.70761         | 34.43572         | 17.34    | 77.57     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_pretrained.pdparams) |
| HRNet_W48_C_ssld | 0.8363    | 0.9682    | 13.70761         | 34.43572         | 17.34    | 77.57     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_ssld_pretrained.pdparams) |
| HRNet_W64_C | 0.7930    | 0.9461    | 17.57527         | 47.9533          | 28.97    | 128.18    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W64_C_pretrained.pdparams) |
| SE_HRNet_W64_C_ssld | 0.8475    |  0.9726    |    31.69770      |     94.99546      | 29.00    | 129.12    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SE_HRNet_W64_C_ssld_pretrained.pdparams) |

<a name="9"></a>

## 9. Inception 系列

Inception 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[Inception 系列模型文档](../models/Inception.md)。

| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                         |
|--------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|---------------------------------------------------------------------------------------------|
| GoogLeNet          | 0.7070    | 0.8966    | 1.88038               | 4.48882              | 1.44     | 11.54      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams)          |
| Xception41         | 0.7930    | 0.9453    | 4.96939               | 17.01361             | 8.57    | 23.02     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams)         |
| Xception41_deeplab | 0.7955    | 0.9438    | 5.33541               | 17.55938             | 9.28    | 27.08     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_deeplab_pretrained.pdparams) |
| Xception65         | 0.8100    | 0.9549    | 7.26158               | 25.88778             | 13.25    | 36.04     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams)         |
| Xception65_deeplab | 0.8032    | 0.9449    | 7.60208               | 26.03699             | 13.96    | 40.10     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_deeplab_pretrained.pdparams) |
| Xception71         | 0.8111    | 0.9545    | 8.72457               | 31.55549             | 16.21    | 37.86     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams)         |
| InceptionV3        | 0.7914    | 0.9459    | 6.64054              | 13.53630              | 5.73    | 23.87     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/InceptionV3_pretrained.pdparams)        |
| InceptionV4        | 0.8077    | 0.9526    | 12.99342              | 25.23416             | 12.29    | 42.74     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams)        |

<a name="10"></a>

## 10. EfficientNet 与 ResNeXt101_wsl 系列

EfficientNet 与 ResNeXt101_wsl 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[EfficientNet 与 ResNeXt101_wsl 系列模型文档](../models/EfficientNet_and_ResNeXt101_wsl.md)。


| 模型                        | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                               |
|---------------------------|-----------|-----------|------------------|------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| ResNeXt101_<br>32x8d_wsl      | 0.8255    | 0.9674    | 18.52528         | 34.25319         | 16.48    | 88.99     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x8d_wsl_pretrained.pdparams)      |
| ResNeXt101_<br>32x16d_wsl     | 0.8424    | 0.9726    | 25.60395         | 71.88384         | 36.26    | 194.36    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x16d_wsl_pretrained.pdparams)     |
| ResNeXt101_<br>32x32d_wsl     | 0.8497    | 0.9759    | 54.87396         | 160.04337        | 87.28   | 469.12    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x32d_wsl_pretrained.pdparams)     |
| ResNeXt101_<br>32x48d_wsl     | 0.8537    | 0.9769    | 99.01698256      | 315.91261        | 153.57   | 829.26     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x48d_wsl_pretrained.pdparams)     |
| Fix_ResNeXt101_<br>32x48d_wsl | 0.8626    | 0.9797    | 160.0838242      | 595.99296        | 313.41   | 829.26     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Fix_ResNeXt101_32x48d_wsl_pretrained.pdparams) |
| EfficientNetB0            | 0.7738    | 0.9331    | 3.442            | 6.11476          | 0.40     | 5.33       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_pretrained.pdparams)            |
| EfficientNetB1            | 0.7915    | 0.9441    | 5.3322           | 9.41795          | 0.71     | 7.86      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB1_pretrained.pdparams)            |
| EfficientNetB2            | 0.7985    | 0.9474    | 6.29351          | 10.95702         | 1.02     | 9.18      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB2_pretrained.pdparams)            |
| EfficientNetB3            | 0.8115    | 0.9541    | 7.67749          | 16.53288         | 1.88     | 12.324     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB3_pretrained.pdparams)            |
| EfficientNetB4            | 0.8285    | 0.9623    | 12.15894         | 30.94567         | 4.51     | 19.47     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB4_pretrained.pdparams)            |
| EfficientNetB5            | 0.8362    | 0.9672    | 20.48571         | 61.60252         | 10.51    | 30.56     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB5_pretrained.pdparams)            |
| EfficientNetB6            | 0.8400    | 0.9688    | 32.62402         | -                | 19.47    | 43.27        | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB6_pretrained.pdparams)            |
| EfficientNetB7            | 0.8430    | 0.9689    | 53.93823         | -                | 38.45    | 66.66     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB7_pretrained.pdparams)            |
| EfficientNetB0_<br>small      | 0.7580    | 0.9258    | 2.3076           | 4.71886          | 0.40     | 4.69      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_small_pretrained.pdparams)      |

<a name="11"></a>

## 11. ResNeSt 与 RegNet 系列

ResNeSt 与 RegNet 系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[ResNeSt 与 RegNet 系列模型文档](../models/ResNeSt_RegNet.md)。


| 模型                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                                                                 |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| ResNeSt50_<br>fast_1s1x64d | 0.8035    | 0.9528    | 3.45405                | 8.72680                | 4.36     | 26.27      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_fast_1s1x64d_pretrained.pdparams) |
| ResNeSt50              | 0.8083    | 0.9542    | 6.69042    | 8.01664                | 5.40    | 27.54      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_pretrained.pdparams)              |
| RegNetX_4GF            | 0.785     | 0.9416    |    6.46478              |      11.19862           | 4.00        | 22.23      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams)            |

<a name="12"></a>

## 12. ViT_and_DeiT 系列

ViT(Vision Transformer) 与 DeiT（Data-efficient Image Transformers）系列模型的精度、速度指标如下表所示. 更多关于该系列模型的介绍可以参考： [ViT_and_DeiT 系列模型文档](../models/ViT_and_DeiT.md)。


| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址 |
|------------------------|-----------|-----------|------------------|------------------|----------|------------------------|------------------------|
| ViT_small_<br/>patch16_224 | 0.7769  | 0.9342   | -                | -                |   9.41   | 48.60 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams) |
| ViT_base_<br/>patch16_224 | 0.8195   | 0.9617   | -    | -                |  16.85   | 86.42 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams) |
| ViT_base_<br/>patch16_384 | 0.8414  | 0.9717   |    -              |      -           |    49.35     | 86.42 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams) |
| ViT_base_<br/>patch32_384 | 0.8176   | 0.9613   | - | - | 12.66 | 88.19 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams) |
| ViT_large_<br/>patch16_224 | 0.8323  | 0.9650   | - | - | 59.65 | 304.12 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams) |
| ViT_large_<br/>patch16_384 | 0.8513  | 0.9736  | - | - | 174.70 | 304.12

 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams) |
| ViT_large_<br/>patch32_384 | 0.8153   | 0.9608  | - | - | 44.24 | 306.48 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams) |



| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址 |
|------------------------|-----------|-----------|------------------|------------------|----------|------------------------|------------------------|
| DeiT_tiny_<br>patch16_224 | 0.718 | 0.910 | -                | -                |   1.07   | 5.68 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams) |
| DeiT_small_<br>patch16_224 | 0.796 | 0.949 | -    | -                |  4.24   | 21.97 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>patch16_224 | 0.817 | 0.957 |    -              |      -           |    16.85     | 86.42 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>patch16_384 | 0.830 | 0.962 | - | - | 49.35 | 86.42 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_384_pretrained.pdparams) |
| DeiT_tiny_<br>distilled_patch16_224 | 0.741 | 0.918 | - | - | 1.08 | 5.87 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_distilled_patch16_224_pretrained.pdparams) |
| DeiT_small_<br>distilled_patch16_224 | 0.809 | 0.953 | - | - | 4.26 | 22.36 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>distilled_patch16_224 | 0.831 | 0.964 | - | - | 16.93 | 87.18 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>distilled_patch16_384 | 0.851 | 0.973 | - | - | 49.43 | 87.18 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams) |

<a name="13"></a>

## 13. RepVGG 系列

关于 RepVGG 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[RepVGG 系列模型文档](../models/RepVGG.md)。


| 模型                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址 |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| RepVGG_A0   | 0.7131    | 0.9016    |  |  | 1.36 | 8.31 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A0_pretrained.pdparams) |
| RepVGG_A1   | 0.7380    | 0.9146    |  |  | 2.37 | 12.79 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A1_pretrained.pdparams) |
| RepVGG_A2   | 0.7571    | 0.9264    |  |  | 5.12 | 25.50 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A2_pretrained.pdparams) |
| RepVGG_B0   | 0.7450    | 0.9213    |  |  | 3.06 | 14.34 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B0_pretrained.pdparams) |
| RepVGG_B1   | 0.7773    | 0.9385    |  |  | 11.82 | 51.83 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1_pretrained.pdparams) |
| RepVGG_B2   | 0.7813    | 0.9410    |  |  | 18.38 | 80.32 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2_pretrained.pdparams) |
| RepVGG_B1g2 | 0.7732    | 0.9359    |  |  | 8.82 | 41.36 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g2_pretrained.pdparams) |
| RepVGG_B1g4 | 0.7675    | 0.9335    |  |  | 7.31 | 36.13 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g4_pretrained.pdparams) |
| RepVGG_B2g4 | 0.7881    | 0.9448    |  |  | 11.34 | 55.78 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2g4_pretrained.pdparams) |
| RepVGG_B3g4 | 0.7965    | 0.9485    |  |  | 16.07 | 75.63 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B3g4_pretrained.pdparams) |

<a name="14"></a>

## 14. MixNet 系列

关于 MixNet 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[MixNet 系列模型文档](../models/MixNet.md)。

| 模型     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(M) | Params(M) | 下载地址                                                     |
| -------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| MixNet_S | 0.7628    | 0.9299    |                  |                  | 252.977  | 4.167     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_S_pretrained.pdparams) |
| MixNet_M | 0.7767    | 0.9364    |                  |                  | 357.119  | 5.065     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_M_pretrained.pdparams) |
| MixNet_L | 0.7860    | 0.9437    |                  |                  | 579.017  | 7.384     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_L_pretrained.pdparams) |

<a name="15"></a>

## 15. ReXNet 系列

关于 ReXNet 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[ReXNet 系列模型文档](../models/ReXNet.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| ReXNet_1_0 | 0.7746    | 0.9370    |                  |                  | 0.415    | 4.84     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams) |
| ReXNet_1_3 | 0.7913    | 0.9464    |                  |                  | 0.68    | 7.61     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_3_pretrained.pdparams) |
| ReXNet_1_5 | 0.8006    | 0.9512    |                  |                  | 0.90    | 9.79     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_5_pretrained.pdparams) |
| ReXNet_2_0 | 0.8122    | 0.9536    |                  |                  | 1.56    | 16.45    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_2_0_pretrained.pdparams) |
| ReXNet_3_0 | 0.8209    | 0.9612    |                  |                  | 3.44    | 34.83    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_3_0_pretrained.pdparams) |

<a name="16"></a>

## 16. SwinTransformer 系列

关于 SwinTransformer 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[SwinTransformer 系列模型文档](../models/SwinTransformer.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| SwinTransformer_tiny_patch4_window7_224    | 0.8069 | 0.9534 |                  |                  | 4.35  | 28.26   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_small_patch4_window7_224   | 0.8275 | 0.9613 |                  |                  | 8.51  | 49.56   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_small_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_base_patch4_window7_224    | 0.8300 | 0.9626 |                  |                  | 15.13 | 87.70   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_base_patch4_window12_384   | 0.8439 | 0.9693 |                  |                  | 44.45 | 87.70   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window12_384_pretrained.pdparams) |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>     | 0.8487 | 0.9746 |                  |                  | 15.13 | 87.70   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_22kto1k_pretrained.pdparams) |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>    | 0.8642 | 0.9807 |                  |                  | 44.45 | 87.70   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window12_384_22kto1k_pretrained.pdparams) |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>    | 0.8596 | 0.9783 |                  |                  | 34.02 | 196.43  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained.pdparams) |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup>   | 0.8719 | 0.9823 |                  |                  | 99.97 | 196.43 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams) |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。

<a name="17"></a>

## 17. LeViT 系列

关于 LeViT 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[LeViT 系列模型文档](../models/LeViT.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(M) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| LeViT_128S | 0.7598    | 0.9269    |                  |                  | 281    | 7.42     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams) |
| LeViT_128 | 0.7810    | 0.9371    |                  |                  | 365    | 8.87     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams) |
| LeViT_192 | 0.7934    | 0.9446    |                  |                  | 597    | 10.61     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams) |
| LeViT_256 | 0.8085    | 0.9497    |                  |                  | 1049    | 18.45    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams) |
| LeViT_384 | 0.8191   | 0.9551    |                  |                  | 2234    | 38.45    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams) |

**注**：与 Reference 的精度差异源于数据预处理不同及未使用蒸馏的 head 作为输出。

<a name="18"></a>

## 18. Twins 系列

关于 Twins 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[Twins 系列模型文档](../models/Twins.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| pcpvt_small | 0.8082    | 0.9552    |                  |                  |3.67    | 24.06    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams) |
| pcpvt_base | 0.8242    | 0.9619    |                  |                  | 6.44    | 43.83    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams) |
| pcpvt_large | 0.8273    | 0.9650    |                  |                  | 9.50    | 60.99     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams) |
| alt_gvt_small | 0.8140    | 0.9546    |                  |                  |2.81   | 24.06   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams) |
| alt_gvt_base | 0.8294   | 0.9621    |                  |                  | 8.34   | 56.07   | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams) |
| alt_gvt_large | 0.8331   | 0.9642    |                  |                  | 14.81   | 99.27    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams) |

**注**：与 Reference 的精度差异源于数据预处理不同。

<a name="19"></a>

## 19. HarDNet 系列

关于 HarDNet 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[HarDNet 系列模型文档](../models/HarDNet.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| HarDNet39_ds | 0.7133    |0.8998    |                  |                  | 0.44   |  3.51    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet39_ds_pretrained.pdparams) |
| HarDNet68_ds |0.7362    | 0.9152   |                  |                  | 0.79   | 4.20 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_ds_pretrained.pdparams) |
| HarDNet68| 0.7546   | 0.9265   |                  |                  | 4.26   | 17.58    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_pretrained.pdparams) |
| HarDNet85 | 0.7744   | 0.9355   |                  |                  | 9.09   | 36.69  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet85_pretrained.pdparams) |

<a name="20"></a>

## 20. DLA 系列

关于 DLA 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[DLA 系列模型文档](../models/DLA.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| DLA102 | 0.7893    |0.9452    |                  |                  | 7.19   |  33.34    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams) |
| DLA102x2 |0.7885    | 0.9445  |                  |                  | 9.34   | 41.42 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x2_pretrained.pdparams) |
| DLA102x| 0.781   | 0.9400   |                  |                  | 5.89  | 26.40    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x_pretrained.pdparams) |
| DLA169 | 0.7809  | 0.9409   |                  |                  | 11.59  | 53.50  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA169_pretrained.pdparams) |
| DLA34 | 0.7603   | 0.9298    |                  |                  | 3.07   |  15.76    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams) |
| DLA46_c |0.6321   | 0.853   |                  |                  | 0.54   | 1.31 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA46_c_pretrained.pdparams) |
| DLA60 | 0.7610   | 0.9292   |                  |                  | 4.26   | 22.08    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60_pretrained.pdparams) |
| DLA60x_c | 0.6645   | 0.8754   |                  |                  | 0.59   | 1.33  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_c_pretrained.pdparams) |
| DLA60x | 0.7753  | 0.9378  |                  |                  | 3.54   | 17.41  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_pretrained.pdparams) |

<a name="21"></a>

## 21. RedNet 系列

关于 RedNet 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[RedNet 系列模型文档](../models/RedNet.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| RedNet26 | 0.7595   |0.9319  |                  |                  | 1.69   |  9.26    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet26_pretrained.pdparams) |
| RedNet38 |0.7747  | 0.9356  |                  |                  | 2.14   | 12.43 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet38_pretrained.pdparams) |
| RedNet50| 0.7833  | 0.9417   |                  |                  | 2.61   | 15.60    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet50_pretrained.pdparams) |
| RedNet101 | 0.7894  | 0.9436   |                  |                  | 4.59  | 25.76 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet101_pretrained.pdparams) |
| RedNet152 | 0.7917  | 0.9440   |                  |                  | 6.57  | 34.14  | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet152_pretrained.pdparams) |

<a name="22"></a>

## 22. TNT 系列

关于 TNT 系列模型的精度、速度指标如下表所示，更多介绍可以参考：[TNT 系列模型文档](../models/TNT.md)。

| 模型       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| TNT_small | 0.8121   |0.9563  |                  |                  | 4.83   |  23.68    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams) |               |  

**注**：TNT 模型的数据预处理部分 `NormalizeImage` 中的 `mean` 与 `std` 均为 0.5。

<a name="23"></a>

## 23. 其他模型

关于 AlexNet、SqueezeNet 系列、VGG 系列、DarkNet53 等模型的精度、速度指标如下表所示，更多介绍可以参考：[其他模型文档](../models/Others.md)。


| 模型                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | FLOPs(G) | Params(M) | 下载地址 |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| AlexNet       | 0.567 | 0.792 | 1.44993         | 2.46696         | 0.71 | 61.10 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams) |
| SqueezeNet1_0 | 0.596 | 0.817 | 0.96736 | 2.53221         | 0.78 | 1.25 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams) |
| SqueezeNet1_1 | 0.601 | 0.819 | 0.76032       | 1.877      | 0.35   | 1.24 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams) |
| VGG11 | 0.693 | 0.891 | 3.90412 | 9.51147 | 7.61 | 132.86 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG11_pretrained.pdparams) |
| VGG13 | 0.700 | 0.894 | 4.64684 | 12.61558 | 11.31 | 133.05 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG13_pretrained.pdparams) |
| VGG16 | 0.720 | 0.907 | 5.61769 | 16.40064 | 15.470 | 138.35 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG16_pretrained.pdparams) |
| VGG19 | 0.726 | 0.909 | 6.65221 | 20.4334 | 19.63 | 143.66 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG19_pretrained.pdparams) |
| DarkNet53 | 0.780 | 0.941 | 4.10829 | 12.1714 | 9.31 | 41.65 | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams) |
