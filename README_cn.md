简体中文 | [English](README.md)

# PaddleClas

## 简介

飞桨图像分类套件PaddleClas是飞桨为工业界和学术界所准备的一个图像分类任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

**近期更新**
- 2020.11.09 添加`InceptionV3 `结构和模型，在ImageNet-1k上Top-1 Acc可达79.14%。
- 2020.11.04 添加图像分类[常见问题2020第一季第一期](./docs/zh_CN/faq_series/faq_2020_s1.md) 7个新问题，并且计划以后每周一会更新，欢迎大家持续关注。
- 2020.10.12 添加Paddle-Lite demo。
- 2020.10.10 添加cpp inference demo，完善`FAQ 30问`教程。
- 2020.09.17 添加 `HRNet_W48_C_ssld `模型，在ImageNet-1k上Top-1 Acc可达83.62%；添加 `ResNet34_vd_ssld `模型，在ImageNet-1k上Top-1 Acc可达79.72%。
- 2020.09.07 添加 `HRNet_W18_C_ssld `模型，在ImageNet-1k上Top-1 Acc可达81.16%；添加 `MobileNetV3_small_x0_35_ssld `模型，在ImageNet-1k上Top-1 Acc可达55.55%。
- 2020.07.14 添加 `Res2Net200_vd_26w_4s_ssld `模型，在ImageNet-1k上Top-1 Acc可达85.13%；添加 `Fix_ResNet50_vd_ssld_v2 `模型，在ImageNet-1k上Top-1 Acc可达84.0%。
- 2020.06.17 添加英文文档。
- 2020.06.12 添加对windows和CPU环境的训练与评估支持。
- [more](./docs/zh_CN/update_history.md)


## 特性

- 丰富的模型库：基于ImageNet1k分类数据集，PaddleClas提供了24个系列的分类网络结构和训练配置，122个预训练模型和性能评估。

- SSLD知识蒸馏：基于该方案蒸馏模型的识别准确率普遍提升3%以上。

- 数据增广：支持AutoAugment、Cutout、Cutmix等8种数据增广算法详细介绍、代码复现和在统一实验环境下的效果评估。

- 10万类图像分类预训练模型：百度自研并开源了基于10万类数据集训练的 `ResNet50_vd `模型，在一些实际场景中，使用该预训练模型的识别准确率最多可以提升30%。

- 多种训练方案，包括多机训练、混合精度训练等。

- 多种预测推理、部署方案，包括TensorRT预测、Paddle-Lite预测、模型服务化部署、模型量化、Paddle Hub等。

- 可运行于Linux、Windows、MacOS等多种系统。


## 欢迎加入技术交流群

* 微信扫描二维码加入添加飞桨小姐姐的微信，添加成功后私信小姐姐暗号【分类】，即可收到系统进群邀请，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="./docs/images/joinus.png"  width = "200" height = "200" />
</div>


## 文档教程

- [快速安装](./docs/zh_CN/tutorials/install.md)
- [30分钟玩转PaddleClas](./docs/zh_CN/tutorials/quick_start.md)
- [模型库介绍和预训练模型](./docs/zh_CN/models/models_intro.md)
    - [模型库概览图](#模型库概览图)
    - [ResNet及其Vd系列](#ResNet及其Vd系列)
    - [移动端系列](#移动端系列)
    - [SEResNeXt与Res2Net系列](#SEResNeXt与Res2Net系列)
    - [DPN与DenseNet系列](#DPN与DenseNet系列)
    - [HRNet](HRNet系列)
    - [Inception系列](#Inception系列)
    - [EfficientNet与ResNeXt101_wsl系列](#EfficientNet与ResNeXt101_wsl系列)
    - [ResNeSt与RegNet系列](#ResNeSt与RegNet系列)
    - HS-ResNet: arxiv文章链接: [https://arxiv.org/pdf/2010.07621.pdf](https://arxiv.org/pdf/2010.07621.pdf)。 代码和预训练模型即将开源，敬请期待。
- 模型训练/评估
    - [数据准备](./docs/zh_CN/tutorials/data.md)
    - [模型训练与微调](./docs/zh_CN/tutorials/getting_started.md)
    - [模型评估](./docs/zh_CN/tutorials/getting_started.md)
- 模型预测
    - [基于训练引擎预测推理](./docs/zh_CN/extension/paddle_inference.md)
    - [基于Python预测引擎预测推理](./docs/zh_CN/extension/paddle_inference.md)
    - [基于C++预测引擎预测推理](./deploy/cpp_infer/readme.md)
    - [服务化部署](./docs/zh_CN/extension/paddle_serving.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [模型量化压缩](docs/zh_CN/extension/paddle_quantization.md)
- 高阶使用
    - [知识蒸馏](./docs/zh_CN/advanced_tutorials/distillation/distillation.md)
    - [数据增广](./docs/zh_CN/advanced_tutorials/image_augmentation/ImageAugment.md)
- 特色拓展应用
    - [迁移学习](./docs/zh_CN/application/transfer_learning.md)
    - [10万类图像分类预训练模型](./docs/zh_CN/application/transfer_learning.md)
    - [通用目标检测](./docs/zh_CN/application/object_detection.md)
- FAQ
    - [图像分类2020第一季精选问题(近期更新2020.11.04)](./docs/zh_CN/faq_series/faq_2020_s1.md)
    - [图像分类通用30个问题](./docs/zh_CN/faq.md)
    - [PaddleClas实战15个问题](./docs/zh_CN/faq.md)
- [赛事支持](./docs/zh_CN/competition_support.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)


## 模型库

<a name="模型库概览图"></a>
### 模型库概览图

基于ImageNet1k分类数据集，PaddleClas支持24种系列分类网络结构以及对应的122个图像分类预训练模型，训练技巧、每个系列网络结构的简单介绍和性能评估将在相应章节展现，下面所有的速度指标评估环境如下：
* CPU的评估环境基于骁龙855（SD855）。
* GPU评估环境基于T4机器，在FP32+TensorRT配置下运行500次测得（去除前10次的warmup时间）。

常见服务器端模型的精度指标与其预测耗时的变化曲线如下图所示。

![](./docs/images/models/T4_benchmark/t4.fp32.bs1.main_fps_top1.png)


常见移动端模型的精度指标与其预测耗时、模型存储大小的变化曲线如下图所示。

![](./docs/images/models/mobile_arm_storage.png)

![](./docs/images/models/mobile_arm_top1.png)


<a name="ResNet及其Vd系列"></a>
### ResNet及其Vd系列

ResNet及其Vd系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[ResNet及其Vd系列模型文档](./docs/zh_CN/models/ResNet_and_vd.md)。

| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                         |
|---------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------|
| ResNet18            | 0.7098    | 0.8992    | 1.45606               | 3.56305              | 3.66     | 11.69     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar)            |
| ResNet18_vd         | 0.7226    | 0.9080    | 1.54557               | 3.85363              | 4.14     | 11.71     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar)         |
| ResNet34            | 0.7457    | 0.9214    | 2.34957               | 5.89821              | 7.36     | 21.8      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar)            |
| ResNet34_vd         | 0.7598    | 0.9298    | 2.43427               | 6.22257              | 7.39     | 21.82     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_pretrained.tar)         |
| ResNet34_vd_ssld         | 0.7972    | 0.9490    | 2.43427               | 6.22257              | 7.39     | 21.82     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_vd_ssld_pretrained.tar)         |
| ResNet50            | 0.7650    | 0.9300    | 3.47712               | 7.84421              | 8.19     | 25.56     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar)            |
| ResNet50_vc         | 0.7835    | 0.9403    | 3.52346               | 8.10725              | 8.67     | 25.58     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vc_pretrained.tar)         |
| ResNet50_vd         | 0.7912    | 0.9444    | 3.53131               | 8.09057              | 8.67     | 25.58     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar)         |
| ResNet50_vd_v2      | 0.7984    | 0.9493    | 3.53131               | 8.09057              | 8.67     | 25.58     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_v2_pretrained.tar)      |
| ResNet101           | 0.7756    | 0.9364    | 6.07125               | 13.40573             | 15.52    | 44.55     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar)           |
| ResNet101_vd        | 0.8017    | 0.9497    | 6.11704               | 13.76222             | 16.1     | 44.57     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar)        |
| ResNet152           | 0.7826    | 0.9396    | 8.50198               | 19.17073             | 23.05    | 60.19     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_pretrained.tar)           |
| ResNet152_vd        | 0.8059    | 0.9530    | 8.54376               | 19.52157             | 23.53    | 60.21     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet152_vd_pretrained.tar)        |
| ResNet200_vd        | 0.8093    | 0.9533    | 10.80619              | 25.01731             | 30.53    | 74.74     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet200_vd_pretrained.tar)        |
| ResNet50_vd_<br>ssld    | 0.8239    | 0.9610    | 3.53131               | 8.09057              | 8.67     | 25.58     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar)    |
| ResNet50_vd_<br>ssld_v2 | 0.8300    | 0.9640    | 3.53131               | 8.09057              | 8.67     | 25.58     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_v2_pretrained.tar) |
| ResNet101_vd_<br>ssld   | 0.8373    | 0.9669    | 6.11704               | 13.76222             | 16.1     | 44.57     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar)   |


<a name="移动端系列"></a>
### 移动端系列

移动端系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[移动端系列模型文档](./docs/zh_CN/models/Mobile.md)。

| 模型                               | Top-1 Acc | Top-5 Acc | SD855 time(ms)<br>bs=1 | Flops(G) | Params(M) | 模型大小(M) | 下载地址                                                                                                      |
|----------------------------------|-----------|-----------|------------------------|----------|-----------|---------|-----------------------------------------------------------------------------------------------------------|
| MobileNetV1_<br>x0_25                | 0.5143    | 0.7546    | 3.21985                | 0.07     | 0.46      | 1.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_25_pretrained.tar)                |
| MobileNetV1_<br>x0_5                 | 0.6352    | 0.8473    | 9.579599               | 0.28     | 1.31      | 5.2     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_5_pretrained.tar)                 |
| MobileNetV1_<br>x0_75                | 0.6881    | 0.8823    | 19.436399              | 0.63     | 2.55      | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_x0_75_pretrained.tar)                |
| MobileNetV1                      | 0.7099    | 0.8968    | 32.523048              | 1.11     | 4.19      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar)                      |
| MobileNetV1_<br>ssld                 | 0.7789    | 0.9394    | 32.523048              | 1.11     | 4.19      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_ssld_pretrained.tar)                 |
| MobileNetV2_<br>x0_25                | 0.5321    | 0.7652    | 3.79925                | 0.05     | 1.5       | 6.1     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar)                |
| MobileNetV2_<br>x0_5                 | 0.6503    | 0.8572    | 8.7021                 | 0.17     | 1.93      | 7.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar)                 |
| MobileNetV2_<br>x0_75                | 0.6983    | 0.8901    | 15.531351              | 0.35     | 2.58      | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_75_pretrained.tar)                |
| MobileNetV2                      | 0.7215    | 0.9065    | 23.317699              | 0.6      | 3.44      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar)                      |
| MobileNetV2_<br>x1_5                 | 0.7412    | 0.9167    | 45.623848              | 1.32     | 6.76      | 26      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar)                 |
| MobileNetV2_<br>x2_0                 | 0.7523    | 0.9258    | 74.291649              | 2.32     | 11.13     | 43      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar)                 |
| MobileNetV2_<br>ssld                 | 0.7674    | 0.9339    | 23.317699              | 0.6      | 3.44      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_ssld_pretrained.tar)                 |
| MobileNetV3_<br>large_x1_25          | 0.7641    | 0.9295    | 28.217701              | 0.714    | 7.44      | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_25_pretrained.tar)          |
| MobileNetV3_<br>large_x1_0           | 0.7532    | 0.9231    | 19.30835               | 0.45     | 5.47      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar)           |
| MobileNetV3_<br>large_x0_75          | 0.7314    | 0.9108    | 13.5646                | 0.296    | 3.91      | 16      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_75_pretrained.tar)          |
| MobileNetV3_<br>large_x0_5           | 0.6924    | 0.8852    | 7.49315                | 0.138    | 2.67      | 11      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar)           |
| MobileNetV3_<br>large_x0_35          | 0.6432    | 0.8546    | 5.13695                | 0.077    | 2.1       | 8.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_35_pretrained.tar)          |
| MobileNetV3_<br>small_x1_25          | 0.7067    | 0.8951    | 9.2745                 | 0.195    | 3.62      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_25_pretrained.tar)          |
| MobileNetV3_<br>small_x1_0           | 0.6824    | 0.8806    | 6.5463                 | 0.123    | 2.94      | 12      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar)           |
| MobileNetV3_<br>small_x0_75          | 0.6602    | 0.8633    | 5.28435                | 0.088    | 2.37      | 9.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x0_75_pretrained.tar)          |
| MobileNetV3_<br>small_x0_5           | 0.5921    | 0.8152    | 3.35165                | 0.043    | 1.9       | 7.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x0_5_pretrained.tar)           |
| MobileNetV3_<br>small_x0_35          | 0.5303    | 0.7637    | 2.6352                 | 0.026    | 1.66      | 6.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x0_35_pretrained.tar)          |
| MobileNetV3_<br>small_x0_35_ssld          | 0.5555    | 0.7771    | 2.6352                 | 0.026    | 1.66      | 6.9     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x0_35_ssld_pretrained.tar)          |
| MobileNetV3_<br>large_x1_0_ssld      | 0.7896    | 0.9448    | 19.30835               | 0.45     | 5.47      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar)      |
| MobileNetV3_large_<br>x1_0_ssld_int8 | 0.7605    |     -      | 14.395                 |    -     |      -     | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_int8_pretrained.tar) |
| MobileNetV3_small_<br>x1_0_ssld      | 0.7129    | 0.9010    | 6.5463                 | 0.123    | 2.94      | 12      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar)      |
| ShuffleNetV2                     | 0.6880    | 0.8845    | 10.941                 | 0.28     | 2.26      | 9       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar)                     |
| ShuffleNetV2_<br>x0_25               | 0.4990    | 0.7379    | 2.329                  | 0.03     | 0.6       | 2.7     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_25_pretrained.tar)               |
| ShuffleNetV2_<br>x0_33               | 0.5373    | 0.7705    | 2.64335                | 0.04     | 0.64      | 2.8     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_33_pretrained.tar)               |
| ShuffleNetV2_<br>x0_5                | 0.6032    | 0.8226    | 4.2613                 | 0.08     | 1.36      | 5.6     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x0_5_pretrained.tar)                |
| ShuffleNetV2_<br>x1_5                | 0.7163    | 0.9015    | 19.3522                | 0.58     | 3.47      | 14      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x1_5_pretrained.tar)                |
| ShuffleNetV2_<br>x2_0                | 0.7315    | 0.9120    | 34.770149              | 1.12     | 7.32      | 28      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_x2_0_pretrained.tar)                |
| ShuffleNetV2_<br>swish               | 0.7003    | 0.8917    | 16.023151              | 0.29     | 2.26      | 9.1     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_swish_pretrained.tar)               |
| DARTS_GS_4M                      | 0.7523    | 0.9215    | 47.204948              | 1.04     | 4.77      | 21      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_4M_pretrained.tar)                      |
| DARTS_GS_6M                      | 0.7603    | 0.9279    | 53.720802              | 1.22     | 5.69      | 24      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DARTS_GS_6M_pretrained.tar)                      |
| GhostNet_<br>x0_5                    | 0.6688    | 0.8695    | 5.7143                 | 0.082    | 2.6       | 10      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/GhostNet_x0_5_pretrained.pdparams)               |
| GhostNet_<br>x1_0                    | 0.7402    | 0.9165    | 13.5587                | 0.294    | 5.2       | 20      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/GhostNet_x1_0_pretrained.pdparams)               |
| GhostNet_<br>x1_3                    | 0.7579    | 0.9254    | 19.9825                | 0.44     | 7.3       | 29      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/GhostNet_x1_3_pretrained.pdparams)               |


<a name="SEResNeXt与Res2Net系列"></a>
### SEResNeXt与Res2Net系列

SEResNeXt与Res2Net系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[SEResNeXt与Res2Net系列模型文档](./docs/zh_CN/models/SEResNext_and_Res2Net.md)。


| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                         |
|---------------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| Res2Net50_<br>26w_4s          | 0.7933    | 0.9457    | 4.47188               | 9.65722              | 8.52     | 25.7      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_26w_4s_pretrained.tar)          |
| Res2Net50_vd_<br>26w_4s       | 0.7975    | 0.9491    | 4.52712               | 9.93247              | 8.37     | 25.06     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_vd_26w_4s_pretrained.tar)       |
| Res2Net50_<br>14w_8s          | 0.7946    | 0.9470    | 5.4026                | 10.60273             | 9.01     | 25.72     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net50_14w_8s_pretrained.tar)          |
| Res2Net101_vd_<br>26w_4s      | 0.8064    | 0.9522    | 8.08729               | 17.31208             | 16.67    | 45.22     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net101_vd_26w_4s_pretrained.tar)      |
| Res2Net200_vd_<br>26w_4s      | 0.8121    | 0.9571    | 14.67806              | 32.35032             | 31.49    | 76.21     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net200_vd_26w_4s_pretrained.tar)      |
| Res2Net200_vd_<br>26w_4s_ssld | 0.8513    | 0.9742    | 14.67806              | 32.35032             | 31.49    | 76.21     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Res2Net200_vd_26w_4s_ssld_pretrained.tar) |
| ResNeXt50_<br>32x4d           | 0.7775    | 0.9382    | 7.56327               | 10.6134              | 8.02     | 23.64     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_32x4d_pretrained.tar)           |
| ResNeXt50_vd_<br>32x4d        | 0.7956    | 0.9462    | 7.62044               | 11.03385             | 8.5      | 23.66     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_32x4d_pretrained.tar)        |
| ResNeXt50_<br>64x4d           | 0.7843    | 0.9413    | 13.80962              | 18.4712              | 15.06    | 42.36     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_64x4d_pretrained.tar)           |
| ResNeXt50_vd_<br>64x4d        | 0.8012    | 0.9486    | 13.94449              | 18.88759             | 15.54    | 42.38     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt50_vd_64x4d_pretrained.tar)        |
| ResNeXt101_<br>32x4d          | 0.7865    | 0.9419    | 16.21503              | 19.96568             | 15.01    | 41.54     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x4d_pretrained.tar)          |
| ResNeXt101_vd_<br>32x4d       | 0.8033    | 0.9512    | 16.28103              | 20.25611             | 15.49    | 41.56     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_32x4d_pretrained.tar)       |
| ResNeXt101_<br>64x4d          | 0.7835    | 0.9452    | 30.4788               | 36.29801             | 29.05    | 78.12     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_64x4d_pretrained.tar)          |
| ResNeXt101_vd_<br>64x4d       | 0.8078    | 0.9520    | 30.40456              | 36.77324             | 29.53    | 78.14     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_vd_64x4d_pretrained.tar)       |
| ResNeXt152_<br>32x4d          | 0.7898    | 0.9433    | 24.86299              | 29.36764             | 22.01    | 56.28     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_32x4d_pretrained.tar)          |
| ResNeXt152_vd_<br>32x4d       | 0.8072    | 0.9520    | 25.03258              | 30.08987             | 22.49    | 56.3      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_32x4d_pretrained.tar)       |
| ResNeXt152_<br>64x4d          | 0.7951    | 0.9471    | 46.7564               | 56.34108             | 43.03    | 107.57    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_64x4d_pretrained.tar)          |
| ResNeXt152_vd_<br>64x4d       | 0.8108    | 0.9534    | 47.18638              | 57.16257             | 43.52    | 107.59    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt152_vd_64x4d_pretrained.tar)       |
| SE_ResNet18_vd            | 0.7333    | 0.9138    | 1.7691                | 4.19877              | 4.14     | 11.8      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet18_vd_pretrained.tar)            |
| SE_ResNet34_vd            | 0.7651    | 0.9320    | 2.88559               | 7.03291              | 7.84     | 21.98     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet34_vd_pretrained.tar)            |
| SE_ResNet50_vd            | 0.7952    | 0.9475    | 4.28393               | 10.38846             | 8.67     | 28.09     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNet50_vd_pretrained.tar)            |
| SE_ResNeXt50_<br>32x4d        | 0.7844    | 0.9396    | 8.74121               | 13.563               | 8.02     | 26.16     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_32x4d_pretrained.tar)        |
| SE_ResNeXt50_vd_<br>32x4d     | 0.8024    | 0.9489    | 9.17134               | 14.76192             | 10.76    | 26.28     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt50_vd_32x4d_pretrained.tar)     |
| SE_ResNeXt101_<br>32x4d       | 0.7912    | 0.9420    | 18.82604              | 25.31814             | 15.02    | 46.28     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SE_ResNeXt101_32x4d_pretrained.tar)       |
| SENet154_vd               | 0.8140    | 0.9548    | 53.79794              | 66.31684             | 45.83    | 114.29    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/SENet154_vd_pretrained.tar)               |


<a name="DPN与DenseNet系列"></a>
### DPN与DenseNet系列

DPN与DenseNet系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[DPN与DenseNet系列模型文档](./docs/zh_CN/models/DPN_DenseNet.md)。


| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                         |
|-------------|-----------|-----------|-----------------------|----------------------|----------|-----------|--------------------------------------------------------------------------------------|
| DenseNet121 | 0.7566    | 0.9258    | 4.40447               | 9.32623              | 5.69     | 7.98      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) |
| DenseNet161 | 0.7857    | 0.9414    | 10.39152              | 22.15555             | 15.49    | 28.68     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) |
| DenseNet169 | 0.7681    | 0.9331    | 6.43598               | 12.98832             | 6.74     | 14.15     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet169_pretrained.tar) |
| DenseNet201 | 0.7763    | 0.9366    | 8.20652               | 17.45838             | 8.61     | 20.01     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) |
| DenseNet264 | 0.7796    | 0.9385    | 12.14722              | 26.27707             | 11.54    | 33.37     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet264_pretrained.tar) |
| DPN68       | 0.7678    | 0.9343    | 11.64915              | 12.82807             | 4.03     | 10.78     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DPN68_pretrained.tar)       |
| DPN92       | 0.7985    | 0.9480    | 18.15746              | 23.87545             | 12.54    | 36.29     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DPN92_pretrained.tar)       |
| DPN98       | 0.8059    | 0.9510    | 21.18196              | 33.23925             | 22.22    | 58.46     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DPN98_pretrained.tar)       |
| DPN107      | 0.8089    | 0.9532    | 27.62046              | 52.65353             | 35.06    | 82.97     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DPN107_pretrained.tar)      |
| DPN131      | 0.8070    | 0.9514    | 28.33119              | 46.19439             | 30.51    | 75.36     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/DPN131_pretrained.tar)      |



<a name="HRNet系列"></a>
### HRNet系列

HRNet系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[HRNet系列模型文档](./docs/zh_CN/models/HRNet.md)。


| 模型          | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                 |
|-------------|-----------|-----------|------------------|------------------|----------|-----------|--------------------------------------------------------------------------------------|
| HRNet_W18_C | 0.7692    | 0.9339    | 7.40636          | 13.29752         | 4.14     | 21.29     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) |
| HRNet_W18_C_ssld | 0.81162    | 0.95804    | 7.40636          | 13.29752         | 4.14     | 21.29     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_ssld_pretrained.tar) |
| HRNet_W30_C | 0.7804    | 0.9402    | 9.57594          | 17.35485         | 16.23    | 37.71     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W30_C_pretrained.tar) |
| HRNet_W32_C | 0.7828    | 0.9424    | 9.49807          | 17.72921         | 17.86    | 41.23     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W32_C_pretrained.tar) |
| HRNet_W40_C | 0.7877    | 0.9447    | 12.12202         | 25.68184         | 25.41    | 57.55     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W40_C_pretrained.tar) |
| HRNet_W44_C | 0.7900    | 0.9451    | 13.19858         | 32.25202         | 29.79    | 67.06     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W44_C_pretrained.tar) |
| HRNet_W48_C | 0.7895    | 0.9442    | 13.70761         | 34.43572         | 34.58    | 77.47     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar) |
| HRNet_W48_C_ssld | 0.8363    | 0.9682    | 13.70761         | 34.43572         | 34.58    | 77.47     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar) |
| HRNet_W64_C | 0.7930    | 0.9461    | 17.57527         | 47.9533          | 57.83    | 128.06    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar) |


<a name="Inception系列"></a>
### Inception系列

Inception系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[Inception系列模型文档](./docs/zh_CN/models/Inception.md)。

| 模型                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                         |
|--------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|---------------------------------------------------------------------------------------------|
| GoogLeNet          | 0.7070    | 0.8966    | 1.88038               | 4.48882              | 2.88     | 8.46      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/GoogLeNet_pretrained.tar)          |
| Xception41         | 0.7930    | 0.9453    | 4.96939               | 17.01361             | 16.74    | 22.69     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_pretrained.tar)         |
| Xception41_deeplab | 0.7955    | 0.9438    | 5.33541               | 17.55938             | 18.16    | 26.73     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) |
| Xception65         | 0.8100    | 0.9549    | 7.26158               | 25.88778             | 25.95    | 35.48     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_pretrained.tar)         |
| Xception65_deeplab | 0.8032    | 0.9449    | 7.60208               | 26.03699             | 27.37    | 39.52     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) |
| Xception71         | 0.8111    | 0.9545    | 8.72457               | 31.55549             | 31.77    | 37.28     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Xception71_pretrained.tar)         |
| InceptionV3        | 0.7914    | 0.9459    | 6.64054              | 13.53630              | 11.46    | 23.83     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV3_pretrained.tar)        |
| InceptionV4        | 0.8077    | 0.9526    | 12.99342              | 25.23416             | 24.57    | 42.68     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar)        |


<a name="EfficientNet与ResNeXt101_wsl系列"></a>
### EfficientNet与ResNeXt101_wsl系列

EfficientNet与ResNeXt101_wsl系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[EfficientNet与ResNeXt101_wsl系列模型文档](./docs/zh_CN/models/EfficientNet_and_ResNeXt101_wsl.md)。


| 模型                        | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                               |
|---------------------------|-----------|-----------|------------------|------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| ResNeXt101_<br>32x8d_wsl      | 0.8255    | 0.9674    | 18.52528         | 34.25319         | 29.14    | 78.44     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x8d_wsl_pretrained.tar)      |
| ResNeXt101_<br>32x16d_wsl     | 0.8424    | 0.9726    | 25.60395         | 71.88384         | 57.55    | 152.66    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x16d_wsl_pretrained.tar)     |
| ResNeXt101_<br>32x32d_wsl     | 0.8497    | 0.9759    | 54.87396         | 160.04337        | 115.17   | 303.11    | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x32d_wsl_pretrained.tar)     |
| ResNeXt101_<br>32x48d_wsl     | 0.8537    | 0.9769    | 99.01698256      | 315.91261        | 173.58   | 456.2     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeXt101_32x48d_wsl_pretrained.tar)     |
| Fix_ResNeXt101_<br>32x48d_wsl | 0.8626    | 0.9797    | 160.0838242      | 595.99296        | 354.23   | 456.2     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/Fix_ResNeXt101_32x48d_wsl_pretrained.tar) |
| EfficientNetB0            | 0.7738    | 0.9331    | 3.442            | 6.11476          | 0.72     | 5.1       | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_pretrained.tar)            |
| EfficientNetB1            | 0.7915    | 0.9441    | 5.3322           | 9.41795          | 1.27     | 7.52      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB1_pretrained.tar)            |
| EfficientNetB2            | 0.7985    | 0.9474    | 6.29351          | 10.95702         | 1.85     | 8.81      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB2_pretrained.tar)            |
| EfficientNetB3            | 0.8115    | 0.9541    | 7.67749          | 16.53288         | 3.43     | 11.84     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB3_pretrained.tar)            |
| EfficientNetB4            | 0.8285    | 0.9623    | 12.15894         | 30.94567         | 8.29     | 18.76     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB4_pretrained.tar)            |
| EfficientNetB5            | 0.8362    | 0.9672    | 20.48571         | 61.60252         | 19.51    | 29.61     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB5_pretrained.tar)            |
| EfficientNetB6            | 0.8400    | 0.9688    | 32.62402         | -                | 36.27    | 42        | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB6_pretrained.tar)            |
| EfficientNetB7            | 0.8430    | 0.9689    | 53.93823         | -                | 72.35    | 64.92     | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB7_pretrained.tar)            |
| EfficientNetB0_<br>small      | 0.7580    | 0.9258    | 2.3076           | 4.71886          | 0.72     | 4.65      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/EfficientNetB0_small_pretrained.tar)      |


<a name="ResNeSt与RegNet系列"></a>
### ResNeSt与RegNet系列

ResNeSt与RegNet系列模型的精度、速度指标如下表所示，更多关于该系列的模型介绍可以参考：[ResNeSt与RegNet系列模型文档](./docs/zh_CN/models/ResNeSt_RegNet.md)。


| 模型                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | 下载地址                                                                                                 |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| ResNeSt50_<br>fast_1s1x64d | 0.8035    | 0.9528    | 3.45405                | 8.72680                | 8.68     | 26.3      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeSt50_fast_1s1x64d_pretrained.pdparams) |
| ResNeSt50              | 0.8102    | 0.9542    | 6.69042    | 8.01664                | 10.78    | 27.5      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/ResNeSt50_pretrained.pdparams)              |
| RegNetX_4GF            | 0.785     | 0.9416    |    6.46478              |      11.19862           | 8        | 22.1      | [下载链接](https://paddle-imagenet-models-name.bj.bcebos.com/RegNetX_4GF_pretrained.pdparams)            |


<a name="许可证书"></a>
## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleCLS/blob/master/LICENSE">Apache 2.0 license</a>许可认证。


<a name="贡献代码"></a>
## 贡献代码
我们非常欢迎你为PaddleClas贡献代码，也十分感谢你的反馈。

- 非常感谢[nblib](https://github.com/nblib)修正了PaddleClas中RandErasing的数据增广配置文件。
- 非常感谢[chenpy228](https://github.com/chenpy228)修正了PaddleClas文档中的部分错别字。
