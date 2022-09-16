# 版本更新信息
----------
## 目录
* [1. v2.3](#1)
* [2. v2.2](#2)

<a name='1'></a>

## 1. v2.3

- 模型更新
  - 添加轻量化模型预训练权重，包括检测模型、特征模型
  - 发布 PP-LCNet 系列模型，此系列模型是专门在 CPU 上设计运行的自研模型
  - SwinTransformer、Twins、Deit 支持从 scrach 直接训练，达到论文精度
- 框架基础能力
  - 添加 DeepHash 模块，支持特征模型直接输出二值特征
  - 添加 PKSampler，特征模型不能多机多卡的训练的问题
  - 支持 PaddleSlim：支持分类模型、特征模型的量化、裁剪训练及离线量化功能
  - Legendary models 支持模型中间结果输出
  - 支持多标签分类训练
- 预测部署
  - 使用 Faiss 替换原有特征检索库，提升平台适配性
  - 支持 PaddleServing：支持分类模型、图像识别流程的部署

- 推荐库版本
  - python 版本：3.7
  - PaddlePaddle 版本：2.1.3
  - PaddleSlim 版本：2.2.0
  - PaddleServing 版本：0.6.1

<a name='2'></a>

## 2. v2.2

- 模型更新
  - 添加 LeViT、Twins、TNT、DLA、HardNet、RedNet、SwinTransfomer 模型
- 框架基础能力
  - 将分类模型分为两类
    -  legendary models：引入 TheseusLayer 基类，及增加了修改网络功能接口，同时支持网络截断输出功能
    - model zoo：其他普通分类模型
  - 添加 Metric Learning 算法支持
    - 添加多种相关 Loss 算法，及基础网络模块 gears（支持与 backbone、loss 组合）方便使用
    - 同时支持普通分类及 metric learning 相关任务训练
  - 支持静态图训练
  - 分类训练支持 dali 加速
  - 支持 fp16 训练
- 应用更新
  - 添加商品识别、车辆识别（车辆细粒度分类、车辆 ReID）、logo 识别、动漫人物识别应用具体案例及相关模型
  - 添加图像识别完整 pipeline，包含检测模块、特征提取模块、向量检索模块
- 预测部署
  - 添加百度自研向量检索模块 Mobius，支持图像识别系统预测部署
  - 图像识别，建立特征库支持 batch_size>1
- 文档更新
  - 添加图像识别相关文档
  - 修复之前文档 bug
- 推荐库版本
  - python 版本：3.7
  - PaddlePaddle：2.1.2

# 更新日志

- 2022.4.21 新增 CVPR2022 oral论文 [MixFormer](https://arxiv.org/pdf/2204.02557.pdf) 相关[代码](https://github.com/PaddlePaddle/PaddleClas/pull/1820/files)。
- 2021.11.1 发布[PP-ShiTu技术报告](https://arxiv.org/pdf/2111.00775.pdf)，新增饮料识别demo。
- 2021.10.23 发布轻量级图像识别系统PP-ShiTu，CPU上0.2s即可完成在10w+库的图像识别。[点击这里](quick_start/quick_start_recognition.md)立即体验。
- 2021.09.17 发布PP-LCNet系列超轻量骨干网络模型, 在Intel CPU上，单张图像预测速度约5ms，ImageNet-1K数据集上Top1识别准确率达到80.82%，超越ResNet152的模型效果。PP-LCNet的介绍可以参考[论文](https://arxiv.org/pdf/2109.15099.pdf), 或者[PP-LCNet模型介绍](../models/PP-LCNet.md)，相关指标和预训练权重可以从 [这里](models/ImageNet1k/model_list.md)下载。
- 2021.08.11 更新 7 个[FAQ](FAQ/faq_2021_s2.md)。
- 2021.06.29 添加 Swin-transformer 系列模型，ImageNet1k 数据集上 Top1 acc 最高精度可达 87.2%；支持训练预测评估与 whl 包部署，预训练模型可以从[这里](models/ImageNet1k/model_list.md)下载。
- 2021.06.22,23,24 PaddleClas 官方研发团队带来技术深入解读三日直播课。课程回放：[https://aistudio.baidu.com/aistudio/course/introduce/24519](https://aistudio.baidu.com/aistudio/course/introduce/24519)
- 2021.06.16 PaddleClas v2.2 版本升级，集成 Metric learning，向量检索等组件。新增商品识别、动漫人物识别、车辆识别和 logo 识别等 4 个图像识别应用。新增 LeViT、Twins、TNT、DLA、HarDNet、RedNet 系列 30 个预训练模型。
- 2021.04.15
   - 添加 `MixNet_L` 和 `ReXNet_3_0` 系列模型，在 ImageNet-1k 上 `MixNet` 模型 Top1 Acc 可达 78.6%，`ReXNet` 模型可达 82.09%
- 2021.01.27
   * 添加 ViT 与 DeiT 模型，在 ImageNet 上，ViT 模型 Top-1 Acc 可达 81.05%，DeiT 模型可达 85.5%。
- 2021.01.08
    * 添加 whl 包及其使用说明，直接安装 paddleclas whl 包，即可快速完成模型预测。
- 2020.12.16
    * 添加对 cpp 预测的 tensorRT 支持，预测加速更明显。
- 2020.12.06
    * 添加 SE_HRNet_W64_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.8475。
- 2020.11.23
    * 添加 GhostNet_x1_3_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.7938。
- 2020.11.09
    * 添加 InceptionV3 结构和模型，在 ImageNet 上 Top-1 Acc 可达 0.791。
- 2020.10.20
    * 添加 Res2Net50_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.831；添加 Res2Net101_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.839。
- 2020.10.12
    * 添加 Paddle-Lite demo。
- 2020.10.10
    * 添加 cpp inference demo。
    * 添加 FAQ 30 问。
- 2020.09.17
    * 添加 HRNet_W48_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.836；添加 ResNet34_vd_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.797。

* 2020.09.07
    * 添加 HRNet_W18_C_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.81162；添加 MobileNetV3_small_x0_35_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 0.5555。

* 2020.07.14
    * 添加 Res2Net200_vd_26w_4s_ssld 模型，在 ImageNet 上 Top-1 Acc 可达 85.13%。
    * 添加 Fix_ResNet50_vd_ssld_v2 模型，，在 ImageNet 上 Top-1 Acc 可达 84.0%。

* 2020.06.17
    * 添加英文文档。

* 2020.06.12
    * 添加对 windows 和 CPU 环境的训练与评估支持。

* 2020.05.17
    * 添加混合精度训练。

* 2020.05.09
    * 添加 Paddle Serving 使用文档。
    * 添加 Paddle-Lite 使用文档。
    * 添加 T4 GPU 的 FP32/FP16 预测速度 benchmark。

* 2020.04.10:
    * 第一次提交。
