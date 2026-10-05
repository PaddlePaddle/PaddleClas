# MobileNetV5 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 当前支持的模型](#1.2)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
  - [4.2 基于 Python 预测引擎推理](#4.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

MobileNetV5 是面向高效视觉任务设计的轻量级网络结构。当前 PaddleClas 中提供的实现参考了 `timm` 中的 MobileNetV5 定义，核心模块包括：

- `EdgeResidual`
- `UniversalInvertedResidual`
- `MobileAttention`
- `MSFA (Multi-Scale Fusion Adapter)`

其中，`MobileNetV5_300M` 与 `MobileNetV5_base` 用于 ImageNet1k 分类任务；`MobileNetV5_300M_enc` 为特征提取版本，默认输出特征图；`MobileNetV5_300M_enc_cls` 则是在 `enc` 主干设定基础上补充分类头后的分类版本。

当前仓库已完成以下核验工作：

- 与 `timm` 参考实现的前向对齐
- 随机初始化权重转换与加载验证
- `lite_train_lite_infer` 模式下的 TIPC 训练、评估、导出与推理验证

在前向对齐阶段，使用 `timm` 中的 `MobileNetV5` 参考实现作为基线，采用固定输入和同源随机初始化权重进行比对。当前已经完成 `MobileNetV5_300M` 与 `MobileNetV5_base` 的同权重前向核验，绝对误差处于正常框架数值误差范围内：

| Models | 对齐节点 | max abs diff | mean abs diff |
|:--:|:--:|:--:|:--:|
| MobileNetV5_300M | `forward_features` | `1.43e-05` | `1.99e-06` |
| MobileNetV5_300M | `out` | `1.91e-06` | `4.51e-07` |
| MobileNetV5_300M_enc | `forward_features` | `1.31e-05` | `1.89e-06` |
| MobileNetV5_300M_enc | `out` | `1.31e-05` | `1.89e-06` |
| MobileNetV5_base | `forward_features` | `1.34e-05` | `1.96e-06` |
| MobileNetV5_base | `out` | `1.25e-06` | `2.79e-07` |

上述结果说明当前 Paddle 实现与参考实现已经完成前向数值对齐。

说明：

- 当前文档阶段不提供全量 ImageNet 训练精度与预训练权重下载链接
- `MobileNetV5_300M_enc` 为纯特征提取模型，不直接对应分类推理入口

<a name='1.2'></a>

### 1.2 当前支持的模型

| Models | 用途 | 输出形式 | 训练配置 | TIPC mode 1 |
|:--:|:--:|:--:|:--:|:--:|
| MobileNetV5_300M | ImageNet1k 分类 | logits | 支持 | 已验证 |
| MobileNetV5_base | ImageNet1k 分类 | logits | 支持 | 已验证 |
| MobileNetV5_300M_enc | 特征提取 | feature map | 不适用分类配置 | 不适用分类 TIPC |
| MobileNetV5_300M_enc_cls | ImageNet1k 分类 | logits | 支持 | 已验证 |

训练配置位于：

- `ppcls/configs/ImageNet/MobileNetV5/MobileNetV5_300M.yaml`
- `ppcls/configs/ImageNet/MobileNetV5/MobileNetV5_base.yaml`
- `ppcls/configs/ImageNet/MobileNetV5/MobileNetV5_300M_enc_cls.yaml`

TIPC 配置位于：

- `test_tipc/configs/MobileNetV5/MobileNetV5_300M_train_infer_python.txt`
- `test_tipc/configs/MobileNetV5/MobileNetV5_base_train_infer_python.txt`
- `test_tipc/configs/MobileNetV5/MobileNetV5_300M_enc_cls_train_infer_python.txt`

基于 TIPC 自动准备的 lite ImageNet 数据，对 `MobileNetV5_300M` 做了一个 10 epoch 的快速收敛性验证。训练过程中未出现 `NaN/Inf`，并且训练集平均 loss 呈现明显下降趋势：

| Epoch | Train Avg Loss |
|:--:|:--:|
| 1 | 7.84382 |
| 2 | 8.15822 |
| 3 | 7.57893 |
| 4 | 6.41290 |
| 5 | 6.17161 |
| 6 | 5.56597 |
| 7 | 5.47614 |
| 8 | 5.07293 |
| 9 | 5.05199 |
| 10 | 4.98907 |

从第 1 个 epoch 的 `7.84382` 下降到第 10 个 epoch 的 `4.98907`，说明当前实现具备正常的优化与收敛趋势。该结果用于验证模型训练链路与收敛行为，非全量 ImageNet 最终精度。

同样地，`MobileNetV5_base` 在 lite ImageNet 上也完成了 10 epoch 的快速收敛性验证，训练过程中未出现 `NaN/Inf`，训练集平均 loss 整体下降：

| Epoch | Train Avg Loss |
|:--:|:--:|
| 1 | 7.77208 |
| 2 | 7.62792 |
| 3 | 6.14518 |
| 4 | 5.48411 |
| 5 | 5.33627 |
| 6 | 4.84439 |
| 7 | 4.94096 |
| 8 | 4.71992 |
| 9 | 4.91374 |
| 10 | 4.75953 |

从第 1 个 epoch 的 `7.77208` 下降到第 10 个 epoch 的 `4.75953`，说明 `MobileNetV5_base` 同样具备正常的优化与收敛趋势。该结果同样用于验证训练链路与收敛行为，非全量 ImageNet 最终精度。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考 [ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

若使用分类模型，可将 `Arch.name` 设置为：

- `MobileNetV5_300M`
- `MobileNetV5_base`
- `MobileNetV5_300M_enc_cls`

若使用特征提取模型：

- `MobileNetV5_300M_enc`

该模型默认输出特征图，适合作为 backbone 或特征抽取器使用。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet 数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。`ppcls/configs/ImageNet/MobileNetV5/` 中提供了该系列分类模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

当前推荐优先使用以下配置：

- `MobileNetV5_300M.yaml`
- `MobileNetV5_base.yaml`
- `MobileNetV5_300M_enc_cls.yaml`

如果仅验证链路可用性，可优先运行 TIPC 的 `lite_train_lite_infer` 模式。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference 官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备)。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 Python 预测引擎推理的示例。您可以参考 [ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理)。

对于分类模型，可直接使用 `deploy/python/predict_cls.py`。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考 [服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md) 来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考 [基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md) 完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于 Paddle Serving 的介绍，可以参考 [Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考 [模型服务化部署](../../deployment/image_classification/paddle_serving.md) 来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考 [Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考 [端侧部署](../../deployment/image_classification/paddle_lite.md) 来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT、OpenVINO、MNN、TNN、NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考 [Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md) 来完成相应的部署工作。
