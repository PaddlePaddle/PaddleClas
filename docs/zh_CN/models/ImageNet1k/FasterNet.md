# FasterNet 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
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

[FasterNet](https://arxiv.org/pdf/2303.03667) 是一个旨在提高运行速度的神经网络，改进点主要如下：
1. 重新审视了流行的运算符，发现低 FLOPS 主要是由于运算符频繁的内存访问，特别是深度卷积；
2. 提出了部分卷积 (PConv)，通过减少冗余计算和内存访问来更高效地提取空间特征；
3. 基于 PConv 推出了 FasterNet 系列模型，这是一种新的神经网络家族，在不影响各种视觉任务准确性的情况下，在各种设备上实现了显著更高的运行速度。

结果表明在 ImageNet-1K 数据集上，小型的 FasterNet-T0 在 GPU、CPU 和 ARM 处理器上分别比 MobileViT-XX S快 2.8 倍、3.3 倍和 2.4 倍，同时在准确性上提高了 2.9%。大型的FasterNet-L取得了 83.5% 的 Top-1 准确性，与新兴的 Swin-B 相当，同时在GPU上的推理吞吐量提高了 36%，并在 CPU 上节省了 37% 的计算时间。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1    | Top5    | Reference<br>top1 | Reference<br>top5 |  FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FasterNet_T0     | 0.719   | 0.909   | 0.719             |                   | 0.34          | 3.9           |
| FasterNet_T1     | 0.759   | 0.932   | 0.762             |                   | 0.85          | 7.6           |
| FasterNet_T2     | 0.791   | 0.948   | 0.789             |                   | 1.90          | 15.0          |
| FasterNet_S      | 0.813   | 0.952   | 0.813             |                   | 4.55          | 31.1          |  
| FasterNet_M      | 0.830   | 0.956   | 0.830             |                   | 8.72          | 53.5          |
| FasterNet_L      | 0.835   | 0.964   | 0.835             |                   | 15.49         | 93.4          |

**备注：** PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/FasterNet/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 。

<a name="4.3"></a>

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../../deployment/image_classification/cpp/windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a>

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/image_classification/paddle_serving.md)来完成相应的部署工作。

<a name="4.5"></a>

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

<a name="4.6"></a>

### 4.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。
