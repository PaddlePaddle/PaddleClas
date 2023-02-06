# RepVGG 系列
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

RepVGG(Making VGG-style ConvNets Great Again)系列模型是由清华大学(丁贵广团队)、旷视科技(孙剑等人)、港科大和阿伯里斯特威斯大学在 2021 年提出的一个简单但强大的卷积神经网络架构，该架构具有类似于 VGG 的推理时间主体，该主体仅由 3x3 卷积和 ReLU 的堆栈组成，而训练时间模型具有多分支拓扑。训练时间和推理时间架构的这种解耦是通过结构重新参数化(re-parameterization)技术实现的，因此该模型称为 RepVGG。[论文地址](https://arxiv.org/abs/2101.03697)。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| RepVGG_A0 | 0.7131 | 0.9016 | 0.7241   | - | - | - |
| RepVGG_A1 | 0.7380 | 0.9146 | 0.7446   | - | - | - |
| RepVGG_A2 | 0.7571 | 0.9264 | 0.7648   | - | - | - |
| RepVGG_B0 | 0.7450 | 0.9213 | 0.7514   | - | - | - |
| RepVGG_B1 | 0.7773 | 0.9385 | 0.7837   | - | - | - |
| RepVGG_B2 | 0.7813 | 0.9410 | 0.7878   | - | - | - |
| RepVGG_B1g2 | 0.7732 | 0.9359 | 0.7778 | - | - | - |
| RepVGG_B1g4 | 0.7675 | 0.9335 | 0.7758 | - | - | - |
| RepVGG_B2g4 | 0.7881 | 0.9448 | 0.7938 | - | - | - |
| RepVGG_B3 | 0.8031 | 0.9517 | 0.8052 | - | - | - |
| RepVGG_B3g4 | 0.8005 | 0.9502 | 0.8021 | - | - | - |
| RepVGG_D2se | 0.8339 | 0.9665 | 0.8355 | - | - | - |

关于 Params、FLOPs、Inference speed 等信息，敬请期待。

**备注：** PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models     | Size  | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
| -------- | --------- | ----------------- | ------------------------------ | ------------------------------ |
| RepVGG_A0   | 224       | 1.38                           | 1.85                           | 2.81                          |
| RepVGG_A1 | 224       | 1.68                          | 2.33                          | 3.70                          |
| RepVGG_A2  | 224       | 2.31                          | 4.46                          | 6.53                          |
| RepVGG_B0   | 224       | 1.99                           | 2.87                          | 4.67                          |
| RepVGG_B1    | 224       | 3.56                           | 7.64                           | 13.94                           |
| RepVGG_B2  | 224       | 4.45                           | 9.79                           | 19.13                           |
| RepVGG_B1g2    | 224       | 4.18                           | 6.93                           | 11.99                           |
| RepVGG_B1g4 | 224       | 4.73                           | 7.23                           | 11.14                           |
| RepVGG_B2g4   | 224       | 5.47                           | 8.94                           | 14.73                          |
| RepVGG_B3  | 224       | 4.28                           | 11.64                           | 21.14                           |
| RepVGG_B3g4 | 224       | 4.21                           | 8.22                           | 14.68                           |
| RepVGG_D2se   | 224       | -                           | -                           | -                          |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/RepVGG/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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
