# ResNeSt 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
      - [1.3.2 基于 T4 GPU 的预测速度](#1.3.2)
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

ResNeSt 系列模型是在 2020 年提出的，在原有的 resnet 网络结构上做了改进，通过引入 K 个 Group 和在不同 Group 中加入类似于 SEBlock 的 attention 模块，使得精度相比于基础模型 ResNet 有了大幅度的提高，且参数量和 flops 与基础的 ResNet 基本保持一致。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50_fast_1s1x64d        | 0.8061 | 0.9527|  0.8035 |            -| 5.40     | 26.3   |
| ResNeSt50        | 0.8102 | 0.9546|  0.8103 |            -| 5.40     | 27.5   |
| ResNeSt101        | 0.8279 | 0.9642|  0.8283 |            -| 10.25     | 48.4   |
| ResNeSt200        | 0.8418 | 0.9698|  0.8384 |            -| 17.50     | 70.4   |
| ResNeSt269        | 0.8444 | 0.9698|  0.8454 |            -| 22.54     | 111.2   |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models      | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
| ---------------------- | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| ResNeSt50_fast_1s1x64d | 224       | 2.43                           | 4.34                           | 6.96                           |
| ResNeSt50              | 224       | 13.08                           | 16.38                          | 23.18                          |
| ResNeSt101 | 256       | 19.16                           | 22.62                           | 32.59                           |
| ResNeSt200              | 224       | -                           | -                          | -                          |
| ResNeSt269 | 224       | -                           | -                           | -                           |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度
| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNeSt50_fast_1s1x64d | 224   | 3.46466           | 5.56647           | 9.11848          | 3.45405      |   8.72680    |    15.48710     |
| ResNeSt50              | 224   | 7.05851           | 8.97676            | 13.34704          | 6.16248      |   12.0633    |    21.49936     |

**备注：** 推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/ResNeSt/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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
