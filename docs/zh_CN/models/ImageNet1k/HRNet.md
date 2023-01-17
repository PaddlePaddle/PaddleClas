# HRNet 系列
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

HRNet 是 2019 年由微软亚洲研究院提出的一种全新的神经网络，不同于以往的卷积神经网络，该网络在网络深层仍然可以保持高分辨率，因此预测的关键点热图更准确，在空间上也更精确。此外，该网络在对分辨率敏感的其他视觉任务中，如检测、分割等，表现尤为优异。

该系列模型的 FLOPs、参数量以及 T4 GPU 上的预测耗时如下图所示。

![](../../../images/models/T4_benchmark/t4.fp32.bs4.HRNet.flops.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs4.HRNet.params.png)

![](../../../images/models/T4_benchmark/t4.fp32.bs4.HRNet.png)

![](../../../images/models/T4_benchmark/t4.fp16.bs4.HRNet.png)

目前 PaddleClas 开源的这类模型的预训练模型一共有 7 个，其指标如图所示，其中 HRNet_W48_C 指标精度异常的原因可能是因为网络训练的正常波动。

<a name='1.2'></a>

### 1.2 模型指标

| Models      | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNet_W18_C | 0.769  | 0.934  | 0.768             | 0.934             | 4.140        | 21.290            |
| HRNet_W18_C_ssld | 0.816  | 0.958  | 0.768             | 0.934             | 4.140        | 21.290            |
| HRNet_W30_C | 0.780  | 0.940  | 0.782             | 0.942             | 16.230       | 37.710            |
| HRNet_W32_C | 0.783  | 0.942  | 0.785             | 0.942             | 17.860       | 41.230            |
| HRNet_W40_C | 0.788  | 0.945  | 0.789             | 0.945             | 25.410       | 57.550            |
| HRNet_W44_C | 0.790  | 0.945  | 0.789             | 0.944             | 29.790       | 67.060            |
| HRNet_W48_C | 0.790  | 0.944  | 0.793             | 0.945             | 34.580       | 77.470            |
| HRNet_W48_C_ssld | 0.836  | 0.968  | 0.793             | 0.945             | 34.580       | 77.470            |
| HRNet_W64_C | 0.793  | 0.946  | 0.795             | 0.946             | 57.830       | 128.060           |
| SE_HRNet_W64_C_ssld | 0.847  | 0.973  |                |                   | 57.830       | 128.970           |

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models      | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|-------------|-----------|-------------------|-------------------|-------------------|
| HRNet_W18_C | 224       | 6.33               | 8.12               | 10.91              |
| HRNet_W18_C_ssld | 224  | 6.33               | 8.12               | 10.91              |
| HRNet_W30_C | 224       | 8.34               | 10.65              | 13.95              |
| HRNet_W32_C | 224       | 8.03               | 10.46              | 14.11              |
| HRNet_W40_C | 224       | 9.64              | 14.27             | 19.54             |
| HRNet_W44_C | 224       | 10.54             | 15.41             | 24.50             |
| HRNet_W48_C | 224       | 10.81             | 15.67             | 15.53             |
| HRNet_W48_C_ssld | 224  | 10.81                          | 15.67                          | 15.53                          |
| HRNet_W64_C | 224       | 13.12             | 19.49             | 33.80             |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name='1.3.2'></a>

#### 1.3.2 基于 T4 GPU 的预测速度

| Models            | Size | Latency(ms)<br>FP16<br>bs=1 | Latency(ms)<br>FP16<br>bs=4 | Latency(ms)<br>FP16<br>bs=8 | Latency(ms)<br>FP32<br>bs=1 | Latency(ms)<br>FP32<br>bs=4 | Latency(ms)<br>FP32<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNet_W18_C | 224       | 6.79093                      | 11.50986                     | 17.67244                     | 7.40636                     | 13.29752                     | 23.33445                     |
| HRNet_W18_C_ssld | 224  | 6.79093                      | 11.50986                     | 17.67244                     | 7.40636                     | 13.29752                     | 23.33445                     |
| HRNet_W30_C | 224       | 8.98077                      | 14.08082                     | 21.23527                     | 9.57594                     | 17.35485                     | 32.6933                      |
| HRNet_W32_C | 224       | 8.82415                      | 14.21462                     | 21.19804                     | 9.49807                     | 17.72921                     | 32.96305                     |
| HRNet_W40_C | 224       | 11.4229                      | 19.1595                      | 30.47984                     | 12.12202                     | 25.68184                     | 48.90623                     |
| HRNet_W44_C | 224       | 12.25778                     | 22.75456                     | 32.61275                     | 13.19858                     | 32.25202                     | 59.09871                     |
| HRNet_W48_C | 224       | 12.65015                     | 23.12886                     | 33.37859                     | 13.70761                     | 34.43572                     | 63.01219                     |
| HRNet_W48_C_ssld | 224  | 12.65015                     | 23.12886                     | 33.37859                     | 13.70761                     | 34.43572                     | 63.01219                     |
| HRNet_W64_C | 224       | 15.10428                     | 27.68901                     | 40.4198                      | 17.57527                     | 47.9533                      | 97.11228                     |
| SE_HRNet_W64_C_ssld | 224    |           32.33651           |          69.31189            |           116.07245            |                   31.69770   |           94.99546            |             174.45766        |

**备注：** 推理过程使用 TensorRT。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/HRNet/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

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
