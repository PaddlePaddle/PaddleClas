# DeiT 系列
-----

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型指标](#1.2)
    - [1.3 Benchmark](#1.3)
      - [1.3.1 基于 V100 GPU 的预测速度](#1.3.1)
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

DeiT（Data-efficient Image Transformers）系列模型是由 FaceBook 在 2020 年底提出的，针对 ViT 模型需要大规模数据集训练的问题进行了改进，最终在 ImageNet 上取得了 83.1%的 Top1 精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在 ImageNet 数据集上可以达到 85.2% 的 Top1 精度。[论文地址](https://arxiv.org/abs/2012.12877)。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DeiT_tiny_patch16_224            | 0.718 | 0.910 | 0.722 | 0.911 | 1.07 | 5.68 |
| DeiT_small_patch16_224           | 0.796 | 0.949 | 0.799 | 0.950 | 4.24 | 21.97 |
| DeiT_base_patch16_224            | 0.817 | 0.957 | 0.818 | 0.956 | 16.85 | 86.42 |
| DeiT_base_patch16_384            | 0.830 | 0.962 | 0.829 | 0.972 | 49.35 | 86.42 |
| DeiT_tiny_distilled_patch16_224  | 0.741 | 0.918 | 0.745 | 0.919 | 1.08 | 5.87 |
| DeiT_small_distilled_patch16_224 | 0.809 | 0.953 | 0.812 | 0.954 | 4.26 | 22.36 |
| DeiT_base_distilled_patch16_224  | 0.831 | 0.964 | 0.834 | 0.965 | 16.93 | 87.18 |
| DeiT_base_distilled_patch16_384  | 0.851 | 0.973 | 0.852 | 0.972 | 49.43 | 87.18 |

**备注：** PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

| Models                               | Size | Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
| ------------------------------------ | ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| DeiT_tiny_<br>patch16_224            | 224               | 3.87                           | 3.58                           | 4.64                           |
| DeiT_small_<br>patch16_224           | 224               | 3.52                           | 5.90                           | 9.44                          |
| DeiT_base_<br>patch16_224            | 224               | 5.97                           | 15.52                          | 27.37                          |
| DeiT_base_<br>patch16_384            | 384               | 13.78                          | 45.94                          | 89.38                          |
| DeiT_tiny_<br>distilled_patch16_224  | 224               | 3.31                           | 3.61                           | 4.57                           |
| DeiT_small_<br>distilled_patch16_224 | 224               | 3.57                           | 5.91                           | 9.51                          |
| DeiT_base_<br>distilled_patch16_224  | 224               | 6.00                           | 15.43                          | 27.10                          |
| DeiT_base_<br>distilled_patch16_384  | 384               | 13.76                          | 45.61                          | 89.15                          |

**备注：** 精度类型为 FP32，推理过程使用 TensorRT-8.0.3.4。

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/DeiT/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

**备注：** 由于 DeiT 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。

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
