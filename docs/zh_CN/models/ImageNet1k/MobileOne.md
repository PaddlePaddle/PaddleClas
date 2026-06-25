# MobileOne 系列
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

[MobileOne](https://arxiv.org/abs/2206.04040) 是面向移动端低时延场景设计的卷积骨干网络。其核心思想是将训练阶段的多分支结构（卷积分支、尺度分支、恒等分支）在部署阶段重参数化为单分支卷积，从而兼顾训练表达能力和推理效率。

PaddleClas 当前已支持 `MobileOne_S0`、`MobileOne_S0_unfused`、`MobileOne_S1`、`MobileOne_S1_unfused`、`MobileOne_S2`、`MobileOne_S2_unfused`、`MobileOne_S3`、`MobileOne_S3_unfused`、`MobileOne_S4`、`MobileOne_S4_unfused` 十个变体，支持 unfused（多分支）和 fused（单分支）两种结构，分别对应不同的参数权重并提供对应的 ImageNet 训练配置。

<a name='1.2'></a>

### 1.2 模型指标

| Models                 | Top1    | Top5    | Reference<br>top1 | Reference<br>top5 |  FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| mobileOne_s0           | 0.7139  | 0.8983  | 0.7140             | -                 | 0.2789       | 2.0785        |
| mobileOne_s0_unfused   | 0.7138  | 0.8984  | 0.7140             | -                 | 1.0885       | 5.3815        |
| mobileOne_s1           | 0.7578  | 0.9275  | 0.7590             | -                 | 0.8314       | 4.7648        |
| mobileOne_s1_unfused   | 0.7581  | 0.9275  | 0.7590             | -                 | 0.8585       | 4.8935        |
| mobileOne_s2           | 0.7743  | 0.9359  | 0.7740             | -                 | 1.3061       | 7.8082        |
| mobileOne_s2_unfused   | 0.7743  | 0.9360  | 0.7740             | -                 | 1.3386       | 7.9717        |
| mobileOne_s3           | 0.7796  | 0.9385  | 0.7810             | -                 | 1.9050       | 10.0783       |
| mobileOne_s3_unfused   | 0.7797  | 0.9385  | 0.7810             | -                 | 1.9440       | 10.2753       |
| mobileOne_s4           | 0.7927  | 0.9440  | 0.7940             | -                 | 2.9912       | 14.8384       |
| mobileOne_s4_unfused   | 0.7926  | 0.9440  | 0.7940             | -                 | 3.0409       | 15.0790       |



**备注：**
1. Reference Top1 指标来自 MobileOne 原论文/官方仓库公开结果。
2. FLOPs/Params 基于 PaddleClas 当前实现在 `inference_mode=True` 下统计，输入分辨率为 `224x224`。
3. 当前文档暂未提供线上预训练模型与 inference 下载链接，可通过本地权重路径进行加载。

<a name="2"></a>

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/MobileOne/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。


示例：

```bash
# 训练

#fused
python3 -m paddle.distributed.launch  tools/train.py \
-c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=True

#unfused
python3 -m paddle.distributed.launch  tools/train.py \
-c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=False


# 评估（以本地权重为例）

#fused
python tools/eval.py -c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=True \
-o Global.pretrained_model=/path/to/mobileone_s0_paddle.pdparams

#unfused
python tools/eval.py -c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=False \
-o Global.pretrained_model=/path/to/mobileone_s0_unfused_paddle.pdparams


# 模型预测
#fused
python3 tools/infer.py \
-c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=True \
-o Global.pretrained_model=/path/to/mobileone_s0_paddle.pdparams

#unfused
python3 tools/infer.py \
-c PaddleClas/ppcls/configs/ImageNet/MobileOne/MobileOne_S0.yaml \
-o Arch.inference_mode=False \
-o Global.pretrained_model=/path/to/mobileone_s0_unfused_paddle.pdparams
```

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库，作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference 可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference 官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a>

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 Python 预测引擎推理的示例。您可以参考 [ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 。

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

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考 [Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md) 来完成相应的部署工作。
