# SwinTransformerV2
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

SwinTransformerV2 在 SwinTransformer 的基础上进行改进，可处理大尺寸图像。通过提升模型容量与输入分辨率，SwinTransformerV2 在四个代表性基准数据集上取得了新记录。[论文地址](https://arxiv.org/abs/2111.09883)。

<a name='1.2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SwinTransformerV2_tiny_patch4_window8_256    | 0.8177 | 0.9588 | 0.818 | 0.959 | 4.3  | 21.9   |
| SwinTransformerV2_tiny_patch4_window16_256   | 0.8283 | 0.9623 | 0.828 | 0.962 | 4.4  | 21.9   |
| SwinTransformerV2_small_patch4_window8_256    | 0.8373 | 0.9662 | 0.837 | 0.966 | 8.4 | 37.9   |
| SwinTransformerV2_small_patch4_window16_256   | 0.8414 | 0.9681 | 0.841 | 0.968 | 8.5 | 37.9   |
| SwinTransformerV2_base_patch4_window8_256    | 0.8419 | 0.9687 | 0.842 | 0.969 | 15.0 | 67.0   |
| SwinTransformerV2_base_patch4_window16_256   | 0.8458 | 0.9706 | 0.846 | 0.970 | 15.1 | 67.0   |
| SwinTransformerV2_base_patch4_window24_384<sup>[1]</sup>   | 0.8714 | 0.9824 | 0.871 | 0.982 | 34.0 | 67.0   |
| SwinTransformerV2_large_patch4_window16_256<sup>[1]</sup>   | 0.8689 | 0.9804 | 0.869 | 0.980 | 33.8 | 149.6 |
| SwinTransformerV2_large_patch4_window24_384<sup>[1]</sup>  | 0.8747 | 0.9827 | 0.876 | 0.983 | 76.1 | 149.6 |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。

**备注：**
1. 与 Reference 的精度差异源于数据预处理不同。
2. PaddleClas 所提供的该系列模型的预训练模型权重，均是基于其官方提供的权重转得。

<a name='1.3'></a>

### 1.3 Benchmark

<a name='1.3.1'></a>

#### 1.3.1 基于 V100 GPU 的预测速度

敬请期待

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、SwinTransformerV2 在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/SwinTransformerV2/` 中提供了 SwinTransformerV2 的训练配置，可以通过如下脚本启动训练：此部分内容可以参考[ResNet50 模型训练、评估和预测](./ResNet.md#3)。

**备注：** 由于 SwinTransformer 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#4.1) 。

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
