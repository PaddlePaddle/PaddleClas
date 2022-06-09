# SwinTransformer

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

### 1.1 模型简介

Swin Transformer 是一种新的视觉 Transformer 网络，可以用作计算机视觉领域的通用骨干网路。SwinTransformer 由移动窗口（shifted windows）表示的层次 Transformer 结构组成。移动窗口将自注意计算限制在非重叠的局部窗口上，同时允许跨窗口连接，从而提高了网络性能。[论文地址](https://arxiv.org/abs/2103.14030)。

<a name='2'></a>

### 1.2 模型指标

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SwinTransformer_tiny_patch4_window7_224    | 0.8069 | 0.9534 | 0.812 | 0.955 | 4.5  | 28   |
| SwinTransformer_small_patch4_window7_224   | 0.8275 | 0.9613 | 0.832 | 0.962 | 8.7  | 50   |
| SwinTransformer_base_patch4_window7_224    | 0.8300 | 0.9626 | 0.835 | 0.965 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384   | 0.8439 | 0.9693 | 0.845 | 0.970 | 47.1 | 88   |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>    | 0.8487 | 0.9746 | 0.852 | 0.975 | 15.4 | 88   |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>   | 0.8642 | 0.9807 | 0.864 | 0.980 | 47.1 | 88   |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>   | 0.8596 | 0.9783 | 0.863 | 0.979 | 34.5 | 197 |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup>  | 0.8719 | 0.9823 | 0.873 | 0.982 | 103.9 | 197 |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。

**注**：与 Reference 的精度差异源于数据预处理不同。

<a name='3'></a>

### 1.3 Benchmark

#### 1.3.1 基于 V100 GPU 的预测速度

| Models  | Size |  Latency(ms)<br>bs=1 | Latency(ms)<br>bs=4 | Latency(ms)<br>bs=8 |
|:--:|:--:|:--:|:--:|:--:|
| SwinTransformer_tiny_patch4_window7_224                 | 224       | 6.59                           | 9.68                           | 16.32                          |
| SwinTransformer_small_patch4_window7_224                | 224       | 12.54                          | 17.07                          | 28.08                          |
| SwinTransformer_base_patch4_window7_224                 | 224       | 13.37                          | 23.53                          | 39.11                          |
| SwinTransformer_base_patch4_window12_384                | 384       | 19.52                          | 64.56                          | 123.30                         |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>   | 224       | 13.53                          | 23.46                          | 39.13                          |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>  | 384       | 19.65                          | 64.72                          | 123.42                         |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>  | 224       | 15.74                          | 38.57                          | 71.49                          |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup> | 384       | 32.61                          | 116.59                         | 223.23                         |

[1]：基于 ImageNet22k 数据集预训练，然后在 ImageNet1k 数据集迁移学习得到。

**备注：** 精度类型为 FP32，推理过程使用 TensorRT。


<a name="2"></a>   
    
## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a> 
    
## 3. 模型训练、评估和预测


此部分内容包括训练环境配置、ImageNet数据的准备、SwinTransformer 在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/SwinTransformer/` 中提供了 SwinTransformer 的训练配置，可以通过如下脚本启动训练：此部分内容可以参考[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

**备注：** 由于 SwinTransformer 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。

    
<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a> 

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。
    
Inference 的获取可以参考 [ResNet50 推理模型准备](./ResNet.md#41-推理模型准备) 。

<a name="4.2"></a> 

### 4.2 基于 Python 预测引擎推理

PaddleClas 提供了基于 python 预测引擎推理的示例。您可以参考[ResNet50 基于 Python 预测引擎推理](./ResNet.md#42-基于-python-预测引擎推理) 对 SwinTransformer 完成推理预测。

<a name="4.3"></a> 

### 4.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考[基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md)完成相应的预测库编译和模型预测工作。

<a name="4.4"></a> 

### 4.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考[Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。
    
PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

<a name="4.5"></a> 

### 4.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考[Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。
    
PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

<a name="4.6"></a> 

### 4.6 Paddle2ONNX 模型转换与预测
    
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考[Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](@shuilong)来完成相应的部署工作。

