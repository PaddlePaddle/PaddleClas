# DSNet
---

## 目录

- [1. 模型介绍](#1)
    - [1.1 模型简介](#1.1)
    - [1.2 模型细节](#1.2)
      - [1.2.1 尺度内传播模块](#1.2.1)
      - [1.2.2 尺度间对齐模块](#1.2.2)
    - [1.3 实验结果](#1.3)
- [2. 模型快速体验](#2)
- [3. 模型训练、评估和预测](#3)
- [4. 模型推理部署](#4)
  - [4.1 推理模型准备](#4.1)
  - [4.2 基于 Python 预测引擎推理](#4.2)
  - [4.3 基于 C++ 预测引擎推理](#4.3)
  - [4.4 服务化部署](#4.4)
  - [4.5 端侧部署](#4.5)
  - [4.6 Paddle2ONNX 模型转换与预测](#4.6)
- [5. 引用](#5)

<a name="1"></a>

## 1. 模型介绍

### 1.1 模型简介

​        具有优越的全局表示能力的Transformer在视觉任务中取得了具有竞争性的结果。但是，这些基于Transformer的模型未能考虑输入图像中的细粒度更高的局部信息。现有的一些工作，如ContNet、CrossViT、CvT、PVT等尝试以不同的方式将卷积引入到Transformer中，从而兼顾局部特征与图片全局信息。然而，这些方法要么依次执行卷积和注意力机制，要么只是在注意力机制中用卷积投影代替线性投影，且关注局部模式的卷积操作与关注全局模式的注意力机制可能会在训练过程中产生冲突，妨碍了两者的优点合并。

​        本文提出了一种通用的双流网络（Dual-Stream Network, DSNet），以充分探索用于图像分类的局部和全局模式特征的表示能力。DSNet可以并行地同时计算细粒度的局部特征和集成的全局特征，并有效地对两种特征进行融合。具体而言，我们提出了一个尺度内传播模块（intra-scale propagation）来处理每个DS-Block中的具有两种不同分辨率的局部特征和全局特征，以及一个尺度间对齐模块（inter-scale alignment）来在双尺度上跨特征执行信息交互，并将局部特征与全局特征进行融合。我们所提出的DSNet在ImageNet-1k上的top-1 accuracy比DeiT-Small精度高出2.4%，并与其他Vision Transformers和ResNet模型相比实现了最先进的性能。在目标检测和实例分割任务上，利用DSNet-Small作为骨干网络的模型，在MSCOCO 2017数据集上的mAP分别比利用ResNet-50作为骨干网络的模型高出6.4%和5.5%，并超过了之前SOTA水平，从而证明了其作为视觉任务通用骨干的潜力。

​        论文地址：https://arxiv.org/abs/2105.14734v4。

<a name="1.2"></a>

### 1.2 模型细节

网络整体结构如下图所示。

<div align="center">
  <img width="700" alt="DSNet" src="https://user-images.githubusercontent.com/71830213/207497958-f9802c03-3eec-4ba5-812f-c6a9158856c1.png">
</div>

​        受ResNet等网络分stage对网络结构进行设计的启发，我们在DSNet中同样设置了4个stage，对应原图的下采样倍数分别为4、8、16、32。每个stage均采用若干个DS-Block来生成并组合双尺度（即，卷积操作和注意力机制）的特征表示。我们的关键思想是以相对较高的分辨率来表征局部特征以保留细节，而以较低的分辨率（图像大小的$\frac{1}{32}$）[^1]表示全局特征以保留全局图案。具体来说，在每个DS-Block中，我们将输入特征图在通道维度上分成两部分。其中一部分用于提取局部特征${f}_ {l}$，另一部分用于汇总全局特征${f}_ {g}$。其中每个stage的所有DS-Block中保持${f}_ {g}$的大小不变。接下来对DS-Block中的尺度内传播模块和尺度间对齐模块进行介绍。

<a name="1.2.1"></a>

#### 1.2.1 尺度内传播模块

​        对于高分辨率的${f}_ {l}$，我们采用${3} \times {3}$的depth-wise卷积来提取局部特征，从而得到特征图${f}_ {L}$，即
$$
{f}_ {L} = \sum^{M, N}_ {m, n} W \left( m, n \right) \odot {f}_ {l} \left( i+m, j+n \right),
$$
其中$W ( m, n ), ( m, n ) \in \{ -1, 0, 1 \}$表示卷积核，$\odot$表示元素级相乘。而对于低分辨率的${f}_ {g}$，我们首先将其展平为长度为${l}_ {g}$的序列，从而将序列中每一个向量视为一个visual token，然后通过self-attention机制得到特征图为
$$
{f}_ {G} = \text{softmax} \left( \frac{{f}_ {Q} {f}_ {K}^{T}}{\sqrt{d}} \right) {f}_ {V},
$$
其中${f}_ {Q} = {W}_ {Q} {f}_ {g}, {f}_ {K} = {W}_ {K} {f}_ {g}, {f}_ {V} = {W}_ {V} {f}_ {g}$。这样的双流架构在两条路径中解耦了细粒度和全局的特征，显著消除了训练过程中的两者的冲突，最大限度地发挥局部和全局特征的优势。

<a name="1.2.2"></a>

#### 1.2.2 尺度间对齐模块

​        双尺度表示的适当融合对于DSNet的成功至关重要，因为它们捕捉了一幅图像的两个不同视角。为了解决这个问题，我们提出了一种新的基于co-attention的尺度间对齐模块。该模块旨在捕获每个局部-全局token对之间的相互关联，并以可学习和动态的方式双向传播信息，从而促使局部特征自适应地探索它们与全局信息的关系，从而使它们更具代表性和信息性，反之亦然。具体地，对于尺度内传播模块计算所得的${f}_ {L}, {f}_ {G}$，我们分别计算它们对应的query、key和value为：
$$
{Q}_ {L} = {f}_ {L} {W}_ {Q}^{l}, {K}_ {L} = {f}_ {L} {W}_ {K}^{l}, {V}_ {L} = {f}_ {L} {W}_ {V}^{l},\\
{Q}_ {G} = {f}_ {G} {W}_ {Q}^{g}, {K}_ {G} = {f}_ {G} {W}_ {K}^{g}, {V}_ {G} = {f}_ {G} {W}_ {V}^{g},
$$
从而计算得到从全局特征到局部特征、从局部特征到全局特征的注意力权重为：
$$
{W}_ { G \rightarrow L} = \text{softmax} \left( \frac{ {Q}_ {L} {K}_ {G}^{T} }{ \sqrt{d} } \right), {W}_ { L \rightarrow  G } = \text{softmax} \left( \frac{ {Q}_ {G} {K}_ {L}^{T} }{ \sqrt{d} } \right)
$$
从而得到混合的特征为：
$$
{h}_ {L} = {W}_ { G \rightarrow L } {V}_ {G}, {h}_ {G} = {W}_ { L \rightarrow G } {V}_ {L},
$$
这种双向信息流能够识别本地和全局token之间的跨尺度关系，通过这种关系，双尺度特征高度对齐并相互耦合。在这之后，我们对低分辨率表示${h}_ {G}$进行上采样，将其与高分辨率${h}_ {L}$拼接起来，并执行${1} \times {1}$卷积，以实现信道级双尺度信息融合。

<a name="1.3"></a>

### 1.3 实验结果

|   Models    | Top1  | Top5  | Reference<br>top1 | Reference<br>top5 | FLOPs<br>(G) | Params<br>(M) |
| :---------: | :---: | :---: | :---------------: | :---------------: | :----------: | :-----------: |
| DSNet-tiny  | 0.792 | 0.948 |       0.790       |         -         |     1.8      |     10.5      |
| DSNet-small | 0.820 | 0.960 |       0.823       |         -         |     3.5      |     23.0      |
| DSNet-base  | 0.818 | 0.952 |       0.831       |         -         |     8.4      |     49.3      |

<a name="2"></a>  

## 2. 模型快速体验

安装 paddlepaddle 和 paddleclas 即可快速对图片进行预测，体验方法可以参考[ResNet50 模型快速体验](./ResNet.md#2-模型快速体验)。

<a name="3"></a>

## 3. 模型训练、评估和预测

此部分内容包括训练环境配置、ImageNet数据的准备、该模型在 ImageNet 上的训练、评估、预测等内容。在 `ppcls/configs/ImageNet/DSNet/` 中提供了该模型的训练配置，启动训练方法可以参考：[ResNet50 模型训练、评估和预测](./ResNet.md#3-模型训练评估和预测)。

**备注：** 由于 DSNet 系列模型默认使用的 GPU 数量为 8 个，所以在训练时，需要指定8个GPU，如`python3 -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" tools/train.py -c xxx.yaml`, 如果使用 4 个 GPU 训练，默认学习率需要减小一半，精度可能有损。

<a name="4"></a>

## 4. 模型推理部署

<a name="4.1"></a>

### 4.1 推理模型准备

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

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

<a name="5"></a>

## 5. 引用

如果你的论文用到了 DSNet 的方法，请添加如下 cite：
```
@article{mao2021dual,
  title={Dual-stream network for visual recognition},
  author={Mao, Mingyuan and Zhang, Renrui and Zheng, Honghui and Ma, Teli and Peng, Yan and Ding, Errui and Zhang, Baochang and Han, Shumin and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={25346--25358},
  year={2021}
}
```

[^1]:若公式无法正常显示，请打开谷歌浏览器前往[此链接](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)安装MathJax插件，安装完毕后用谷歌浏览器重新打开此页面并刷新即可。
