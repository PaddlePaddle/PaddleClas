# DPN与DenseNet系列

## 概述
DenseNet是2017年CVPR best paper提出的一种新的网络结构，该网络设计了一种新的跨层连接的block，即dense-block。相比ResNet中的bottleneck，dense-block设计了一个更激进的密集连接机制，即互相连接所有的层，每个层都会接受其前面所有层作为其额外的输入。DenseNet将所有的dense-block堆叠，组合成了一个密集连接型网络。由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。
DPN的全称是Dual Path Networks，即双通道网络。该网络是由DenseNet和ResNeXt结合的一个网络，其证明了DenseNet能从靠前的层级中提取到新的特征，而ResNeXt本质上是对之前层级中已提取特征的复用。作者进一步分析发现，ResNeXt对特征有高复用率，但冗余度低，DenseNet能创造新特征，但冗余度高。结合二者结构的优势，作者设计了DPN网络。最终DPN网络在同样Params和Flops下，取得了比ResNeXt与DenseNet更好的结果。

该系列模型的FLOPS、参数量以及FP32预测耗时如下图所示。

![](../../images/models/DPN.png.flops.png)

![](../../images/models/DPN.png.params.png)

![](../../images/models/DPN.png.fp32.png)

目前paddleclas开源的这两类模型的预训练模型一共有10个，其指标如图所示，可以看到，在Flops和Params下，相比DenseNet，DPN拥有更高的精度。但是由于DPN有更多的分支，所以其推理速度要慢于DenseNet。由于DenseNet264的网络层数最深，所以该网络是DenseNet系列模型中参数量最大的网络，DenseNet161的网络的宽度最大，导致其是该系列中网络中计算量最大、精度最高的网络。从推理速度来看，计算量大且精度高的的DenseNet161比DenseNet264具有更快的速度，所以其比DenseNet264具有更大的优势。
DPN系列网络的曲线图中规中矩，模型的参数量和计算量越大，模型的精度越高。其中，由于DPN107的网络宽度最大，所以其是该系列网络中参数量与计算量最大的网络。

所有模型在预测时，图像的crop_size设置为224，resize_short_size设置为256。

## 精度、FLOPS和参数量

| Models      | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DenseNet121 | 0.757  | 0.926  | 0.750             |                   | 5.690        | 7.980             |
| DenseNet161 | 0.786  | 0.941  | 0.778             |                   | 15.490       | 28.680            |
| DenseNet169 | 0.768  | 0.933  | 0.764             |                   | 6.740        | 14.150            |
| DenseNet201 | 0.776  | 0.937  | 0.775             |                   | 8.610        | 20.010            |
| DenseNet264 | 0.780  | 0.939  | 0.779             |                   | 11.540       | 33.370            |
| DPN68       | 0.768  | 0.934  | 0.764             | 0.931             | 4.030        | 10.780            |
| DPN92       | 0.799  | 0.948  | 0.793             | 0.946             | 12.540       | 36.290            |
| DPN98       | 0.806  | 0.951  | 0.799             | 0.949             | 22.220       | 58.460            |
| DPN107      | 0.809  | 0.953  | 0.802             | 0.951             | 35.060       | 82.970            |
| DPN131      | 0.807  | 0.951  | 0.801             | 0.949             | 30.510       | 75.360            |




## FP32预测速度

| Models                               | Crop Size | Resize Short Size | Batch Size=1<br>(ms) |
|-------------|-----------|-------------------|--------------------------|
| DenseNet121 | 224       | 256               | 4.371                    |
| DenseNet161 | 224       | 256               | 8.863                    |
| DenseNet169 | 224       | 256               | 6.391                    |
| DenseNet201 | 224       | 256               | 8.173                    |
| DenseNet264 | 224       | 256               | 11.942                   |
| DPN68       | 224       | 256               | 11.805                   |
| DPN92       | 224       | 256               | 17.840                   |
| DPN98       | 224       | 256               | 21.057                   |
| DPN107      | 224       | 256               | 28.685                   |
| DPN131      | 224       | 256               | 28.083                   |
