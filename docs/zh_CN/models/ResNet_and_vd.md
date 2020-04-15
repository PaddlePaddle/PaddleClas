# ResNet及其Vd系列

## 概述

ResNet系列模型是在2015年提出的，一举在ILSVRC2015比赛中取得冠军，top5错误率为3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了ResNet网络。实验表明使用残差块可以有效地提升收敛速度和精度。

斯坦福大学的Joyce Xu将ResNet称为「真正重新定义了我们看待神经网络的方式」的三大架构之一。由于ResNet卓越的性能，越来越多的来自学术界和工业界学者和工程师对其结构进行了改进，比较出名的有Wide-ResNet, ResNet-vc ,ResNet-vd, Res2Net等，其中ResNet-vc与ResNet-vd的参数量和计算量与ResNet几乎一致，所以在此我们将其与ResNet统一归为ResNet系列。

本次发布ResNet系列的模型包括ResNet50，ResNet50_vd，ResNet50_vd_ssld，ResNet200_vd等14个预训练模型。在训练层面上，ResNet的模型采用了训练ImageNet的标准训练流程，而其余改进版模型采用了更多的训练策略，如learning rate的下降方式采用了cosine decay，引入了label smoothing的标签正则方式，在数据预处理加入了mixup的操作，迭代总轮数从120个epoch增加到200个epoch。

其中，ResNet50_vd_v2与ResNet50_vd_ssld采用了知识蒸馏，保证模型结构不变的情况下，进一步提升了模型的精度，具体地，ResNet50_vd_v2的teacher模型是ResNet152_vd（top1准确率80.59%），数据选用的是ImageNet-1k的训练集，ResNet50_vd_ssld的teacher模型是ResNeXt101_32x16d_wsl（top1准确率84.2%），数据选用结合了ImageNet-1k的训练集和ImageNet-22k挖掘的400万数据。知识蒸馏的具体方法正在持续更新中。


该系列模型的FLOPS、参数量以及FP32预测耗时如下图所示。

![](../../images/models/ResNet.png.flops.png)

![](../../images/models/ResNet.png.params.png)

![](../../images/models/ResNet.png.fp32.png)

通过上述曲线可以看出，层数越多，准确率越高，但是相应的参数量、计算量和延时都会增加。ResNet50_vd_ssld通过用更强的teacher和更多的数据，将其在ImageNet-1k上的验证集top-1精度进一步提高，达到了82.39%，刷新了ResNet50系列模型的精度。

**注意**：所有模型在预测时，图像的crop_size设置为224，resize_short_size设置为256。

## 精度、FLOPS和参数量

| Models           | Top1 | Top5 | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet18         | 0.710           | 0.899           | 0.696                    | 0.891                    | 3.660     | 11.690    |
| ResNet18_vd      | 0.723           | 0.908           |                          |                          | 4.140     | 11.710    |
| ResNet34         | 0.746           | 0.921           | 0.732                    | 0.913                    | 7.360     | 21.800    |
| ResNet34_vd      | 0.760           | 0.930           |                          |                          | 7.390     | 21.820    |
| ResNet50         | 0.765           | 0.930           | 0.760                    | 0.930                    | 8.190     | 25.560    |
| ResNet50_vc      | 0.784           | 0.940           |                          |                          | 8.670     | 25.580    |
| ResNet50_vd      | 0.791           | 0.944           | 0.792                    | 0.946                    | 8.670     | 25.580    |
| ResNet50_vd_v2   | 0.798           | 0.949           |                          |                          | 8.670     | 25.580    |
| ResNet101        | 0.776           | 0.936           | 0.776                    | 0.938                    | 15.520    | 44.550    |
| ResNet101_vd     | 0.802           | 0.950           |                          |                          | 16.100    | 44.570    |
| ResNet152        | 0.783           | 0.940           | 0.778                    | 0.938                    | 23.050    | 60.190    |
| ResNet152_vd     | 0.806           | 0.953           |                          |                          | 23.530    | 60.210    |
| ResNet200_vd     | 0.809           | 0.953           |                          |                          | 30.530    | 74.740    |
| ResNet50_vd_ssld | 0.824           | 0.961           |                          |                          | 8.670     | 25.580    |
| ResNet101_vd_ssld | 0.837           | 0.967           |                          |                          | 16.100    | 44.570     |




## FP32预测速度

| Models                 | Crop Size | Resize Short Size | Batch Size=1<br>(ms) |
|------------------|-----------|-------------------|--------------------------|
| ResNet18         | 224       | 256               | 1.499                    |
| ResNet18_vd      | 224       | 256               | 1.603                    |
| ResNet34         | 224       | 256               | 2.272                    |
| ResNet34_vd      | 224       | 256               | 2.343                    |
| ResNet50         | 224       | 256               | 2.939                    |
| ResNet50_vc      | 224       | 256               | 3.041                    |
| ResNet50_vd      | 224       | 256               | 3.165                    |
| ResNet50_vd_v2   | 224       | 256               | 3.165                    |
| ResNet101        | 224       | 256               | 5.314                    |
| ResNet101_vd     | 224       | 256               | 5.252                    |
| ResNet152        | 224       | 256               | 7.205                    |
| ResNet152_vd     | 224       | 256               | 7.200                    |
| ResNet200_vd     | 224       | 256               | 8.885                    |
| ResNet50_vd_ssld | 224       | 256               | 3.165                    |
| ResNet101_vd_ssld  | 224       | 256             | 5.252                  |
