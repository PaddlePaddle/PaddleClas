# Inception系列

## 概述
GoogleNet是2014年由Google设计的一种新的神经网络结构，其与VGG网络并列成为当年ImageNet挑战赛的双雄。GoogleNet设计了一种新的Inception结构，在网络中堆叠该结构使得网络层数达到了22层，这也是卷积网络首次超过20层的标志。由于在Inception结构中使用了1x1的卷积用于通道数降维，并且使用了Global-pooling代替传统的多fc层加工特征的方式，最终的GoogleNet网络的参数量和计算量远小于VGG网络，成为当时神经网络设计的一道亮丽风景线。

Xception 是 Google 继 Inception 后提出的对 Inception-v3 的另一种改进。在Xception中，作者使用了深度可分离卷积代替了传统的卷积操作，该操作大大节省了网络的参数量和计算量。最终相比InceptionV3，Xception的Flops大幅下降，精度反而有所提升。在DeeplabV3+中，作者将Xception做了进一步的改进，同时增加了Xception的层数，设计出了Xception65和Xception71的网络。

InceptionV4是2016年由Google设计的新的神经网络，作者认为Inception 结构可以用很低的计算成本达到很高的性能。而在传统的网络架构中引入残差结构效果也非常好。所以研究者将 Inception 结构和残差结构结合起来做了广泛的实验。最终，研究者通过实验明确地证实了，结合残差连接可以显著加速 Inception 的训练。也有一些证据表明残差 Inception 网络在相近的成本下略微超过没有残差连接的 Inception 网络。最终作者设计出的InceptionV4网络是包含了多个不同Inception块的模型，其在ImageNet上创造了新的精度。

该系列模型的FLOPS、参数量以及FP32预测耗时如下图所示。

![](../../images/models/Inception.png.flops.png)

![](../../images/models/Inception.png.params.png)

![](../../images/models/Inception.png.fp32.png)

上图反映了Xception系列和InceptionV4的精度和其他指标的关系，除了参数量外，InceptionV4模型依然比较有竞争力。在v100,FP16的情形下，InceptionV4的推理速度更具有优势。


## 精度、FLOPS和参数量

| Models             | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| GoogLeNet          | 0.707  | 0.897  | 0.698             |                   | 2.880        | 8.460             |
| Xception41         | 0.793  | 0.945  | 0.790             | 0.945             | 16.740       | 22.690            |
| Xception41<br>_deeplab | 0.796  | 0.944  |                   |                   | 18.160       | 26.730            |
| Xception65         | 0.810  | 0.955  |                   |                   | 25.950       | 35.480            |
| Xception65<br>_deeplab | 0.803  | 0.945  |                   |                   | 27.370       | 39.520            |
| Xception71         | 0.811  | 0.955  |                   |                   | 31.770       | 37.280            |
| InceptionV4        | 0.808  | 0.953  | 0.800             | 0.950             | 24.570       | 42.680            |



## FP32预测速度

| Models                 | Crop Size | Resize Short Size | Batch Size=1<br>(ms) |
|------------------------|-----------|-------------------|--------------------------|
| GoogLeNet              | 224       | 256               | 1.807                    |
| Xception41             | 299       | 320               | 3.972                    |
| Xception41<br>_deeplab | 299       | 320               | 4.408                    |
| Xception65             | 299       | 320               | 6.174                    |
| Xception65<br>_deeplab | 299       | 320               | 6.464                    |
| Xception71             | 299       | 320               | 6.782                    |
| InceptionV4            | 299       | 320               | 11.141                   |
