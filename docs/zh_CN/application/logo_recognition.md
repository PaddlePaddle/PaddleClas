# Logo识别

 Logo识别技术，是现实生活中应用很广的一个领域，比如一张照片中是否出现了Adidas或者Nike的商标Logo，或者一个杯子上是否出现了星巴克或者可口可乐的商标Logo。通常Logo类别数量较多时，往往采用检测+识别两阶段方式，检测模块负责检测出潜在的Logo区域，根据检测区域抠图后输入识别模块进行识别。识别模块多采用检索的方式，根据查询图片和底库图片进行相似度排序获得预测类别。此文档主要对Logo图片的特征提取部分进行相关介绍。

## 1 算法介绍

算法整体流程，详见[特征学习](./feature_learning.md)整体流程。

整体设置详见: [ResNet50_ReID.yaml](../../../ppcls/configs/Logo/ResNet50_ReID.yaml)。

具体模块如下所示

### 1.1数据增强

与普通训练分类不同，此部分主要使用如下图像增强方式：

- 图像`Resize`到224。对于Logo而言，使用的图像，直接为检测器crop之后的图像，因此直接resize到224
- [AugMix](https://arxiv.org/abs/1912.02781v1)：模拟Logo图像形变变化等实际场景
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)：模拟遮挡等实际情况

### 1.2 Backbone的具体设置

使用`ResNet50`作为backbone，同时做了如下修改：

 - last stage stride=1, 保持最后输出特征图尺寸14x14。计算量增加较小，但显著提高模型特征提取能力


具体代码：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

### 1.3 Neck部分

为了降低inferecne时计算特征距离的复杂度，添加一个embedding 卷积层，特征维度为512。

### 1.4 Metric Learning相关Loss的设置

在Logo识别中，使用了[Pairwise Cosface + CircleMargin](https://arxiv.org/abs/2002.10857) 联合训练，其中权重比例为1:1

具体代码详见：[PairwiseCosface](../../../ppcls/loss/pairwisecosface.py) 、[CircleMargin](../../../ppcls/arch/gears/circlemargin.py)

## 2 实验结果

<img src="../../images/logo/logodet3k.jpg" style="zoom:50%;" />

使用LogoDet-3K数据集进行实验，此数据集是具有完整标注的Logo数据集，有3000个标识类别，约20万个高质量的人工标注的标识对象和158652张图片。相关数据介绍参考[原论文](https://arxiv.org/abs/2008.05359)

由于原始的数据集中，图像包含标注的检测框，在识别阶段只考虑检测器抠图后的logo区域，因此采用原始的标注框抠出Logo区域图像构成训练集，排除背景在识别阶段的影响。对数据集进行划分，产生155427张训练集，覆盖3000个logo类别（同时作为测试时gallery图库），3225张测试集，用于作为查询集。抠图后的训练集可[在此下载](https://arxiv.org/abs/2008.05359)

在此数据集上，recall1 达到89.8%。
