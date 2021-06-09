# Logo识别

 Logo识别技术，是现实生活中应用很广的一个领域，比如一张照片中是否出现了Adidas或者Nike的商标Logo，或者一个杯子上是否出现了星巴克或者可口可乐的商标Logo。通常Logo类别数量较多时，往往采用检测+识别两阶段方式，检测模块负责检测出潜在的Logo区域，根据检测区域抠图后输入识别模块进行识别。识别模块多采用检索的方式，根据查询图片和底库图片进行相似度排序获得预测类别。此文档主要对Logo图片的特征提取部分进行相关介绍，内容包括：

-  数据集及预处理方式
-  Backbone的具体设置
-  Loss函数的相关设置

## 数据集及预处理

### LogoDet-3K数据集

<img src="../../images/logo/logodet3k.jpg" alt="logodet3k" style="zoom:20%;" />

LogoDet-3K数据集是具有完整标注的Logo数据集，有3000个标识类别，约20万个高质量的人工标注的标识对象和158652张图片。相关数据介绍参考[原论文](https://arxiv.org/abs/2008.05359)

### 数据预处理

由于原始的数据集中，图像包含标注的检测框，在识别阶段只考虑检测器抠图后的logo区域，因此采用原始的标注框抠出Logo区域图像构成训练集，排除背景在识别阶段的影响。对数据集进行划分，产生155427张训练集，覆盖3000个logo类别（同时作为测试时gallery图库），3225张测试集，用于作为查询集。抠图后的训练集可[在此下载](https://arxiv.org/abs/2008.05359) 

在训练阶段，采用如下的数据增强方式，按照顺序如下：

- 图像`Resize`到224 
- 随机水平翻转
- [AugMix](https://arxiv.org/abs/1912.02781v1)
- Normlize：归一化到0～1
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)

## Backbone的设置

具体是用`ResNet50`作为backbone，但在`ResNet50`基础上做了如下修改：

- 使用ImageNet预训练模型
- last stage stride=1, 保持最后输出特征图尺寸14x14
- 在最后加入一个embedding 卷积层，特征维度为512

## Loss的设置

在Logo识别中，使用了[Pairwise Cosface + CircleMargin](https://arxiv.org/abs/2002.10857) 联合训练，其中权重比例为1:1

具体代码详见：[PairwiseCosface](../../../ppcls/loss/pairwisecosface.py) 、[CircleMargin](../../../ppcls/arch/gears/circlemargin.py)



全部的超参数及具体配置：[ResNet50_ReID.yaml](../../../ppcls/configs/Logo/ResNet50_ReID.yaml)。
