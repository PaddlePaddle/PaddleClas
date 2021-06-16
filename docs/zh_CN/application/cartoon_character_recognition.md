# 动漫人物识别
## 简介
   自七十年代以来，人脸识别已经成为了计算机视觉和生物识别领域研究最多的主题之一。近年来，传统的人脸识别方法已经被基于卷积神经网络（CNN）的深度学习方法代替。目前，人脸识别技术广泛应用于安防、商业、金融、智慧自助终端、娱乐等各个领域。而在行业应用强烈需求的推动下，动漫媒体越来越受到关注，动漫人物的人脸识别也成为一个新的研究领域。

## 数据集
近日，来自爱奇艺的一项新研究提出了一个新的基准数据集，名为iCartoonFace。该数据集由 5013 个动漫角色的 389678 张图像组成，并带有 ID、边界框、姿势和其他辅助属性。 iCartoonFace 是目前图像识别领域规模最大的卡通媒体数据集，而且质量高、注释丰富、内容全面，其中包含相似图像、有遮挡的图像以及外观有变化的图像。
与其他数据集相比，iCartoonFace无论在图像数量还是实体数量上，均具有明显领先的优势:

![icartoon](../../images/icartoon1.png)

论文地址：https://arxiv.org/pdf/1907.1339

### 1.2 数据预处理

由于原始的数据集中，图像包含标注的检测框，在识别阶段只考虑检测器抠图后的logo区域，因此采用原始的标注框抠出Logo区域图像构成训练集，排除背景在识别阶段的影响。对数据集进行划分，产生155427张训练集，覆盖3000个logo类别（同时作为测试时gallery图库），3225张测试集，用于作为查询集。抠图后的训练集可[在此下载](https://arxiv.org/abs/2008.05359)
- 图像`Resize`到224
- 随机水平翻转
- [AugMix](https://arxiv.org/abs/1912.02781v1)
- Normlize：归一化到0~1
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)

## 2 Backbone的具体设置

具体是用`ResNet50`作为backbone，主要做了如下修改：

 - 使用ImageNet预训练模型

 - last stage stride=1, 保持最后输出特征图尺寸14x14

 - 在最后加入一个embedding 卷积层，特征维度为512

   具体代码：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

## 3 Loss的设置

在Logo识别中，使用了[Pairwise Cosface + CircleMargin](https://arxiv.org/abs/2002.10857) 联合训练，其中权重比例为1:1

具体代码详见：[PairwiseCosface](../../../ppcls/loss/pairwisecosface.py) 、[CircleMargin](../../../ppcls/arch/gears/circlemargin.py)



其他部分参数，详见[配置文件](../../../ppcls/configs/Logo/ResNet50_ReID.yaml)。

## 参数设置
详细的参数设置见. [ResNet50_icartoon.yaml](../../../../ppcls/configs/Cartoon/ResNet50_icartoon.yaml)

