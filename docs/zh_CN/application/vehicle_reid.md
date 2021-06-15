# 车辆ReID
ReID，也就是 Re-identification，其定义是利用算法，在图像库中找到要搜索的目标的技术，所以它是属于图像检索的一个子问题。而车辆ReID就是给定一张车辆图像，找出同一摄像头不同的拍摄图像，或者不同摄像头下拍摄的同一车辆图像的过程。在此过程中，如何提取鲁棒特征，尤为重要。因此，此文档主要对车辆ReID中训练特征提取网络部分做相关介绍，内容如下：
- 数据集及预处理方式
- Backbone的具体设置
- Loss函数的相关设置

全部的超参数及具体配置：[ResNet50_ReID.yaml](../../../ppcls/configs/Vehicle/ResNet50_ReID.yaml)
## 数据集及预处理
### VERI-Wild数据集

<img src="../../images/recognotion/vehicle/cars.JPG" style="zoom:50%;" />

此数据集是在一个大型闭路电视监控系统，在无约束的场景下，一个月内（30*24小时）中捕获的。该系统由174个摄像头组成，其摄像机分布在200多平方公里的大型区域。原始车辆图像集包含1200万个车辆图像，经过数据清理和标注，采集了416314张40671个不同的车辆图像。[具体详见论文](https://github.com/PKU-IMRE/VERI-Wild)
## 数据预处理
由于原始的数据集中，车辆图像已经是由检测器检测后crop出的车辆图像，因此无需像训练`ImageNet`中图像crop操作。整体的数据增强方式，按照顺序如下：
- 图像`Resize`到224
- 随机水平翻转
- [AugMix](https://arxiv.org/abs/1912.02781v1)
- Normlize：归一化到0~1
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)

## Backbone的具体设置
具体是用`ResNet50`作为backbone，但在`ResNet50`基础上做了如下修改：
- 对Last Stage（第4个stage），没有做下采样，即第4个stage的feature map和第3个stage的feature map大小一致，都是14x14。
- 在最后加入一个embedding 层，即1x1的卷积层，特征维度为512
具体代码：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

## Loss的设置
车辆ReID中，使用了[SupConLoss](https://arxiv.org/abs/2004.11362) + [ArcLoss](https://arxiv.org/abs/1801.07698)，其中权重比例为1:1
具体代码详见：[SupConLoss代码](../../../ppcls/loss/supconloss.py)、[ArcLoss代码](../../../ppcls/arch/gears/arcmargin.py)



其他部分的具体设置，详见[配置文件](../../../ppcls/configs/Vehicle/ResNet50_ReID.yaml)。
