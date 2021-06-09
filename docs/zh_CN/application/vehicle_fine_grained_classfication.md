# 车辆细粒度分类

细粒度分类，是对属于某一类基础类别的图像进行子类别的细粉，如各种鸟、各种花、各种矿石之间。顾名思义，车辆细粒度分类是对车辆的不同子类别进行分类。

其训练过程与车辆ReID相比，有以下不同：

- 数据集不同
- Loss设置不同

其他部分请详见[车辆ReID](./vehicle_reid.md)

## 数据集

在此demo中，使用[CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)作为训练数据集

<img src="../../images/vehicle/CompCars.png" style="zoom:50%;" />

图像主要来自网络和监控数据，其中网络数据包含163个汽车制造商、1716个汽车型号的汽车。共**136,726**张全车图像，**27,618**张部分车图像。其中网络汽车数据包含bounding box、视角、5个属性（最大速度、排量、车门数、车座数、汽车类型）。监控数据包含**50,000**张前视角图像。

值得注意的是，此数据集中需要根据自己的需要生成不同的label，如本demo中，将不同年份生产的相同型号的车辆视为同一类，因此，类别总数为：431类。

## Loss设置

与车辆ReID不同，在此分类中，Loss使用的是[TtripLet Loss](../../../ppcls/loss/triplet.py) + [ArcLoss](../../../ppcls/arch/gears/arcmargin.py)，权重比例1:1。

整体配置文件：[ResNet50.yaml](../../../ppcls/configs/Vehicle/ResNet50.yaml)
