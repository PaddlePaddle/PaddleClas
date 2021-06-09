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

在配置文件中设置如下，详见`transform_ops`部分：

```yaml
DataLoader:
  Train:
    dataset:
    		#具体使用的Dataset的的名称
        name: "VeriWild"
        #使用此数据集的具体参数
        image_root: "/work/dataset/VeRI-Wild/images/"
        cls_label_path: "/work/dataset/VeRI-Wild/train_test_split/train_list_start0.txt"
        #图像增广策略：ResizeImage、RandFlipImage等
        transform_ops:
          - ResizeImage:
              size: 224
          - RandFlipImage:
              flip_code: 1
          - AugMix:
              prob: 0.5
          - NormalizeImage:
              scale: 0.00392157
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
              order: ''
          - RandomErasing:
              EPSILON: 0.5
              sl: 0.02
              sh: 0.4
              r1: 0.3
              mean: [0., 0., 0.]
    sampler:
        name: DistributedRandomIdentitySampler
        batch_size: 128
        num_instances: 2
        drop_last: False
        shuffle: True
    loader:
        num_workers: 6
        use_shared_memory: False
```



## Backbone的具体设置

具体是用`ResNet50`作为backbone，但在`ResNet50`基础上做了如下修改：

- 对Last Stage（第4个stage），没有做下采样，即第4个stage的feature map和第3个stage的feature map大小一致，都是14x14。
- 在最后加入一个embedding 层，即1x1的卷积层，特征维度为512

具体代码：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

在配置文件中Backbone设置如下：

```yaml
Arch:
	#使用RecModel模型进行训练，目前支持普通ImageNet和RecModel两个方式
  name: "RecModel"
  #使用的Backbone
  Backbone:
    name: "ResNet50_last_stage_stride1"
    pretrained: True
  #使用此层作为Backbone的feature输出，name为具体层的full_name
  BackboneStopLayer:
    name: "adaptive_avg_pool2d_0"
  #Backbone的基础上，新增网络层。此模型添加1x1的卷积层（embedding）
  Neck:
    name: "VehicleNeck"
    in_channels: 2048
    out_channels: 512
  #增加ArcMargin， 即ArcLoss的具体实现
  Head:
    name: "ArcMargin"  
    embedding_size: 512
    class_num: 431
    margin: 0.15
    scale: 32
```



## Loss的设置

车辆ReID中，使用了[SupConLoss](https://arxiv.org/abs/2004.11362) + [ArcLoss](https://arxiv.org/abs/1801.07698)，其中权重比例为1:1

具体代码详见：[SupConLoss代码](../../../ppcls/loss/supconloss.py)、[ArcLoss代码](../../../ppcls/arch/gears/arcmargin.py)

在配置文件中设置如下：

```yaml
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - SupConLoss:
        weight: 1.0
        #SupConLoss的具体参数
        views: 2
  Eval:
    - CELoss:
        weight: 1.0
```



## 其他相关设置

### Optimizer设置

```yaml
Optimizer:
  #使用的优化器名称
  name: Momentum
  #优化器具体参数
  momentum: 0.9
  lr:
    #使用的学习率调节具体名称
    name: MultiStepDecay
    #学习率调节算法具体参数
    learning_rate: 0.01
    milestones: [30, 60, 70, 80, 90, 100, 120, 140]
    gamma: 0.5
    verbose: False
    last_epoch: -1
  regularizer:
    name: 'L2'
    coeff: 0.0005
```



## Eval Metric设置

```yaml
Metric:
  Eval:
    #使用Recallk和mAP两种评价指标
    - Recallk:
        topk: [1, 5]
    - mAP: {}
```
