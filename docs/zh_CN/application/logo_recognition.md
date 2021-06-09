# Logo识别

 Logo识别技术，是现实生活中应用很广的一个领域，比如一张照片中是否出现了Adidas或者Nike的商标Logo，或者一个杯子上是否出现了星巴克或者可口可乐的商标Logo。通常Logo类别数量较多时，往往采用检测+识别两阶段方式，检测模块负责检测出潜在的Logo区域，根据检测区域抠图后输入识别模块进行识别。识别模块多采用检索的方式，根据查询图片和底库图片进行相似度排序获得预测类别。此文档主要对Logo图片的特征提取部分进行相关介绍，内容包括：

-  数据集及预处理方式
-  Backbone的具体设置
-  Loss函数的相关设置

全部的超参数及具体配置：[ResNet50_ReID.yaml](../../../ppcls/configs/Logo/ResNet50_ReID.yaml)

## 数据集及预处理

### LogoDet-3K数据集

<img src="../../images/logo/logo3k.JPG" style="zoom:50%;" />

LogoDet-3K数据集是具有完整标注的Logo数据集，有3000个标识类别，约20万个高质量的人工标注的标识对象和158652张图片。相关数据介绍参考[原论文](https://arxiv.org/abs/2008.05359)

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
        # 具体使用的Dataset的的名称
        name: "LogoDataset"
        # 使用此数据集的具体参数
        image_root: "dataset/LogoDet-3K-crop/train/"
        cls_label_path: "dataset/LogoDet-3K-crop/LogoDet-3K+train.txt"
        # 图像增广策略：ResizeImage、RandFlipImage等
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

具体是用`ResNet50`作为backbone，主要做了如下修改：

 - 使用ImageNet预训练模型

 - last stage stride=1, 保持最后输出特征图尺寸14x14

 - 在最后加入一个embedding 卷积层，特征维度为512

   具体代码：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

在配置文件中Backbone设置如下：

```yaml
Arch:
  # 使用RecModel模型进行训练，目前支持普通ImageNet和RecModel两个方式
  name: "RecModel"
  # 导出inference model的具体配置
  infer_output_key: "features"
  infer_add_softmax: False
  # 使用的Backbone
  Backbone:
    name: "ResNet50_last_stage_stride1"
    pretrained: True
  # 使用此层作为Backbone的feature输出，name为具体层的full_name
  BackboneStopLayer:
    name: "adaptive_avg_pool2d_0"
  # Backbone的基础上，新增网络层。此模型添加1x1的卷积层（embedding）
  Neck:
    name: "VehicleNeck"
    in_channels: 2048
    out_channels: 512
  # 增加CircleMargin head
  Head:
    name: "CircleMargin"
    margin: 0.35
    scale:  64
    embedding_size: 512
```

## Loss的设置

在Logo识别中，使用了[Pairwise Cosface + CircleMargin](https://arxiv.org/abs/2002.10857) 联合训练，其中权重比例为1:1

具体代码详见：[PairwiseCosface](../../../ppcls/loss/pairwisecosface.py) 、[CircleMargin](../../../ppcls/arch/gears/circlemargin.py)

在配置文件中设置如下：

```yaml
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - PairwiseCosface:
        margin: 0.35
        gamma: 64
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0
```

## 其他相关设置

### Optimizer设置

```yaml
Optimizer:
  # 使用的优化器名称
  name: Momentum
  # 优化器具体参数
  momentum: 0.9
  lr:
    # 使用的学习率调节具体名称
    name: Cosine
    # 学习率调节算法具体参数
    learning_rate: 0.01
  regularizer:
    name: 'L2'
    coeff: 0.0001
```

### Eval Metric设置

```yaml
Metric:
  Eval:
    # 使用Recallk和mAP两种评价指标
    - Recallk:
        topk: [1, 5]
    - mAP: {}
```

### 其他超参数设置

```yaml
Global:
  # 如为null则从头开始训练。若指定中间训练保存的状态地址，则继续训练
  checkpoints: null
  pretrained_model: null
  output_dir: "./output/"
  device: "gpu"
  class_num: 3000
  # 保存模型的粒度，每个epoch保存一次
  save_interval: 1
  eval_during_train: True
  eval_interval: 1
  # 训练的epoch数
  epochs: 120
  # log输出频率
  print_batch_step: 10
  # 是否使用visualdl库
  use_visualdl: False
  # used for static mode and model export
  image_shape: [3, 224, 224]
  save_inference_dir: "./inference"
  # 使用retrival的方式进行评测
  eval_mode: "retrieval"
```
