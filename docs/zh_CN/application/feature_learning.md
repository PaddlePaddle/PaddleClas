# 特征学习

此部分主要是针对`RecModel`的训练模式进行说明。`RecModel`的训练模式，主要是为了支持车辆识别（车辆细分类、ReID）、Logo识别、动漫人物识别、商品识别等特征学习的应用。与在`ImageNet`上训练普通的分类网络不同的是，此训练模式，主要有以下特征

- 支持对`backbone`的输出进行截断，即支持提取任意中间层的特征信息
- 支持在`backbone`的feature输出层后，添加可配置的网络层，即`Neck`部分
- 支持`ArcMargin`等`metric learning` 相关loss函数，提升特征学习能力

## yaml文件说明

`RecModel`的训练模式与普通分类训练的配置类似，配置文件主要分为以下几个部分：

### 1 全局设置部分

```yaml
Global:
  # 如为null则从头开始训练。若指定中间训练保存的状态地址，则继续训练
  checkpoints: null
  # pretrained model路径或者 bool类型
  pretrained_model: null
  # 模型保存路径
  output_dir: "./output/"
  device: "gpu"
  class_num: 30671
  # 保存模型的粒度，每个epoch保存一次
  save_interval: 1
  eval_during_train: True
  eval_interval: 1
  # 训练的epoch数
  epochs: 160
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

### 2 数据部分

```yaml
DataLoader:
  Train:
    dataset:
        # 具体使用的Dataset的的名称
        name: "VeriWild"
        # 使用此数据集的具体参数
        image_root: "./dataset/VeRI-Wild/images/"
        cls_label_path: "./dataset/VeRI-Wild/train_test_split/train_list_start0.txt"
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

`val dataset`设置与`train dataset`除图像增广策略外，设置基本一致

### 3 Backbone的具体设置

```yaml
Arch:
  # 使用RecModel模式进行训练
  name: "RecModel"
  # 导出inference model的具体配置
  infer_output_key: "features"
  infer_add_softmax: False
  # 使用的Backbone
  Backbone:
    name: "ResNet50"
    pretrained: True
  # 使用此层作为Backbone的feature输出，name为ResNet50的full_name
  BackboneStopLayer:
    name: "adaptive_avg_pool2d_0"
  # Backbone的基础上，新增网络层。此模型添加1x1的卷积层（embedding）
  Neck:
    name: "VehicleNeck"
    in_channels: 2048
    out_channels: 512
  # 增加ArcMargin， 即ArcLoss的具体实现
  Head:
    name: "ArcMargin"  
    embedding_size: 512
    class_num: 431
    margin: 0.15
    scale: 32
```

`Neck`部分为在`bacbone`基础上，添加的网络层，可根据需求添加。 如在ReID任务中，添加一个输出长度为512的`embedding`层，可由此部分实现。需注意的是，`Neck`部分需对应好`BackboneStopLayer`层的输出维度。一般来说，`Neck`部分为网络的最终特征输出层。

`Head`部分主要是为了支持`metric learning`等具体loss函数，如`ArcMargin`([ArcFace Loss](https://arxiv.org/abs/1801.07698)的fc层的具体实现)，在完成训练后，一般将此部分剔除。

### 4 Loss的设置

```yaml
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - SupConLoss:
        weight: 1.0
        # SupConLoss的具体参数
        views: 2
  Eval:
    - CELoss:
        weight: 1.0
```

训练时同时使用`CELoss`和`SupConLoss`，权重比例为`1:1`，测试时只使用`CELoss`

### 5 优化器设置

```yaml
Optimizer:
  # 使用的优化器名称
  name: Momentum
  # 优化器具体参数
  momentum: 0.9
  lr:
    # 使用的学习率调节具体名称
    name: MultiStepDecay
    # 学习率调节算法具体参数
    learning_rate: 0.01
    milestones: [30, 60, 70, 80, 90, 100, 120, 140]
    gamma: 0.5
    verbose: False
    last_epoch: -1
  regularizer:
    name: 'L2'
    coeff: 0.0005
```

### 6 Eval Metric设置

```yaml
Metric:
  Eval:
    # 使用Recallk和mAP两种评价指标
    - Recallk:
        topk: [1, 5]
    - mAP: {}
```
