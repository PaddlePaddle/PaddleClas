# 一、数据增强分类实战

本节将基于ImageNet-1K的数据集详细介绍数据增强实验，如果想快速体验此方法，可以参考[**30分钟玩转PaddleClas（进阶版）**](../quick_start/quick_start/quick_start_classification_professional.md)中基于CIFAR100的数据增强实验。如果想了解相关算法的内容，请参考[数据增强算法介绍](../algorithm_introduction/DataAugmentation.md)。

## 1.1 参数配置

由于不同的数据增强方式含有不同的超参数，为了便于理解和使用，我们在`configs/DataAugment`里分别列举了8种训练ResNet50的数据增强方式的参数配置文件，用户可以在`tools/run.sh`里直接替换配置文件的路径即可使用。此处分别挑选了图像变换、图像裁剪、图像混叠中的一个示例展示，其他参数配置用户可以自查配置文件。

### AutoAugment

`AotoAugment`的图像增广方式的配置如下。`AutoAugment`是在uint8的数据格式上转换的，所以其处理过程应该放在归一化操作（`NormalizeImage`）之前。

```yaml        
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - AutoAugment:
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```

### RandAugment

`RandAugment`的图像增广方式的配置如下，其中用户需要指定其中的参数`num_layers`与`magnitude`，默认的数值分别是`2`和`5`。`RandAugment`是在uint8的数据格式上转换的，所以其处理过程应该放在归一化操作（`NormalizeImage`）之前。

```yaml        
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - RandAugment:
            num_layers: 2
            magnitude: 5
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```
### TimmAutoAugment

`TimmAutoAugment`的图像增广方式的配置如下，其中用户需要指定其中的参数`config_str`、`interpolation`、`img_size`，默认的数值分别是`rand-m9-mstd0.5-inc1`、`bicubic`、`224`。`TimmAutoAugment`是在uint8的数据格式上转换的，所以其处理过程应该放在归一化操作（`NormalizeImage`）之前。

```yaml        
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - TimmAutoAugment:
            config_str: rand-m9-mstd0.5-inc1
            interpolation: bicubic
            img_size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```

### Cutout

`Cutout`的图像增广方式的配置如下，其中用户需要指定其中的参数`n_holes`与`length`，默认的数值分别是`1`和`112`。类似其他图像裁剪类的数据增强方式，`Cutout`既可以在uint8格式的数据上操作，也可以在归一化（`NormalizeImage`）后的数据上操作，此处给出的是在归一化后的操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - Cutout:
            n_holes: 1
            length: 112
```

### RandomErasing

`RandomErasing`的图像增广方式的配置如下，其中用户需要指定其中的参数`EPSILON`、`sl`、`sh`、`r1`、`attempt`、`use_log_aspect`、`mode`，默认的数值分别是`0.25`、`0.02`、`1.0/3.0`、`0.3`、`10`、`True`、`pixel`。类似其他图像裁剪类的数据增强方式，`RandomErasing`既可以在uint8格式的数据上操作，也可以在归一化（`NormalizeImage`）后的数据上操作，此处给出的是在归一化后的操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - RandomErasing:
            EPSILON: 0.25
            sl: 0.02
            sh: 1.0/3.0
            r1: 0.3
            attempt: 10
            use_log_aspect: True
            mode: pixel
```

### HideAndSeek

`HideAndSeek`的图像增广方式的配置如下。类似其他图像裁剪类的数据增强方式，`HideAndSeek`既可以在uint8格式的数据上操作，也可以在归一化（`NormalizeImage`）后的数据上操作，此处给出的是在归一化后的操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - HideAndSeek:
```

### GridMask

`GridMask`的图像增广方式的配置如下,其中用户需要指定其中的参数`d1`、`d2`、`rotate`、`ratio`、`mode`, 默认的数值分别是`96`、`224`、`1`、`0.5`、`0`。类似其他图像裁剪类的数据增强方式，`GridMask`既可以在uint8格式的数据上操作，也可以在归一化（`NormalizeImage`）后的数据上操作，此处给出的是在归一化后的操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - GridMask:
            d1: 96
            d2: 224
            rotate: 1
            ratio: 0.5
            mode: 0
```


### Mixup

`Mixup`的图像增广方式的配置如下，其中用户需要指定其中的参数`alpha`，默认的数值是`0.2`。类似其他图像混合类的数据增强方式，`Mixup`是在图像做完数据处理后将每个batch内的数据做图像混叠，将混叠后的图像和标签输入网络中训练，所以其是在图像数据处理（图像变换、图像裁剪）后操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
      batch_transform_ops:
        - MixupOperator:
            alpha: 0.2
```

### Cutmix

`Cutmix`的图像增广方式的配置如下，其中用户需要指定其中的参数`alpha`，默认的数值是`0.2`。类似其他图像混合类的数据增强方式，`Cutmix`是在图像做完数据处理后将每个batch内的数据做图像混叠，将混叠后的图像和标签输入网络中训练，所以其是在图像数据处理（图像变换、图像裁剪）后操作。

```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
      batch_transform_ops:
        - CutmixOperator:
            alpha: 0.2
```

### Mixup与Cutmix同时使用

`Mixup``与Cutmix`同时使用的配置如下，其中用户需要指定额外的参数`prob`，该参数控制不同数据增强的概率，默认为`0.5`。
```yaml
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - OpSampler:
            MixupOperator:
              alpha: 0.8
              prob: 0.5
            CutmixOperator:
              alpha: 1.0
              prob: 0.5
```

## 1.2 启动命令

当用户配置完训练环境后，类似于训练其他分类任务，只需要将`tools/train.sh`中的配置文件替换成为相应的数据增强方式的配置文件即可。

其中`train.sh`中的内容如下：

```bash

python3 -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    --log_dir=ResNet50_Cutout \
    tools/train.py \
        -c ./ppcls/configs/ImageNet/DataAugment/ResNet50_Cutout.yaml
```

运行`train.sh`：

```bash
sh tools/train.sh
```

## 1.3 注意事项

* 由于图像混叠时需对label进行混叠，无法计算训练数据的准确率，所以在训练过程中没有打印训练准确率。

* 在使用数据增强后，由于训练数据更难，所以训练损失函数可能较大，训练集的准确率相对较低，但其有拥更好的泛化能力，所以验证集的准确率相对较高。

* 在使用数据增强后，模型可能会趋于欠拟合状态，建议可以适当的调小`l2_decay`的值来获得更高的验证集准确率。

* 几乎每一类图像增强均含有超参数，我们只提供了基于ImageNet-1k的超参数，其他数据集需要用户自己调试超参数，具体超参数的含义用户可以阅读相关的论文，调试方法也可以参考训练技巧的章节。

## 二、实验结果

基于PaddleClas，在ImageNet1k数据集上的分类精度如下。

| 模型          | 初始学习率策略  | l2 decay | batch size | epoch | 数据变化策略         | Top1 Acc    | 论文中结论 |
|-------------|------------------|--------------|------------|-------|----------------|------------|----|
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | 标准变换           | 0.7731 | - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | AutoAugment    | 0.7795 |  0.7763 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | mixup          | 0.7828 |  0.7790 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | cutmix         | 0.7839 |  0.7860 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | cutout         | 0.7801 |  - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | gridmask       | 0.7785 |  0.7790 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | random-augment | 0.7770 |  0.7760 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | random erasing | 0.7791 |  - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | hide and seek  | 0.7743 |  0.7720 |

**注意**：
* 在这里的实验中，为了便于对比，我们将l2 decay固定设置为1e-4，在实际使用中，我们推荐尝试使用更小的l2 decay。结合数据增强，我们发现将l2 decay由1e-4减小为7e-5均能带来至少0.3~0.5%的精度提升。
* 我们目前尚未对不同策略进行组合并验证效果，这一块后续我们会开展更多的对比实验，敬请期待。
