# Image Augmentation
---

Experiments about data augmentation will be introduced in detail in this section. If you want to quickly experience these methods, please refer to [**Quick start PaddleClas in 30 miniutes**](../../tutorials/quick_start_en.md), which based on CIFAR100 dataset. If you want to know the content of related algorithms, please refer to [Data Augmentation Algorithm Introduction](../algorithm_introduction/DataAugmentation_en.md).


## Catalogue

- [1. Configurations](#1)
    - [1.1 AutoAugment](#1.1)
    - [1.2 RandAugment](#1.2)
    - [1.3 TimmAutoAugment](#1.3)
    - [1.4 Cutout](#1.4)
    - [1.5 RandomErasing](#1.5)
    - [1.6 HideAndSeek](#1.6)
    - [1.7 GridMask](#1.7)
    - [1.8 Mixup](#1.8)
    - [1.9 Cutmix](#1.9)
    - [1.10 Use Mixup and Cutmix at the same time](#1.10)
- [2. Start training](#2)
- [3. Matters needing attention](#3)
- [4. Experiments](#4)


<a name="1"></a>
## Configurations

Since hyperparameters differ from different augmentation methods. For better understanding, we list 8 augmentation configuration files in `configs/DataAugment` based on ResNet50. Users can train the model with `tools/run.sh`. The following are 3 of them.

<a name="1.1"></a>
### 1.1 AutoAugment

The configuration of the data augmentation method of `AotoAugment` is as follows. `AutoAugment` is converted on the uint8 data format, so its processing should be placed before the normalization operation (`NormalizeImage`).

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

<a name="1.2"></a>
### 1.2 RandAugment

The configuration of the data augmentation method of `RandAugment` is as follows, where the user needs to specify the parameters `num_layers` and `magnitude`, and the default values ​​are `2` and `5` respectively. `RandAugment` is converted on the uint8 data format, so its processing should be placed before the normalization operation (`NormalizeImage`).

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

<a name="1.3"></a>
### 1.3 TimmAutoAugment

The configuration of the data augmentation method of `TimmAutoAugment` is as follows, in which the user needs to specify the parameters `config_str`, `interpolation`, and `img_size`. The default values ​​are `rand-m9-mstd0.5-inc1` and `bicubic. `, `224`. `TimmAutoAugment` is converted on the uint8 data format, so its processing should be placed before the normalization operation (`NormalizeImage`).

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

<a name="1.4"></a>
### 1.4 Cutout

The configuration of the data augmentation method of `Cutout` is as follows, where the user needs to specify the parameters `n_holes` and `length`, and the default values are `1` and `112` respectively. Similar to other image cropping data augmentation methods, `Cutout` can operate on data in uint8 format, or on data after normalization (`NormalizeImage`).The demo here is operated after normalization.

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

<a name="1.5"></a>
### 1.5 RandomErasing

The configuration of the image augmentation method of `RandomErasing` is as follows, where the user needs to specify the parameters `EPSILON`, `sl`, `sh`, `r1`, `attempt`, `use_log_aspect`, `mode`, and the default values They are `0.25`, `0.02`, `1.0/3.0`, `0.3`, `10`, `True`, and `pixel`.  Similar to other image cropping data augmentation methods, `RandomErasing` can operate on data in uint8 format, or on data after normalization (`NormalizeImage`).The demo here is operated after normalization.

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

<a name="1.6"></a>
### 1.6 HideAndSeek

The configuration of the image augmentation method of `HideAndSeek` is as follows. Similar to other image cropping data augmentation methods, `HideAndSeek` can operate on data in uint8 format, or on data after normalization (`NormalizeImage`).The demo here is operated after normalization.

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

<a name="1.7"></a>
### 1.7 GridMask

The configuration of the image augmentation method of `GridMask` is as follows, where the user needs to specify the parameters `d1`, `d2`, `rotate`, `ratio`, `mode`, and the default values are `96`, `224 respectively `, `1`, `0.5`, `0`. Similar to other image cropping data augmentation methods, `HideAndSeek` can operate on data in uint8 format, or on data after normalization (`GridMask`).The demo here is operated after normalization.

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

<a name="1.8"></a>
### 1.8 Mixup

The configuration of the data augmentation method of `Mixup` is as follows, where the user needs to specify the parameter `alpha`, and the default value is `0.2`. Similar to other image mixing data augmentation methods, `Mixup` is to perform image mix on the data in each batch after the image is processed, and the mixed images and labels are put into the network for training, 
so it operates after image data processing (image transformation, image cropping).

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

<a name="1.9"></a>
### 1.9 Cutmix

The configuration of the image augmentation method of `Cutmix` is as follows, where the user needs to specify the parameter `alpha`, and the default value is `0.2`. Similar to other image mixing data augmentation methods, `Mixup` is to perform image mix on the data in each batch after the image is processed, and the mixed images and labels are put into the network for training, 
so it operates after image data processing (image transformation, image cropping).

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

<a name="1.10"></a>
### 1.10 Use Mixup and Cutmix at the same time

The configuration for both `Mixup` and `Cutmix` is as follows, in which the user needs to specify an additional parameter `prob`, which controls the probability of different data enhancements, and the default is `0.5`.

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

<a name="2"></a>
## 2. Start training

After you configure the training environment, similar to training other classification tasks, you only need to replace the configuration file in `tools/train.sh` with the configuration file of the corresponding data augmentation method.


The contents of `train.sh` are as follows:

```bash
python3 -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    --log_dir=ResNet50_Cutout \
    tools/train.py \
        -c ./ppcls/configs/ImageNet/DataAugment/ResNet50_Cutout.yaml
```

Run `train.sh`:

```bash
sh tools/train.sh
```

<a name="3"></a>
## 3. Matters needing attention

* In addition, because the label needs to be aliased when the image is aliased, the accuracy of the training data cannot be calculated. The training accuracy rate was not printed during the training process.

* The training data is more difficult with data augmentation, so the training loss may be larger, the training set accuracy is relatively low, but it has better generalization ability, so the validation set accuracy is relatively higher.

* After the use of data augmentation, the model may tend to be underfitting. It is recommended to reduce `l2_decay` for better performance on validation set.

* hyperparameters exist in almost all agmenatation methods. Here we provide hyperparameters for ImageNet1k dataset. User may need to finetune the hyperparameters on specified dataset. More training tricks can be referred to [Tricks](../models_training/train_strategy_en.md).


> If this document is helpful to you, welcome to star our project: [https://github.com/PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

<a name="4"></a>
## 4. Experiments

Based on PaddleClas, Metrics of different augmentation methods on ImageNet1k dataset are as follows.


| Model          | Learning strategy  | l2 decay | batch size | epoch | Augmentation method   | Top1 Acc    | Reference |
|-------------|------------------|--------------|------------|-------|----------------|------------|----|
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | Standard transform           | 0.7731 | - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | AutoAugment    | 0.7795 |  0.7763 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | mixup          | 0.7828 |  0.7790 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | cutmix         | 0.7839 |  0.7860 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | cutout         | 0.7801 |  - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | gridmask       | 0.7785 |  0.7790 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | random-augment | 0.7770 |  0.7760 |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | random erasing | 0.7791 |  - |
| ResNet50 | 0.1/cosine_decay | 0.0001       | 256        | 300   | hide and seek  | 0.7743 |  0.7720 |


**note**:
* In the experiment here, for better comparison, we fixed the l2 decay to 1e-4. To achieve higher accuracy, we recommend trying to use a smaller l2 decay. Combined with data augmentaton, we found that reducing l2 decay from 1e-4 to 7e-5 can bring at least 0.3~0.5% accuracy improvement.
* We have not yet combined different strategies or verified, whch is our future work.
