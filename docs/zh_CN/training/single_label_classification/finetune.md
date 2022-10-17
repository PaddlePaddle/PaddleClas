# 模型微调
---

此处提供了用户在 linux 操作系统上使用 PaddleClas 的微调（finetune）模型的教程，此处默认您已经配置好本地环境且已经执行过[快速开始](../../quick_start/quick_start_classification_professional.md)相关的代码。

PaddleClas提供了丰富的 ImageNet 预训练模型，使用这些预训练模型做初始化训练自己的任务可以快速收敛，同时可以一定程度避免模型过拟合。

## 目录

- [1. 数据准备](#1)
- [2. 模型训练](#2)
    - [2.1 更改配置文件](#2.1)
    - [2.1 模型训练](#2.2)
- [FAQ](#faq)


<a name="1"></a>
## 1. 数据准备

PaddleClas 是通过`.txt`来读取数据，所以您需要将准备好的分类数据单独存放于一个文件夹，该文件夹中主要有以下四部分内容。

```
├── train
├── train_list.txt
├── val
├── val_list.txt
```

其中`train`和`val`分别存放训练数据和验证数据，`train_list.txt`和`val_list.txt`分别存放训练数据和验证数据的路径和标签。其对应的格式形如：

```shell
# 每一行采用"空格"分隔图像路径与标注

# 下面是 train_list.txt 中的格式样例
train/n01440764/n01440764_10026.JPEG 0
...

# 下面是 val_list.txt 中的格式样例
val/ILSVRC2012_val_00000001.JPEG 65
...
```

其中，`train/n01440764/n01440764_10026.JPEG` 表示该数据的路径，`0` 表示该数据属于 0 类。

**备注**：`train_list.txt`和`val_list.txt`需要您根据实际情况生成。

<a name="2"></a>
## 2. 模型训练

<a name="2.1"></a>
### 2.1 更改配置文件

在 finetune 自己的任务时，往往需要更改训练的配置文件，在`ppcls/configs/ImageNet/`中，定义了所有模型在 ImageNet 数据上的训练配置，可以直接修改该配置完成训练。下面以 ResNet50_vd 为例子，详述需要修改的配置。其中，该配置文件存在于`ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml`。

1.类别数：类别数量默认为1000（ImageNet-1k类别数），此处需要修改为实际任务中的类别数量。
```
Arch:
  name: ResNet50_vd
  class_num: your_class_num
```

2.预训练模型：增加预训练模型可以使Loss快速收敛，增加方式如下：
```
Arch:
  name: ResNet50_vd
  class_num: your_class_num
  pretrained: True
```

3.学习率：如果训练远小于 ImageNet 数据量的任务时候，学习率需要减小若干倍，具体减小的幅度需要调试，可以先从减小 10 倍开始调试。
```
Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Cosine
    learning_rate: your_learning_rate
```

4.数据路径：数据路径需要改为您自己的数据路径。即在[数据准备](#1)准备好的数据。

```
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: your_image_root
      cls_label_path: your_train_cls_label_path
      
  Eval:
    dataset: 
      name: ImageNetDataset
      image_root: your_image_root
      cls_label_path: your_eval_cls_label_path
```

<a name="2.2"></a>
### 2.2 模型训练

当修改完配置文件之后，就可以开始训练自己的任务，此处默认使用 4 卡训练，训练命令如下：

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml
```

<a name="faq"></a>
## FAQ

**Q1.如何更改分辨率？**

以 224 修改为 320 为例，如果只做训练和评测，可以只修改以下字段：

1.修改配置文件中`DataLoader.Train.dataset.transform_ops.1.RandCropImage.size`, 如从`224`修改到`320`；

2.修改配置文件中`DataLoader.Eval.dataset.transform_ops.1.ResizeImage.resize_short`, 如从`256`修改到`366`；

3.修改配置文件中`DataLoader.Eval.dataset.transform_ops.2.CropImage.size`, 如从`224`修改到`320`。

如果后续涉及模型导出或者infer，需要同时修改以下字段：

1.修改配置文件中`Global.image_shape`,如从`[3, 224, 224]`修改到`[3, 320, 320]`；

2.修改`Infer.transforms.1.ResizeImage.resize_short`, 如从 `256` 修改到 `366`；

3.修改配置文件中`Infer.transforms.2.CropImage.size`, 如从 `224` 修改到 `320`。


**Q2.如何更改数据增强？**

关于数据增强部分可以参考[数据增强文档](../config_description/data_augmentation.md)，里边有详细的介绍。

**Q3.如何更改评价指标？**

PaddleClas中的分类指标目前只支持 `Top-k`，如果需要更改 `k` 值，可以修改配置文件中` Metric.TopkAcc.topk`,如从 `[1, 5]` 修改到` [1, 3]`，即评测指标从 `Top-1`、`Top-5` 改为 `Top-1`、`Top-3`。

**Q4.如何进一步提升模型的精度？**

1.如果不考虑模型的推理速度，可以更换在 ImageNet 上精度更高的模型，或者使用更大的分辨率，如果考虑模型的推理速度，建议使用经过 PaddleClas 团队优化的 PP 系列模型，如[PP-LCNet](../../models/ImageNet1k/PP-LCNet.md)、[PP-HGNet](../../models/ImageNet1k/PP-HGNet.md)等；

2.可以尝试使用更好的预训练权重，PaddleClas 提供了 20+ 个 SSLD 的预训练权重，在很多任务中都可以提升 1 个百分点以上，使用方式只需要在 `Arch` 字段下增加`use_ssld: True` 即可，关于更多SSLD相关的知识，可以参考[SSLD 知识蒸馏实战](../advanced/ssld.md);

3.在实际的任务中，训练轮数对结果影响可能也比较大，初期可以使用较小的轮数（如 20）来调试。

4.模型在微调的过程中，学习率对结果影响比较大，建议多尝试几组学习率。
