# 30 分钟玩转 PaddleClas（进阶版）

此处提供了专业用户在 linux 操作系统上使用 PaddleClas 的快速上手教程，主要内容基于 CIFAR-100 数据集，快速体验不同模型的训练、加载不同预训练模型、SSLD 知识蒸馏方案和数据增广的效果。请事先参考[安装指南](../installation/install_paddleclas.md)配置运行环境和克隆 PaddleClas 代码。

------

## 目录

- [1. 数据和模型准备](#1)
  - [1.1 数据准备](#1.1)
    - [1.1.1 准备 CIFAR100](#1.1.1)
- [2. 模型训练](#2)
  - [2.1 单标签训练](#2.1)
    - [2.1.1 零基础训练：不加载预训练模型的训练](#2.1.1)
    - [2.1.2 迁移学习](#2.1.2)
- [3. 数据增广](#3)
  - [3.1 数据增广的尝试-Mixup](#3.1)
- [4. 知识蒸馏](#4)
- [5. 模型评估与推理](#5)
  - [5.1 单标签分类模型评估与推理](#5.1)
    - [5.1.1 单标签分类模型评估](#5.1.1)
    - [5.1.2 单标签分类模型预测](#5.1.2)
    - [5.1.3 单标签分类使用 inference 模型进行模型推理](#5.1.3)

<a name="1"></a>

## 1. 数据和模型准备

<a name="1.1"></a>

### 1.1 数据准备


* 进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

<a name="1.1.1"></a> 

#### 1.1.1 准备 CIFAR100

* 进入 `dataset/` 目录，下载并解压 CIFAR100 数据集。

```shell
cd dataset
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/CIFAR100.tar
tar -xf CIFAR100.tar
cd ../
```

<a name="2"></a>

## 2. 模型训练

<a name="2.1"></a> 

### 2.1 单标签训练

<a name="2.1.1"></a> 

#### 2.1.1 零基础训练：不加载预训练模型的训练

* 基于 ResNet50_vd 模型，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR"
```

验证集的最高准确率为 0.415 左右。

此处使用了多个 GPU 训练，如果只使用一个 GPU，请将 `CUDA_VISIBLE_DEVICES` 设置指定 GPU，`--gpus`设置指定 GPU，下同。例如，只使用 0 号 GPU 训练：

```shell
export CUDA_VISIBLE_DEVICES=0
python3 -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Optimizer.lr.learning_rate=0.01
```

* **注意**: 

* `--gpus`中指定的 GPU 可以是 `CUDA_VISIBLE_DEVICES` 指定的 GPU 的子集。
* 由于初始学习率和 batch-size 需要保持线性关系，所以训练从 4 个 GPU 切换到 1 个 GPU 训练时，总 batch-size 缩减为原来的 1/4，学习率也需要缩减为原来的 1/4，所以改变了默认的学习率从 0.04 到 0.01。

<a name="2.1.2"></a> 


#### 2.1.2 迁移学习

* 基于 ImageNet1k 分类预训练模型 ResNet50_vd_pretrained（准确率 79.12%）进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

验证集最高准确率为 0.718 左右，加载预训练模型之后，CIFAR100 数据集精度大幅提升，绝对精度涨幅 30%。

* 基于 ImageNet1k 分类预训练模型 ResNet50_vd_ssld_pretrained（准确率 82.39%）进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True \
        -o Arch.use_ssld=True
```

最终 CIFAR100 验证集上精度指标为 0.73，相对于 79.12% 预训练模型的微调结构，新数据集指标可以再次提升 1.2%。

* 替换 backbone 为 MobileNetV3_large_x1_0 进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV3_large_x1_0_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

验证集最高准确率为 0.601 左右, 较 ResNet50_vd 低近 12%。

<a name="3"></a>


## 3. 数据增广

PaddleClas 包含了很多数据增广的方法，如 Mixup、Cutout、RandomErasing 等，具体的方法可以参考[数据增广的章节](../algorithm_introduction/DataAugmentation.md)。

<a name="3.1"></a> 

### 3.1 数据增广的尝试-Mixup

基于[数据增广的章节](../algorithm_introduction/DataAugmentation.md) `3.3 节` 中的训练方法，结合 Mixup 的数据增广方式进行训练，具体的训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_mixup_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR"

```

最终 CIFAR100 验证集上的精度为 0.73，使用数据增广可以使得模型精度再次提升约 1.2%。



* **注意**

  * 其他数据增广的配置文件可以参考 `ppcls/configs/ImageNet/DataAugment/` 中的配置文件。
* 训练 CIFAR100 的迭代轮数较少，因此进行训练时，验证集的精度指标可能会有 1% 左右的波动。

<a name="4"></a>


## 4. 知识蒸馏


PaddleClas 包含了自研的 SSLD 知识蒸馏方案，具体的内容可以参考[知识蒸馏章节](../algorithm_introduction/knowledge_distillation.md), 本小节将尝试使用知识蒸馏技术对 MobileNetV3_large_x1_0 模型进行训练，使用 `2.1.2 小节` 训练得到的 ResNet50_vd 模型作为蒸馏所用的教师模型，首先将 `2.1.2 小节` 训练得到的 ResNet50_vd 模型保存到指定目录，脚本如下。

```shell
mkdir pretrained
cp -r output_CIFAR/ResNet50_vd/best_model.pdparams  ./pretrained/
```

配置文件中模型名字、教师模型和学生模型的配置、预训练地址配置以及 freeze_params 配置如下，其中 `freeze_params_list` 中的两个值分别代表教师模型和学生模型是否冻结参数训练。

```yaml
Arch:
  name: "DistillationModel"
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - True
  - False
  models:
    - Teacher:
        name: ResNet50_vd
        pretrained: "./pretrained/best_model"
    - Student:
        name: MobileNetV3_large_x1_0
        pretrained: True
```

Loss 配置如下，其中训练 Loss 是学生模型的输出和教师模型的输出的交叉熵、验证 Loss 是学生模型的输出和真实标签的交叉熵。

```yaml
Loss:
  Train:
    - DistillationCELoss:
        weight: 1.0
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
```

最终的训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/R50_vd_distill_MV3_large_x1_0_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR"

```

最终 CIFAR100 验证集上的精度为 64.4%，使用教师模型进行知识蒸馏，MobileNetV3 的精度涨幅 4.3%。

* **注意**

  * 蒸馏过程中，教师模型使用的预训练模型为 CIFAR100 数据集上的训练结果，学生模型使用的是 ImageNet1k 数据集上精度为 75.32% 的 MobileNetV3_large_x1_0 预训练模型。
  * 该蒸馏过程无须使用真实标签，所以可以使用更多的无标签数据，在使用过程中，可以将无标签数据生成假的 `train_list.txt`，然后与真实的 `train_list.txt` 进行合并, 用户可以根据自己的数据自行体验。

<a name="5"></a>

## 5. 模型评估与推理

<a name="5.1"></a> 

### 5.1 单标签分类模型评估与推理

<a name="5.1.1"></a> 

#### 5.1.1 单标签分类模型评估。

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model="output_CIFAR/ResNet50_vd/best_model"
```

<a name="5.1.2"></a> 

#### 5.1.2 单标签分类模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Infer.infer_imgs=./dataset/CIFAR100/test/0/0001.png \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```

<a name="5.1.3"></a> 

#### 5.1.3 单标签分类使用 inference 模型进行模型推理

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```

* 默认会在 `inference` 文件夹下生成 `inference.pdiparams`、`inference.pdmodel` 和 `inference.pdiparams.info` 文件。

使用预测引擎进行推理：

进入 deploy 目录下：

```bash
cd deploy
```

更改 `inference_cls.yaml` 文件，由于训练 CIFAR100 采用的分辨率是 32x32，所以需要改变相关的分辨率，最终配置文件中的图像预处理如下：

```yaml
PreProcess:
  transform_ops:
    - ResizeImage:
        resize_short: 36
    - CropImage:
        size: 32
    - NormalizeImage:
        scale: 0.00392157
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
```

执行命令进行预测，由于默认 `class_id_map_file` 是 ImageNet 数据集的映射文件，所以此处需要置 None。

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/CIFAR100/test/0/0001.png \
    -o PostProcess.Topk.class_id_map_file=None
```
