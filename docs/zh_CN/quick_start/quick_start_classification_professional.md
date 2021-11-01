# 30分钟玩转PaddleClas（进阶版）

此处提供了专业用户在linux操作系统上使用PaddleClas的快速上手教程，主要内容基于CIFAR-100数据集，快速体验不同模型的训练、加载不同预训练模型、SSLD知识蒸馏方案和数据增广的效果。请事先参考[安装指南](../installation/install_paddleclas.md)配置运行环境和克隆PaddleClas代码。


## 一、数据和模型准备

### 1.1 数据准备


* 进入PaddleClas目录。

```
cd path_to_PaddleClas
```

#### 1.1.1 准备CIFAR100

* 进入`dataset/`目录，下载并解压CIFAR100数据集。

```shell
cd dataset
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/CIFAR100.tar
tar -xf CIFAR100.tar
cd ../
```


## 二、模型训练

### 2.1 单标签训练

#### 2.1.1 零基础训练：不加载预训练模型的训练

* 基于ResNet50_vd模型，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR"
```


验证集的最高准确率为0.415左右。


#### 2.1.2 迁移学习

* 基于ImageNet1k分类预训练模型ResNet50_vd_pretrained(准确率79.12\%)进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

验证集最高准确率为0.718左右，加载预训练模型之后，CIFAR100数据集精度大幅提升，绝对精度涨幅30\%。

* 基于ImageNet1k分类预训练模型ResNet50_vd_ssld_pretrained(准确率82.39\%)进行微调，训练脚本如下所示。

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

最终CIFAR100验证集上精度指标为0.73，相对于79.12\%预训练模型的微调结构，新数据集指标可以再次提升1.2\%。

* 替换backbone为MobileNetV3_large_x1_0进行微调，训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV3_large_x1_0_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

验证集最高准确率为0.601左右, 较ResNet50_vd低近12%。


## 三、数据增广

PaddleClas包含了很多数据增广的方法，如Mixup、Cutout、RandomErasing等，具体的方法可以参考[数据增广的章节](../algorithm_introduction/DataAugmentation.md)。

### 数据增广的尝试-Mixup

基于`3.3节`中的训练方法，结合Mixup的数据增广方式进行训练，具体的训练脚本如下所示。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_mixup_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR"

```

最终CIFAR100验证集上的精度为0.73，使用数据增广可以使得模型精度再次提升约1.2\%。



* **注意**

    * 其他数据增广的配置文件可以参考`ppcls/configs/DataAugment`中的配置文件。

    * 训练CIFAR100的迭代轮数较少，因此进行训练时，验证集的精度指标可能会有1\%左右的波动。


## 四、知识蒸馏


PaddleClas包含了自研的SSLD知识蒸馏方案，具体的内容可以参考[知识蒸馏章节](../algorithm_introduction/knowledge_distillation.md), 本小节将尝试使用知识蒸馏技术对MobileNetV3_large_x1_0模型进行训练，使用`2.1.2小节`训练得到的ResNet50_vd模型作为蒸馏所用的教师模型，首先将`2.1.2小节`训练得到的ResNet50_vd模型保存到指定目录，脚本如下。

```shell
mkdir pretrained
cp -r output_CIFAR/ResNet50_vd/best_model.pdparams  ./pretrained/
```

配置文件中模型名字、教师模型哈学生模型的配置、预训练地址配置以及freeze_params配置如下，其中freeze_params_list中的两个值分别代表教师模型和学生模型是否冻结参数训练。

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

Loss配置如下，其中训练Loss是学生模型的输出和教师模型的输出的交叉熵、验证Loss是学生模型的输出和真实标签的交叉熵。
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

最终CIFAR100验证集上的精度为64.4\%，使用教师模型进行知识蒸馏，MobileNetV3的精度涨幅4.3\%。

* **注意**

    * 蒸馏过程中，教师模型使用的预训练模型为CIFAR100数据集上的训练结果，学生模型使用的是ImageNet1k数据集上精度为75.32\%的MobileNetV3_large_x1_0预训练模型。

    * 该蒸馏过程无须使用真实标签，所以可以使用更多的无标签数据，在使用过程中，可以将无标签数据生成假的train_list.txt，然后与真实的train_list.txt进行合并, 用户可以根据自己的数据自行体验。


## 五、模型评估与推理

### 5.1 单标签分类模型评估与推理

#### 5.1.1 单标签分类模型评估。

训练好模型之后，可以通过以下命令实现对模型精度的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model="output_CIFAR/ResNet50_vd/best_model"
```

#### 5.1.2 单标签分类模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Infer.infer_imgs=./dataset/CIFAR100/test/0/0001.png \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```


#### 5.1.3 单标签分类使用inference模型进行模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```

* 默认会在`inference`文件夹下生成`inference.pdiparams`、`inference.pdmodel`和`inference.pdiparams.info`文件。

使用预测引擎进行推理：

进入deploy目录下：

```bash
cd deploy
```
更改inference_cls.yaml文件，由于训练CIFAR100采用的分辨率是32x32，所以需要改变相关的分辨率，最终配置文件中的图像预处理如下：

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

执行命令进行预测，由于默认class_id_map_file是ImageNet数据集的映射文件，所以此处需要置None。

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/CIFAR100/test/0/0001.png \
    -o PostProcess.Topk.class_id_map_file=None
```
