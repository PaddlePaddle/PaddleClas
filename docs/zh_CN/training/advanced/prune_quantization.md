
# 模型量化、裁剪使用介绍

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余。此部分提供精简模型的功能，包括两部分：模型量化（量化训练、离线量化）、模型剪枝。

其中模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
模型量化可以在基本不损失模型的精度的情况下，将 FP32 精度的模型参数转换为 Int8 精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

模型剪枝将 CNN 中不重要的卷积核裁剪掉，减少模型参数量，从而降低模型计算复杂度。

本教程将介绍如何使用飞桨模型压缩库 PaddleSlim 做 PaddleClas 模型的压缩，即裁剪、量化功能。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解 [PaddleClas 模型的训练方法](../single_label_classification/training.md)以及 [PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)，相关裁剪、量化方法可以参考[模型裁剪量化算法介绍文档](../../algorithm_introduction/prune_quantization.md)。

-----------

## 目录

  - [1. 准备环境](#1)
    - [1.1 安装 PaddleSlim](#1.1)
    - [1.2 准备训练好的模型](#1.2)
  - [2. 快速开始](#2)
    - [2.1 模型量化](#2.1)
      - [2.1.1 在线量化训练](#2.1.1)
      - [2.1.2 离线量化](#2.1.2)
    - [2.2 模型剪枝](#2.2)
  - [3. 导出模型](#3)
  - [4. 模型部署](#4)
  - [5. 训练超参数建议](#5)


<a name="1"></a>
## 1. 准备环境
当训练出一个模型后，如果希望进一步的压缩模型大小并加速预测，可使用量化或者剪枝的方法压缩模型。

模型压缩主要包括五个步骤：
1. 安装 PaddleSlim
2. 准备训练好的模型
3. 模型压缩
4. 导出量化推理模型
5. 量化模型预测部署

<a name="1.1"></a>
### 1.1 安装 PaddleSlim

* 可以通过 pip install 的方式进行安装。

```bash
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

* 如果获取 PaddleSlim 的最新特性，可以从源码安装。

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
```

<a name="1.2"></a>
### 1.2 准备训练好的模型

PaddleClas 提供了一系列训练好的[模型](../models/models_intro.md)，如果待量化的模型不在列表中，需要按照[常规训练](../single_label_classification/training.md)方法得到训练好的模型。

<a name="2"></a>
## 2. 快速开始

进入 PaddleClas 根目录

```bash
cd PaddleClas
```

`slim` 训练相关代码已经集成到 `ppcls/engine/` 下，离线量化代码位于 `deploy/slim/quant_post_static.py`。

<a name="2.1"></a>
### 2.1 模型量化

量化训练包括离线量化训练和在线量化训练，在线量化训练效果更好，需加载预训练模型，在定义好量化策略后即可对模型进行量化。

<a name="2.1.1"></a>
#### 2.1.1 在线量化训练

训练指令如下：

* CPU/单卡 GPU

以 CPU 为例，若使用 GPU，则将命令中改成 `cpu` 改成 `gpu`

```bash
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_quantization.yaml -o Global.device=cpu
```

其中 `yaml` 文件解析详见[参考文档](../config_description/basic.md)。为了保证精度，`yaml` 文件中已经使用 `pretrained model`.


* 单机多卡/多机多卡启动

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -c ppcls/configs/slim/ResNet50_vd_quantization.yaml
```

<a name="2.1.2"></a>
#### 2.1.2 离线量化

**注意**：目前离线量化，必须使用已经训练好的模型，导出的 `inference model` 进行量化。一般模型导出 `inference model` 可参考[教程](../../deployment/export_model.md).

一般来说，离线量化损失模型精度较多。

生成 `inference model` 后，离线量化运行方式如下

```bash
python3.7 deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

`Global.save_inference_dir` 是 `inference model` 存放的目录。

执行成功后，在 `Global.save_inference_dir` 的目录下，生成 `quant_post_static_model` 文件夹，其中存储生成的离线量化模型，其可以直接进行预测部署，无需再重新导出模型。

<a name="2.2"></a>
### 2.2 模型剪枝

训练指令如下：

- CPU/单卡 GPU

以 CPU 为例，若使用 GPU，则将命令中改成 `cpu` 改成 `gpu`

```bash
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_prune.yaml -o Global.device=cpu
```

- 单机单卡/单机多卡/多机多卡启动

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -c ppcls/configs/slim/ResNet50_vd_prune.yaml
```

<a name="3"></a>
## 3. 导出模型

在得到在线量化训练、模型剪枝保存的模型后，可以将其导出为 inference model，用于预测部署，以模型剪枝为例：

```bash
python3.7 tools/export_model.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
    -o Global.pretrained_model=./output/ResNet50_vd/best_model \
    -o Global.save_inference_dir=./inference
```

<a name="4"></a>
## 4. 模型部署

上述步骤导出的模型可以直接使用 inferecne 进行部署，参考 [inference 部署](../../deployment/)。

也通过 PaddleLite 的 opt 模型转换工具，完成 inference 模型到移动端模型转换，用于移动端的模型部署。
移动端模型部署的可参考 [移动端模型部署](../../deployment/image_classification/paddle_lite.md)。

<a name="5"></a>
## 5. 训练超参数建议

* 量化、裁剪训练时，建议加载常规训练得到的预训练模型，加速量化训练收敛。
* 量化训练时，建议初始学习率修改为常规训练的 `1/20~1/10`，同时将训练 epoch 数修改为常规训练的 `1/5~1/2`，学习率策略方面，加上 Warmup，其他配置信息不建议修改。
* 裁剪训练时，建议超参数配置与普通训练一致。
