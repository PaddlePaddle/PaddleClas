# PaddleClas构建多语言分类案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的多语言的分类模型教程，主要基于多语言场景的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、SKL-UGI知识蒸馏策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的分类模型。

------


## 目录

- [1. 环境配置](#1)
- [2. 多语言场景推理预测](#2)

  - [2.1 下载模型](#2.1)  
  - [2.2 模型推理预测](#2.2)
      - [2.2.1 预测单张图像](#2.2.1)
      - [2.2.2 基于文件夹的批量预测](#2.2.2)
- [3.多语言场景训练](#3)

    - [3.1 数据准备](#3.1)
    - [3.2 模型训练](#3.2)
      - [3.2.1 基于默认超参数训练](#3.2.1)
        - [3.2.1.1 基于默认超参数训练轻量级模型](#3.2.1.1)
        - [3.2.1.2 基于默认超参数训练教师模型](#3.2.1.2)
        - [3.2.1.3 基于默认超参数进行蒸馏训练](#3.2.1.3)
      - [3.2.2 超参数搜索训练](#3.2)  
- [4. 模型评估与推理](#4)
  - [4.1 模型评估](#3.1)
  - [4.2 模型预测](#3.2)
  - [4.3 使用 inference 模型进行推理](#4.3)
    - [4.3.1 导出 inference 模型](#4.3.1)
    - [4.3.2 模型推理预测](#4.3.2)



<a name="1"></a>

## 1. 环境配置

* 安装：请先参考 [Paddle 安装教程](../installation/install_paddle.md) 以及 [PaddleClas 安装教程](../installation/install_paddleclas.md) 配置 PaddleClas 运行环境。

<a name="2"></a>

## 2. 多语言场景推理预测

<a name="2.1"></a>

### 2.1 下载模型

* 进入 `deploy` 运行目录。

```
cd deploy
```

下载多语言分类的模型。

```
mkdir models
cd models
# 下载inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/mlt_infer.tar && tar -xf mlt_cls_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── mlt_cls_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="2.2"></a>

### 2.2 模型推理预测

<a name="2.2.1"></a>

#### 2.2.1 预测单张图像

返回 `deploy` 目录：

```
cd ../
```

运行下面的命令，对图像 `./images/PULC/mlt/ILSVRC2012_val_00010000_5989.jpg` 进行多语言分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/mlt/inference_mlt_cls.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/mlt/inference_mlt_cls.yaml -o Global.use_gpu=False
```

输出结果如下。

```
ILSVRC2012_val_00010000_5989.jpg:    class id(s): [9, 1], score(s): [0.90, 0.01], label_name(s): ['latin', 'chinese_cht']
```

其中，输出为top2的预测结果，`latin` 表示该图中文字为拉丁语，`chinese_cht` 表示该图中文字为中文繁体。

<a name="2.2.2"></a>

#### 2.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/mlt/inference_mlt_cls.yaml -o Global.infer_imgs="./images/PULC/mlt/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00010000_5989.jpg:    class id(s): [9, 1], score(s): [0.90, 0.01], label_name(s): ['latin', 'chinese_cht']
ILSVRC2012_val_00010002_7373.jpg:    class id(s): [1, 9], score(s): [0.87, 0.02], label_name(s): ['chinese_cht', 'latin']
```

其中，输出为top2的预测结果，`latin` 表示该图中文字为拉丁语，`chinese_cht` 表示该图中文字为中文繁体。

<a name="3"></a>

## 3.多语言场景训练

<a name="3.1"></a>

### 3.1 数据准备

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压多语言场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/cls_demo/mlt.tar
tar -xf mlt.tar
cd ../
```

执行上述命令后，`dataset/`下存在`mlt`目录，该目录中具有以下数据：

```

├── arabic_img
│   ├── ILSVRC2012_val_00010000_3502.jpg
...
├── chinese_cht_img
│   ├── ILSVRC2012_val_00010000_1465.jpg
...
├── cyrillic_img
│   ├── ILSVRC2012_val_00010000_3762.jpg
...
├── devanagari_img
│   ├── ILSVRC2012_val_00010000_1820.jpg
...
├── latin_img
│   ├── ILSVRC2012_val_00010000_1705.jpg
...
...
├── train_list.txt
├── train_list.txt.debug
├── train_list_for_distill.txt
├── val_list.txt
├── val_list.txt.debug
└── label_list.txt
```

其中`arabic_img/`、`chinese_cht_img/`、`cyrillic_img/`等以`_img/`为后缀的文件夹中分别存放了10个语种的训练集和验证集数据。`train_list.txt`和`val_list.txt`分别为训练集和验证集的标签文件，`train_list.txt.debug`和`val_list.txt.debug`分别为训练集和验证集的`debug`标签文件，其分别是`train_list.txt`和`val_list.txt`的子集，用该文件可以快速体验本案例的流程。在10个文件夹中，还有部分补充文字数据，该集合和`train`集合的混合数据用于本案例的`SKL-UGI知识蒸馏策略`，对应的训练标签文件为`train_list_for_distill.txt`。



注意

本案例中数据类别共有10类，分别为：`0` 表示阿拉伯语（arabic）；`1` 表示中文繁体（chinese_cht）；`2` 表示斯拉夫语（cyrillic）；`3` 表示梵文（devanagari）；`4` 表示日语（japan）；`5` 表示卡纳达文（ka）；`6` 表示韩语（korean）；`7` 表示泰米尔文（ta）；`8` 表示泰卢固文（te）；`9` 表示拉丁语（latin）。



<a name="3.2"></a>

### 3.2 模型训练

<a name="3.2.1"></a>

#### 3.2.1 基于默认超参数训练

<a name="3.2.1.1"></a>

##### 3.2.1.1 基于默认超参数训练轻量级模型

在`ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml
```

验证集的最佳指标在0.99左右。

<a name="3.2.1.2"></a>

##### 3.2.1.2 基于默认超参数训练教师模型

复用`ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet50_vd
```

验证集的最佳指标为0.99左右，当前教师模型最好的权重保存在`output/ResNet50_vd/best_model.pdparams`。

<a name="3.2.1.3"></a>

##### 3.2.1.3 基于默认超参数进行蒸馏训练

配置文件`ppcls/configs/PULC/mlt/Distillation/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet50_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用训练集以外的文字数据作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/mlt/Distillation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet50_vd/best_model
```

验证集的最佳指标为0.991左右，当前模型最好的权重保存在`output/DistillationModel/best_model_student.pdparams`。

<a name="3.2.2"></a>

#### 3.2.2 超参数搜索训练

[3.2 小节](#3.2) 提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/StrategySearch/mlt.yaml
```

在`ppcls/configs/StrategySearch/mlt.yaml`中指定了具体的 GPU id 号和搜索配置, 默认搜索的训练日志和模型存放于`output/search_mlt`中，最终的蒸馏模型存放于`output/search_mlt/search_res/DistillationModel/best_model_student.pdparams`。

* **注意**:

* 3.1小节提供的默认配置已经经过了搜索，所以此过程不是必要的过程，如果自己的训练数据集有变化，可以尝试此过程。

* 此过程耗时较长，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/cls_demo/mlt/PPLCNet/PPLCNet_x1_0_search.yaml`中的`train_list.txt`和`val_list.txt`分别替换为`train_list.txt.debug`和`val_list.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。另外，搜索空间可以根据当前的机器资源来调整，如果机器资源有限，可以尝试缩小搜索空间，如果机器资源较充足，可以尝试扩大搜索空间。

* 如果此过程搜索的得到的超参数与[3.2.1小节](#3.2.1)提供的超参数不一致，主要是由于训练数据较小造成的波动导致，可以忽略。


<a name="4"></a>

## 4. 模型评估与推理


<a name="4.1"></a>

### 4.1 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/DistillationModel/best_model_student"
```

<a name="4.2"></a>

### 4.2 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs=./dataset/mlt/devanagari_img/ILSVRC2012_val_00010000_1820.jpg  \
    -o Global.pretrained_model=output/DistillationModel/best_model_student
```

输出结果如下：

```
[{'class_ids': [3, 4], 'scores': [0.83068, 0.08127], 'file_name': './dataset/mlt/devanagari_img/ILSVRC2012_val_00010000_1820.jpg', 'label_names': ['devanagari', 'japan']}]
```

<a name="4.3"></a>

### 4.3 使用 inference 模型进行推理

<a name="4.3.1"></a>

### 4.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/mlt/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_mlt
```
执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_mlt`文件夹，该文件夹中的模型与 2.2 节下载的推理预测模型格式一致。

<a name="4.3.2"></a>

### 4.3.2 基于 inference 模型推理预测
推理预测的脚本为：

```
python3.7 python/predict_cls.py -c configs/PULC/mlt/inference_mlt_cls.yaml -o Global.inference_model_dir="models/PPLCNet_x1_0_mlt"
```

**备注：**

- 更多关于推理的细节，可以参考[2.2节](#2.2)。
