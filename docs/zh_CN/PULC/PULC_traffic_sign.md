# PaddleClas构建交通标志分类案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的，主要基于交通标志场景的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、SKL-UGI知识蒸馏策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的交通标志分类模型。

------


## 目录

- [1. 环境配置](#1)
- [2. 交通标志场景推理预测](#2)
  - [2.1 下载模型](#2.1)  
  - [2.2 模型推理预测](#2.2)
      - [2.2.1 预测单张图像](#2.2.1)
      - [2.2.2 基于文件夹的批量预测](#2.2.2)
- [3.交通标志分类场景训练](#3)
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

## 2. 交通标志分类场景推理预测

<a name="2.1"></a>

### 2.1 下载模型

* 进入 `deploy` 运行目录。

```
cd deploy
```

下载交通标志分类分类的模型。

```
mkdir models
cd models
# 下载inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/traffic_sign_cls_infer.tar && tar -xf traffic_sign_cls_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── traffic_sign_cls_infer
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

运行下面的命令，对图像 `./images/PULC/traffic_sign/objects365_02035329.jpg` 进行交通标志分类分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign_cls.yaml -o PostProcess.ThreshOutput.threshold=0.9794
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign_cls.yaml -o PostProcess.ThreshOutput.threshold=0.9794 -o Global.use_gpu=False
```

输出结果如下。

```
objects365_02035329.jpg:    class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```


**备注：** 真实场景中往往需要在假正类率（Fpr）小于某一个指标下求真正类率（Tpr），该场景中的`val`数据集在千分之一Fpr下得到的最佳Tpr所得到的阈值为`0.9794`，故此处的`threshold`为`0.9794`。该阈值的确定方法可以参考[3.2节](#3.2)

<a name="2.2.2"></a>

#### 2.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign_cls.yaml -o Global.infer_imgs="./images/PULC/traffic_sign/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
objects365_01780782.jpg:    class id(s): [0], score(s): [1.00], label_name(s): ['nobody']
objects365_02035329.jpg:    class id(s): [1], score(s): [1.00], label_name(s): ['someone']
```

其中，`someone` 表示该图里存在人，`nobody` 表示该图里不存在人。

<a name="3"></a>

## 3.交通标志分类场景训练

<a name="3.1"></a>

### 3.1 数据准备

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，下载并解压交通标志分类场景的数据。

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/cls_demo/traffic_sign.tar
tar -xf traffic_sign.tar
cd ../
```

执行上述命令后，`dataset/`下存在`traffic_sign`目录，该目录中具有以下数据：

```
traffic_sign
├── train
│   ├── 0_62627.jpg
│   ├── 100000_89031.jpg
│   ├── 100001_89031.jpg
...
├── test
│   ├── 100423_2315.jpg
│   ├── 100424_2315.jpg
│   ├── 100425_2315.jpg
...
├── other
│   ├── 100603_3422.jpg
│   ├── 100604_3422.jpg
...
├── label_list_train.txt
├── label_list_test.txt
├── label_list_other.txt
├── label_list_train_for_distillation.txt
├── label_list_train.txt.debug
├── label_list_test.txt.debug
├── label_name_id.txt
├── deal.py
```

其中`train/`和`test/`分别为训练集和验证集。`label_list_train.txt`和`label_list_test.txt`分别为训练集和验证集的标签文件，`label_list_train.txt.debug`和`label_list_test.txt.debug`分别为训练集和验证集的`debug`标签文件，其分别是`label_list_train.txt`和`label_list_test.txt`的子集，用该文件可以快速体验本案例的流程。`train`与`other`的混合数据用于本案例的`SKL-UGI知识蒸馏策略`，对应的训练标签文件为`label_list_train_for_distillation.txt`。

* **注意**:
    * 本案例中所使用的数据为[Tsinghua-Tencent 100K dataset (CC-BY-NC license)](https://cg.cs.tsinghua.edu.cn/traffic-sign/)，在使用的过程中，对交通标志检测框进行随机扩充与裁剪，从而得到用于训练与测试的图像，具体的处理脚本位于上述下载文件的`deal.py`文件中。

<a name="3.2"></a>

### 3.2 模型训练

<a name="3.2.1"></a>

#### 3.2.1 基于默认超参数训练

<a name="3.2.1.1"></a>

##### 3.2.1.1 基于默认超参数训练轻量级模型

在`ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml
```

验证集的最佳指标在0.981左右（数据集较小，可能有0.3%左右的精度波动）。

<a name="3.2.1.2"></a>

##### 3.2.1.2 基于默认超参数训练教师模型

复用`ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml`中的超参数，训练教师模型，训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

验证集的最佳指标为0.986左右，当前教师模型最好的权重保存在`output/ResNet101_vd/best_model.pdparams`。

<a name="3.2.1.3"></a>

##### 3.2.1.3 基于默认超参数进行蒸馏训练

配置文件`ppcls/configs/PULC/PULC/Distillation/PPLCNet_x1_0_distillation.yaml`提供了`SKL-UGI知识蒸馏策略`的配置。该配置将`ResNet101_vd`当作教师模型，`PPLCNet_x1_0`当作学生模型，使用ImageNet数据集的验证集作为新增的无标签数据。训练脚本如下：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/Distillation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

验证集的最佳指标为0.983左右，当前模型最好的权重保存在`output/DistillationModel/best_model_student.pdparams`。

<a name="3.2.2"></a>

#### 3.2.2 超参数搜索训练

[3.2 小节](#3.2) 提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/StrategySearch/traffic_sign.yaml
```

在`ppcls/configs/StrategySearch/traffic_sign.yaml`中指定了具体的 GPU id 号和搜索配置, 默认搜索的训练日志和模型存放于`output/search_traffic_sign`中，最终的蒸馏模型存放于`output/search_traffic_sign/search_res/DistillationModel/best_model_student.pdparams`。

* **注意**:
    * 3.1小节提供的默认配置已经经过了搜索，所以此过程不是必要的过程，如果自己的训练数据集有变化，可以尝试此过程。
    * 此过程基于当前数据集在 V100 4 卡上大概需要耗时 11 小时，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/cls_demo/traffic_sign/PPLCNet/PPLCNet_x1_0_search.yaml`中的`label_list_train.txt`和`label_list_test.txt`分别替换为`label_list_train.txt.debug`和`label_list_test.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。另外，搜索空间可以根据当前的机器资源来调整，如果机器资源有限，可以尝试缩小搜索空间，如果机器资源较充足，可以尝试扩大搜索空间。
    * 如果此过程搜索的得到的超参数与[3.2.1小节](#3.2.1)提供的超参数不一致，主要是由于训练数据较小造成的波动导致，可以忽略。


<a name="4"></a>

## 4. 模型评估与推理


<a name="4.1"></a>

### 4.1 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/DistillationModel/best_model_student"
```

<a name="4.2"></a>

### 4.2 模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs="./dataset/traffic_sign/test/99603_17806.jpg"  \
    -o Global.pretrained_model="output/DistillationModel/best_model_student"
```

输出结果如下：

```
todo
```

<a name="4.3"></a>

### 4.3 使用 inference 模型进行推理

<a name="4.3.1"></a>

### 4.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_traffic_sign
```
执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_traffic_sign`文件夹，该文件夹中的模型与 2.2 节下载的推理预测模型格式一致。

<a name="4.3.2"></a>

### 4.3.2 基于 inference 模型推理预测

推理预测的脚本为：

```
python3.7 python/predict_cls.py -c configs/PULC/traffic/inference_traffic_sign_cls.yaml -o Global.inference_model_dir="models/PPLCNet_x1_0_traffic_sign"
```

**备注：**

- 更多关于推理的细节，可以参考[2.2节](#2.2)。
