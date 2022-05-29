# PaddleClas构建车辆属性识别案例

此处提供了用户使用 PaddleClas 快速构建轻量级、高精度、可落地的车辆属性识别模型教程，主要基于车辆属性识别的数据，融合了轻量级骨干网络PPLCNet、SSLD预训练权重、EDA数据增强策略、SHAS超参数搜索策略，得到精度高、速度快、易于部署的车辆属性识别模型。

------


## 目录

- [1. 环境配置](#1)
- [2. 车辆属性识别推理预测](#2)
  - [2.1 下载模型](#2.1)  
  - [2.2 模型推理预测](#2.2)
      - [2.2.1 预测单张图像](#2.2.1)
      - [2.2.2 基于文件夹的批量预测](#2.2.2)
- [3.车辆属性识别场景训练](#3)
    - [3.1 数据准备](#3.1)
    - [3.2 模型训练](#3.2)
      - [3.2.1 基于默认超参数训练](#3.2.1)
        - [3.2.1.1 基于默认超参数训练轻量级模型](#3.2.1.1)
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

## 2. 车辆属性识别场景推理预测

<a name="2.1"></a>

### 2.1 下载模型

* 进入 `deploy` 运行目录。

```
cd deploy
```

下载车辆属性识别分类的模型。

```
mkdir models
cd models
# 下载inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/vehicle_attr_infer.tar && tar -xf vehicle_attr_infer.tar
```

解压完毕后，`models` 文件夹下应有如下文件结构：

```
├── vehicle_attr_infer
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

运行下面的命令，对图像 `./images/PULC/vehicle_attr/0002_c002_00030670_0.jpg` 进行车辆属性识别分类。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/vehicle_attr/inference_vehicle_attr.yaml
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_cls.py -c configs/PULC/vehicle_attr/inference_vehicle_attr.yaml -o Global.use_gpu=False
```

输出结果如下。

```
0002_c002_00030670_0.jpg:        attributes: Color: (yellow, prob: 0.9995124340057373), Type: (hatchback, prob: 0.933827817440033)
```

分别表示车的颜色为yellow，置信度为0.999，车的类型为hatchback，置信度为0.933。

<a name="2.2.2"></a>

#### 2.2.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 `Global.infer_imgs` 字段，也可以通过下面的 `-o` 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测，如果希望使用 CPU 预测，可以在命令后面添加 -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/vehicle_attr/inference_vehicle_attr.yaml -o Global.infer_imgs="./images/PULC/vehicle_attr/"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
0002_c002_00030670_0.jpg:        attributes: Color: (yellow, prob: 0.9995124340057373), Type: (hatchback, prob: 0.933827817440033)
```

<a name="3"></a>

## 3.车辆属性识别场景训练

<a name="3.1"></a>

### 3.1 数据准备

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

参考[VeRi数据转换教程(coming soon)]()，制作车辆属性识别数据集，并放在`dataset/`目录下，最终`dataset/`目录下的结构如下所示。

```
VeRi
├── image_train
│   ├── 0001_c001_00016450_0.jpg
│   ├── 0001_c001_00016460_0.jpg
│   ├── 0001_c001_00016470_0.jpg
...
├── image_test
│   ├── 0002_c002_00030600_0.jpg
│   ├── 0002_c002_00030605_1.jpg
│   ├── 0002_c002_00030615_1.jpg
...

...
├── train_list.txt
├── test_list.txt
├── label_list_train.txt.debug
├── label_list_test.txt.debug

```

其中`train/`和`test/`分别为训练集和验证集。`train_list.txt`和`test_list.txt`分别为训练集和验证集的标签文件，`train_list.txt.debug`和`test_list.txt.debug`分别为训练集和验证集的`debug`标签文件，其分别是`train_list.txt`和`test_list.txt`的子集，用该文件可以快速体验本案例的流程。

* **注意**:
    * 本案例中所使用的数据为[VeRi 数据集](https://www.v7labs.com/open-datasets/veri-dataset)。

<a name="3.2"></a>

### 3.2 模型训练

<a name="3.2.1"></a>

#### 3.2.1 基于默认超参数训练

<a name="3.2.1.1"></a>

##### 3.2.1.1 基于默认超参数训练轻量级模型

在`ppcls/configs/PULC/vehicle_attr/PPLCNet/PPLCNet_x1_0.yaml`中提供了基于该场景的训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/vehicle_attr/PPLCNet/PPLCNet_x1_0.yaml
```

验证集的最佳指标在0.910左右（数据集较小，可能有0.5%左右的精度波动）。

<a name="3.2.2"></a>

#### 3.2.2 超参数搜索训练

[3.2 小节](#3.2) 提供了在已经搜索并得到的超参数上进行了训练，此部分内容提供了搜索的过程，此过程是为了得到更好的训练超参数。

* 搜索运行脚本如下：

```shell
python tools/search_strategy.py -c ppcls/configs/StrategySearch/vehicle_attr.yaml
```

在`ppcls/configs/StrategySearch/vehicle_attr.yaml`中指定了具体的 GPU id 号和搜索配置, 默认搜索的训练日志和模型存放于`output/search_vehicle_attr`中。

* **注意**:
    * 3.1小节提供的默认配置已经经过了搜索，所以此过程不是必要的过程，如果自己的训练数据集有变化，可以尝试此过程。
    * 此过程基于当前数据集在 V100 4 卡上大概需要耗时 11 小时，如果缺少机器资源，希望体验搜索过程，可以将`ppcls/configs/demo/vehicle_attr/PPLCNet/PPLCNet_x1_0_search.yaml`中的`train_list.txt`和`test_list.txt`分别替换为`train_list.txt.debug`和`test_list.txt.debug`。替换list只是为了加速跑通整个搜索过程，由于数据量较小，其搜素的结果没有参考性。另外，搜索空间可以根据当前的机器资源来调整，如果机器资源有限，可以尝试缩小搜索空间，如果机器资源较充足，可以尝试扩大搜索空间。
    * 如果此过程搜索的得到的超参数与[3.2.1小节](#3.2.1)提供的超参数不一致，主要是由于训练数据较小造成的波动导致，可以忽略。


<a name="4"></a>

## 4. 模型评估与推理


<a name="4.1"></a>

### 4.1 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/vehicle_attr/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

<a name="4.2"></a>

### 4.2 模型预测


模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/vehicle_attr/PPLCNet/PPLCNet_x1_0.yaml \
    -o Infer.infer_imgs="./dataset/vehicle_attr/test/99603_17806.jpg"  \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

输出结果如下：

```
[{'attr': 'Color: (yellow, prob: 0.986522376537323), Type: (hatchback, prob: 0.9965125918388367)', 'pred': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'file_name': './deploy/images/PULC/vehicle_attr/0002_c002_00030670_0.jpg'}]
```

<a name="4.3"></a>

### 4.3 使用 inference 模型进行推理

<a name="4.3.1"></a>

### 4.3.1 导出 inference 模型

通过导出 inference 模型，PaddlePaddle 支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/vehicle_attr/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_vehicle_attr
```
执行完该脚本后会在`deploy/models/`下生成`PPLCNet_x1_0_vehicle_attr`文件夹，该文件夹中的模型与 2.2 节下载的推理预测模型格式一致。

<a name="4.3.2"></a>

### 4.3.2 基于 inference 模型推理预测

推理预测的脚本为：

```
python3.7 python/predict_cls.py -c configs/PULC/vehicle_attr/inference_vehicle_attr.yaml -o Global.inference_model_dir="models/PPLCNet_x1_0_vehicle_attr"
```

**备注：**

- 更多关于推理的细节，可以参考[2.2节](#2.2)。
