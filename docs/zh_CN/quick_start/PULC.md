# PULC 快速体验

------

本文主要介绍通过 PaddleClas whl 包，使用 PULC 系列模型进行预测。

## 目录

- [1. 安装](#1)
  - [1.1 安装PaddlePaddle](#11)
  - [1.2 安装PaddleClas whl包](#12)
- [2. 快速体验](#2)
  - [2.1 命令行使用](#2.1)
  - [2.2 Python脚本使用](#2.2)
  - [2.3 模型列表](#2.3)
- [3.小结](#3)

<a name="1"></a>

## 1. 安装

<a name="1.1"></a>

### 1.1 安装 PaddlePaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- 您的机器是CPU，请运行以下命令安装

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

更多的版本需求，请参照[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

<a name="1.2"></a>

### 1.2 安装 PaddleClas whl 包

```bash
pip3 install paddleclas
```

<a name="2"></a>

## 2. 快速体验

PaddleClas 提供了一系列测试图片，里边包含人、车、OCR等方向的多个场景的demo数据。点击[这里](https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip)下载并解压，然后在终端中切换到相应目录。

<a name="2.1"></a>

### 2.1 命令行使用

```
cd /path/to/pulc_demo_imgs
```

使用命令行预测：

```bash
paddleclas --model_name=person_exists --infer_imgs=pulc_demo_imgs/person_exists/objects365_01780782.jpg
```

结果如下：
```
>>> result
class_ids: [0], scores: [0.9955421453341842], label_names: ['nobody'], filename: pulc_demo_imgs/person_exists/objects365_01780782.jpg
Predict complete!
```

若预测结果为 `nobody`，表示该图中没有人，若预测结果为 `someone`，则表示该图中有人。此处预测结果为 `nobody`，表示该图中没有人。

**备注**： 更换其他预测的数据时，只需要改变 `--infer_imgs=xx` 中的字段即可，支持传入整个文件夹，如需要替换模型，更改 `--model_name` 中的模型名字即可，模型名字可以参考[2.3 模型列表](#2.3)。

<a name="2.2"></a>

### 2.2 Python 脚本使用

此处提供了在 python 脚本中使用 PULC 有人/无人分类模型预测的例子。

```python
import paddleclas
model = paddleclas.PaddleClas(model_name="person_exists")
result = model.predict(input_data="pulc_demo_imgs/person_exists/objects365_01780782.jpg")
print(next(result))
```

打印的结果如下：

```
>>> result
[{'class_ids': [0], 'scores': [0.9955421453341842], 'label_names': ['nobody'], 'filename': 'pulc_demo_imgs/person_exists/objects365_01780782.jpg'}]
```

**备注**：`model.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果, 默认 `batch_size` 为 1，如果需要更改 `batch_size`，实例化模型时，需要指定 `batch_size`，如 `model = paddleclas.PaddleClas(model_name="person_exists",  batch_size=2)`。更换其他模型只需要替换`model_name`, `model_name`,可以参考[2.3 模型列表](#2.3)。

<a name="2.3"></a>

### 2.3 模型列表

PULC 系列模型的名称和简介如下：

|模型名称|模型简介|
| --- | --- |
| person_exists | PULC有人/无人分类模型 |
| person_attribute | PULC人体属性识别模型 |
| safety_helmet | PULC佩戴安全帽分类模型 |
| traffic_sign | PULC交通标志分类模型 |
| vehicle_attribute | PULC车辆属性识别模型 |
| car_exists | PULC有车/无车分类模型 |
| text_image_orientation | PULC含文字图像方向分类模型 |
| textline_orientation | PULC文本行方向分类模型 |
| language_classification | PULC语种分类模型 |

<a name="3"></a>

## 3. 小结

通过本节内容，相信您已经熟练掌握 PaddleClas whl 包的 PULC 模型使用方法并获得了初步效果。

PULC 方法产出的系列模型在人、车、OCR等方向的多个场景中均验证有效，用超轻量模型就可实现与 SwinTransformer 模型接近的精度，预测速度提高 40+ 倍。并且打通数据、模型训练、压缩和推理部署全流程，具体地，您可以参考[PULC有人/无人分类模型](PULC_person_exists.md)、[PULC人体属性识别模型](PULC_person_attribute.md)、[PULC佩戴安全帽分类模型](PULC_safety_helmet.md)、[PULC交通标志分类模型](PULC_traffic_sign.md)、[PULC车辆属性识别模型](PULC_vehicle_attribute.md)、[PULC有车/无车分类模型](PULC_car_exists.md)、[PULC含文字图像方向分类模型](PULC_text_image_orientation.md)、[PULC文本行方向分类模型](PULC_textline_orientation.md)、[PULC语种分类模型](PULC_language_classification.md)。
