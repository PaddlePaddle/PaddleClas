# PaddleClas Whl 使用说明

PaddleClas 支持 Python Whl 包方式进行预测，目前 Whl 包方式仅支持图像分类，暂不支持主体检测、特征提取及向量检索。

---

## 目录

- [1. 安装 paddleclas](#1)
- [2. 快速开始](#2)
- [3. 参数解释](#3)
- [4. 使用示例](#4)
   - [4.1 查看帮助信息](#4.1)
   - [4.2 使用 PaddleClas 提供的预训练模型进行预测](#4.2)
   - [4.3 使用本地模型文件预测](#4.3)
   - [4.4 批量预测](#4.4)
   - [4.5 对网络图片进行预测](#4.5)
   - [4.6 对 `NumPy.ndarray` 格式数据进行预测](#4.6)
   - [4.7 保存预测结果](#4.7)
   - [4.8 指定 label name](#4.8)


<a name="1"></a>
## 1. 安装 paddleclas

* pip 安装

```bash
pip3 install paddleclas==2.2.1
```

* 本地构建并安装

```bash
python3 setup.py bdist_wheel
pip3 install dist/*
```

<a name="2"></a>
## 2. 快速开始
* 使用 `ResNet50` 模型，以下图（`PaddleClas/docs/images/inference_deployment/whl_demo.jpg`）为例进行说明。

![](../../images/inference_deployment/whl_demo.jpg)


* 在 Python 代码中使用
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs='docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

**注意**：`PaddleClas.predict()` 为可迭代对象（`generator`），因此需要使用 `next()` 函数或 `for` 循环对其迭代调用。每次调用将以 `batch_size` 为单位进行一次预测，并返回预测结果。返回结果示例如下：

```
>>> result
[{'class_ids': [8, 7, 136, 80, 84], 'scores': [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], 'label_names': ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']}]
```

* 在命令行中使用
```bash
paddleclas --model_name=ResNet50  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```

```
>>> result
filename: docs/images/inference_deployment/whl_demo.jpg, top-5, class_ids: [8, 7, 136, 80, 84], scores: [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], label_names: ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']
Predict complete!
```

<a name="3"></a>
## 3. 参数解释
以下参数可在命令行方式使用中通过参数指定，或在 Python 代码中实例化 PaddleClas 对象时作为构造函数的参数使用。
* model_name(str): 模型名称，使用 PaddleClas 提供的基于 ImageNet1k 的预训练模型。
* inference_model_dir(str): 本地模型文件目录，当未指定 `model_name` 时该参数有效。该目录下需包含 `inference.pdmodel` 和 `inference.pdiparams` 两个模型文件。
* infer_imgs(str): 待预测图片文件路径，或包含图片文件的目录，或网络图片的 URL。
* use_gpu(bool): 是否使用 GPU，默认为 `True`。
* gpu_mem(int): 使用的 GPU 显存大小，当 `use_gpu` 为 `True` 时有效，默认为 8000。
* use_tensorrt(bool): 是否开启 TensorRT 预测，可提升 GPU 预测性能，需要使用带 TensorRT 的预测库，默认为 `False`。
* enable_mkldnn(bool): 是否开启 MKLDNN，当 `use_gpu` 为 `False` 时有效，默认 `False`。
* cpu_num_threads(int): CPU 预测时的线程数，当 `use_gpu` 为 `False` 且 `enable_mkldnn` 为 `True` 时有效，默认值为 `10`。
* batch_size(int): 预测时每个 batch 的样本数量，默认为 `1`。
* resize_short(int): 按图像较短边进行等比例缩放，默认为 `256`。
* crop_size(int): 将图像裁剪到指定大小，默认为 `224`。
* topk(int): 打印（返回）预测结果的前 `topk` 个类别和对应的分类概率，默认为 `5`。
* class_id_map_file(str): `class id` 与 `label` 的映射关系文件。默认使用 `ImageNet1K` 数据集的映射关系。
* save_dir(str): 将预测结果作为预标注数据保存的路径，默认为 `None`，即不保存。

**注意**: 如果使用 `Transformer` 系列模型，如 `DeiT_***_384`, `ViT_***_384` 等，请注意模型的输入数据尺寸，需要设置参数 `resize_short=384`, `crop_size=384`，如下所示。

* 命令行中
```bash
from paddleclas import PaddleClas, get_default_confg
paddleclas --model_name=ViT_base_patch16_384 --infer_imgs='docs/images/inference_deployment/whl_demo.jpg' --resize_short=384 --crop_size=384
```

* Python 代码中
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ViT_base_patch16_384', resize_short=384, crop_size=384)
```

<a name="4"></a>
## 4. 使用示例

PaddleClas 提供两种使用方式：
1. Python 代码中使用；
2. 命令行中使用。

<a name="4.1"></a>
### 4.1 查看帮助信息

* CLI
```bash
paddleclas -h
```

<a name="4.2"></a>
### 4.2 使用 PaddleClas 提供的预训练模型进行预测
可以使用 PaddleClas 提供的预训练模型来预测，并通过参数 `model_name` 指定。此时 PaddleClas 会根据 `model_name` 自动下载指定模型，并保存在目录`~/.paddleclas/`下。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/inference_deployment/whl_demo.jpg'
```

<a name="4.3"></a>
### 4.3 使用本地模型文件预测
可以使用本地的模型文件进行预测，通过参数 `inference_model_dir` 指定模型文件目录即可。需要注意，模型文件目录下必须包含 `inference.pdmodel` 和 `inference.pdiparams` 两个文件。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(inference_model_dir='./inference/')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --inference_model_dir='./inference/' --infer_imgs='docs/images/inference_deployment/whl_demo.jpg'
```

<a name="4.4"></a>
### 4.4 批量预测
当参数 `infer_imgs` 为包含图片文件的目录时，可以对图片进行批量预测，只需通过参数 `batch_size` 指定 batch 大小。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', batch_size=2)
infer_imgs = 'docs/images/'
result=clas.predict(infer_imgs)
for r in result:
    print(r)
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/' --batch_size 2
```

<a name="4.5"></a>
### 4.5 对网络图片进行预测
可以对网络图片进行预测，只需通过参数 `infer_imgs` 指定图片 `url`。此时图片会下载并保存在`~/.paddleclas/images/`目录下。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'https://raw.githubusercontent.com/paddlepaddle/paddleclas/release/2.2/docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='https://raw.githubusercontent.com/paddlepaddle/paddleclas/release/2.2/docs/images/inference_deployment/whl_demo.jpg'
```

<a name="4.6"></a>
### 4.6 对 `NumPy.ndarray` 格式数据进行预测
在 Python 中，可以对 `Numpy.ndarray` 格式的图像数据进行预测，只需通过参数 `infer_imgs` 指定即可。注意，PaddleClas 所提供的模型仅支持 3 通道图像数据，且通道顺序为 `RGB`。

* python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = cv2.imread("docs/images/inference_deployment/whl_demo.jpg")[:, :, ::-1]
result=clas.predict(infer_imgs)
print(next(result))
```

<a name="4.7"></a>
### 4.7 保存预测结果
可以指定参数 `pre_label_out_dir='./output_pre_label/'`，将图片按其 top1 预测结果保存到 `pre_label_out_dir` 目录下对应类别的文件夹中。

* python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', save_dir='./output_pre_label/')
infer_imgs = 'docs/images/' # it can be infer_imgs folder path which contains all of images you want to predict.
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/' --save_dir='./output_pre_label/'
```

<a name="4.8"></a>
### 4.8 指定 label name
可以通过参数 `class_id_map_file` 指定 `class id` 与 `lable` 的对应关系。PaddleClas 默认使用 ImageNet1K 的 label_name（`ppcls/utils/imagenet1k_label_list.txt`）。

`class_id_map_file` 文件内容格式应为：

```
class_id<space>class_name<\n>
```

例如：

```
0 tench, Tinca tinca
1 goldfish, Carassius auratus
2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
......
```

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/inference_deployment/whl_demo.jpg' --class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt'
```
