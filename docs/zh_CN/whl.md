# PaddleClas wheel package使用说明

## 安装

* pip安装

```bash
pip install paddleclas==2.2.0
```

* 本地构建并安装

```bash
python3 setup.py bdist_wheel
pip3 install dist/*
```


## 快速开始
* 使用`ResNet50`模型，以下图（`'docs/images/whl/demo.jpg'`）为例进行测试。

<div align="center">
<img src="../images/whl/demo.jpg"  width = "400" />
</div>


* 在Python代码中调用
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs='docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

```
>>> result
[{'class_ids': [8, 7, 136, 80, 84], 'scores': [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], 'label_names': ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']}]
```

* 使用命令行式交互方法
```bash
paddleclas --model_name=ResNet50  --infer_imgs="docs/images/whl/demo.jpg"
```

```
>>> result
filename: docs/images/whl/demo.jpg, top-5, class_ids: [8, 7, 136, 80, 84], scores: [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], label_names: ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']
Predict complete!
```


## 参数解释
以下参数可在命令行交互使用时通过参数指定，或在Python代码中实例化PaddleClas对象时作为构造函数的参数使用。
* model_name(str): 模型名称，使用PaddleClas提供的基于ImageNet1k的预训练模型。
* inference_model_dir: 本地模型文件目录，当未至定 `model_name` 时该参数有效。该目录下需包含 `inference.pdmodel` 和 `inference.pdiparams` 两个模型文件。
* infer_imgs(str): 待预测图片文件路径，或包含图片文件的目录，或图片的URL。
* use_gpu(bool): 是否使用GPU，默认为 `True`。
* batch_size(int): 预测时每个batch的样本数量，默认为 `1`。
* use_tensorrt(bool): 是否开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库，默认为 `False`。
* resize_short(int): 按图像较短边进行等比例缩放,默认为 `256`。
* crop_size(int): 将图像裁剪到指定大小，默认为 `224`。
* gpu_mem(int): 使用的GPU显存大小，当 `use_gpu` 为 `True` 时有效，默认为8000。
* topk(int): 打印（返回）预测结果的前 `topk` 个类别和对应的分类概率，默认为1。
* enable_mkldnn(bool): 是否开启MKLDNN，当 `use_gpu` 为 `False` 时有效，默认 `False`。
* cpu_num_threads(int): cpu预测时的线程数，当 `use_gpu` 为 `False` 时有效，默认设置为 `10`。
* class_id_map_file(str): `class id` 与 `label` 的映射关系文件。默认使用 `ImageNet1K` 数据集的映射关系。
* save_dir(str): 将预测结果作为预标注数据保存的路径，默认为 `None`，即不保存。


**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `crop_size=384`，如下所示。

* 在命令行中使用：
```bash
from paddleclas import PaddleClas, get_default_confg
paddleclas --model_name=ViT_base_patch16_384 --infer_imgs='docs/images/whl/demo.jpg' --resize_short=384 --crop_size=384
```

* 在python代码中：
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ViT_base_patch16_384', resize_short=384, crop_size=384)
```


## 使用示例

PaddleClas提供两种使用方式：
1. Python接口调用；
2. 命令行式使用。


### 查看帮助信息

* CLI
```bash
paddleclas -h
```


### 使用PaddleClas提供的预训练模型进行预测
可以使用PaddleClas提供的预训练模型来预测，并通过参数`model_name`指定。此时PaddleClas会根据`model_name`自动下载指定模型，并保存在目录`~/.paddleclas/`下。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/whl/demo.jpg'
```


### 使用本地模型文件预测
可以使用本地的模型文件进行预测，通过参数`inference_model_dir`指定模型文件目录即可。需要注意，模型文件目录下必须包含`inference.pdmodel`和`inference.pdiparams`两个文件。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(inference_model_dir='./inference/')
infer_imgs = 'docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --inference_model_dir='./inference/' --infer_imgs='docs/images/whl/demo.jpg'
```


### 批量预测
可以对指定目录下的图片批量预测，只需通过参数`infer_imgs`指定图片所在目录。此时，PaddleClas会以`batch_size`为单位进行预测。

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


### 对网络图片进行预测
可以对网络图片进行预测，只需通过参数`infer_imgs`指定图片`url`。此时图片会下载并保存在`~/.paddleclas/images/`目录下。

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'https://github.com/paddlepaddle/paddleclas/blob/release%2F2.2/docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='https://github.com/paddlepaddle/paddleclas/blob/release%2F2.2/docs/images/whl/demo.jpg'
```


### 对`NumPy.ndarray`格式数据进行预测
在Python中，可以对`Numpy.ndarray`格式的图像数据进行预测，只需通过参数`infer_imgs`指定即可。注意该图像数据必须为三通道图像数据。

* python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = cv2.imread("docs/images/whl/demo.jpg")
result=clas.predict(infer_imgs)
print(next(result))
```


### 保存预测结果
可以指定参数`pre_label_out_dir='./output_pre_label/'`，将图片按其top1预测结果保存到`pre_label_out_dir`目录下对应类别的文件夹中。

* python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', save_dir='./output_pre_label/')
infer_imgs = 'docs/images/whl/' # it can be infer_imgs folder path which contains all of images you want to predict.
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/whl/' --save_dir='./output_pre_label/'
```


### 指定label name
可以通过参数`class_id_map_file`指定`class id`与`lable`的对应关系。PaddleClas默认使用ImageNet1K的label_name（`ppcls/utils/imagenet1k_label_list.txt`）。如果不提供将不会输出label_name信息。

`class_id_map_file`文件内容格式应为：

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

* python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt')
infer_imgs = 'docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/whl/demo.jpg' --class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt'
```
