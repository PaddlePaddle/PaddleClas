# paddleclas package使用说明

## 快速上手

### 安装whl包

pip安装
```bash
pip install paddleclas==2.0.1
```

本地构建并安装
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddleclas-x.x.x-py3-none-any.whl # x.x.x是paddleclas的版本号，默认为0.0.0
```
### 1. 快速开始
* 指定`image_file='docs/images/whl/demo.jpg'`,使用Paddle提供的inference model,`model_name='ResNet50'`, 使用图片`docs/images/whl/demo.jpg`。

**下图是使用的demo图片**

<div align="center">
<img src="../images/whl/demo.jpg"  width = "400" />
</div>

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', top_k=5)
image_file='docs/images/whl/demo.jpg'
result=clas.predict(image_file)
print(result)
```

```
    >>> result
    [{'class_ids': array([ 8,  7, 86, 82, 80]), 'scores': array([9.7967714e-01, 2.0280687e-02, 2.7053760e-05, 6.1860351e-06,
       2.6378802e-06], dtype=float32), 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'black grouse'], 'filename': 'docs/images/whl/demo.jpg'}]
```

* 使用命令行式交互方法。直接获得结果。
```bash
paddleclas --model_name=ResNet50 --top_k=5 --image_file='docs/images/whl/demo.jpg'
```

```
    >>> result
    **********docs/images/whl/demo.jpg**********
    filename: docs/images/whl/demo.jpg; class id: 8, 7, 86, 82, 80; scores: 0.9797, 0.0203, 0.0000, 0.0000, 0.0000; label: hen, cock, partridge, ruffed grouse, partridge, Bonasa umbellus, black grouse
    Predict complete!
```

**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `reize=384`。


### 2. 参数解释
以下参数可在命令行交互使用时通过参数指定，或在Python代码中实例化PaddleClas对象时作为构造函数的参数使用。
* model_name(str): 模型名称，没有指定自定义的model_file和params_file时，可以指定该参数，使用PaddleClas提供的基于ImageNet1k的inference model，默认值为ResNet50。
* image_file(str or numpy.ndarray): 图像地址，支持指定单一图像的路径或图像的网址进行预测，支持指定包含图像的文件夹路径，支持numpy.ndarray格式的三通道图像数据，且通道顺序为[B, G, R]。
* use_gpu(bool): 是否使用GPU，如果使用，指定为True。默认为False。
* use_tensorrt(bool): 是否开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库。当使用TensorRT推理加速，指定为True。默认为False。
* is_preprocessed(bool): 当image_file为numpy.ndarray格式的图像数据时，图像数据是否已经过预处理。如果该参数为True，则不再对image_file数据进行预处理，否则将转换通道顺序后，按照resize_short，resize，normalize参数对图像进行预处理。默认值为False。
* resize_short(int): 将图像的高宽二者中小的值，调整到指定的resize_short值，大的值按比例放大。默认为256。
* resize(int): 将图像裁剪到指定的resize值大小，默认224。
* normalize(bool): 是否对图像数据归一化，默认True。
* batch_size(int): 预测时每个batch的样本数量，默认为1。
* model_file(str): 模型.pdmodel的路径，若不指定该参数，需要指定model_name，获得下载的模型。
* params_file(str): 模型参数.pdiparams的路径，若不指定，则需要指定model_name,以获得下载的模型。
* ir_optim(bool): 是否开启IR优化，默认为True。
* gpu_mem(int): 使用的GPU显存大小，默认为8000。
* enable_profile(bool): 是否开启profile功能，默认False。
* top_k(int): 指定的topk，打印（返回）预测结果的前k个类别和对应的分类概率，默认为1。
* enable_mkldnn(bool): 是否开启MKLDNN，默认False。
* cpu_num_threads(int): 指定cpu线程数，默认设置为10。
* label_name_path(str): 指定一个表示所有的label name的文件路径。当用户使用自己训练的模型，可指定这一参数，打印结果时可以显示图像对应的类名称。若用户使用Paddle提供的inference model，则可不指定该参数，使用imagenet1k的label_name，默认为空字符串。
* pre_label_image(bool): 是否需要进行预标注。
* pre_label_out_idr(str): 进行预标注后，输出结果的文件路径，默认为None。

**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `reize=384`。

### 3. 代码使用方法

**提供两种使用方式：1、python交互式编程。2、bash命令行式编程**

* 查看帮助信息

###### bash
```bash
paddleclas -h
```

* 用户使用自己指定的模型,需要指定模型路径参数`model_file`和参数`params_file`

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file='the path of model file',
    params_file='the path of params file')
image_file = 'docs/images/whl/demo.jpg' # image_file 可指定为前缀是https的网络图片，也可指定为本地图片
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_file='user-specified model path' --params_file='parmas path' --image_file='docs/images/whl/demo.jpg'
```

* 用户使用PaddlePaddle训练好的inference model来预测，并通过参数`model_name`指定。
此时无需指定`model_file`,模型会根据`model_name`自动下载指定模型到当前目录,并保存在目录`~/.paddleclas/`下以`model_name`命名的文件夹中。

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = 'docs/images/whl/demo.jpg' # image_file 可指定为前缀是https的网络图片，也可指定为本地图片
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/demo.jpg'
```

* 用户可以使用numpy.ndarray格式的图像数据，并通过参数`image_file`指定。注意该图像数据必须为三通道图像数据。如需对图像进行预处理，则图像通道顺序必须为[B, G, R]。

###### python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = cv2.imread("docs/images/whl/demo.jpg")
result=clas.predict(image_file)
```

* 用户可以将`image_file`指定为包含图片的文件夹路径。

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = 'docs/images/whl/' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/'
```

* 用户可以指定`pre_label_image=True`, `pre_label_out_idr='./output_pre_label/'`，将图片按其top1预测结果保存到`pre_label_out_dir`目录下对应类别的文件夹中。

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', pre_label_image=True,pre_label_out_idr='./output_pre_label/')
image_file = 'docs/images/whl/' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/' --pre_label_image=True --pre_label_out_idr='./output_pre_label/'
```

* 用户可以通过参数`label_name_path`指定模型的`label_dict_file`文件路径，文件内容格式应为(class_id<space>class_name<\n>)，例如：

```
0 tench, Tinca tinca
1 goldfish, Carassius auratus
2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
......
```

* 用户如果使用Paddle提供的inference model，则不需要提供`label_name_path`，会默认使用`ppcls/utils/imagenet1k_label_list.txt`。
如果用户希望使用自己的模型，则可以提供`label_name_path`，将label_name与结果一并输出。如果不提供将不会输出label_name信息。


###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file='the path of model file', params_file ='the path of params file', label_name_path='./ppcls/utils/imagenet1k_label_list.txt')
image_file = 'docs/images/whl/demo.jpg' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_file='the path of model file' --params_file='the path of params file' --image_file='docs/images/whl/demo.jpg' --label_name_path='./ppcls/utils/imagenet1k_label_list.txt'
```

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = 'docs/images/whl/' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/'
```
