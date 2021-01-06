# paddleclas package使用说明

## 快速上手

### 安装whl包

pip安装
```bash
pip install paddleclas=2.0.0rc1
```

本地构建并安装
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddleclas-x.x.x-py3-none-any.whl # x.x.x是paddleclas的版本号
```
### 1. 快速开始
* 指定image_file='docs/images/whl/demo.jpg',使用Paddle提供的inference model,model_name=ResNet50。
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50',use_gpu=False,use_tensorrt=False)
image_file='docs/images/whl/demo.jpg'
result=clas.predict(image_file)
print(result)
```
**下图是使用的demo图片**
![](../images/whl/demo.jpg)
**结果展示**
```
    >>> result
    [{'filename': '/Users/mac/Downloads/PaddleClas/docs/images/whl/demo.jpg', 'class_ids': array([8]), 'scores': array([0.9796774], dtype=float32), 'label_names': ['hen']}]
```

### 2. 参数解释
* model_name(str): 模型名称，没有指定自定义的model_file和params_file时，可以指定该参数，使用PaddleClas提供的基于ImageNet1k的inference model，默认值为ResNet50。
* image_file(str): 图像地址，支持指定单一图像的路径或图像的网址进行预测，支持指定包含图像的文件夹路径。
* use_gpu(bool): 是否使用GPU，如果使用，指定为True。默认为False。
* use_tensorrt(bool): 是否开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库。当使用TensorRT推理加速，指定为True。默认为False。
* resize_short(int): 将图像的高宽二者中小的值，调整到指定的resize_short值，大的值按比例放大。默认为256。
* resize(int): 将图像裁剪到指定的resize值大小，默认224。
* normalize(bool): 是否对图像数据归一化，默认True。
* batch_size(int): 预测时每个batch的样本数，默认为1。
* model_file(str): 模型.pdmodel的路径，若不指定该参数，需要指定model_name，获得下载的模型。
* params_file(str): 模型参数.pdiparams的路径，若不与model_file指定，则需要指定model_name,以获得下载的模型。
* ir_optim(bool): 是否开启IR优化，默认为True。
* gpu_mem(int): 使用的GPU显存大小，默认为8000。
* enable_profile(bool): 是否开启profile功能，默认False。
* top_k(int): 指定的topk，预测的前k个类别和对应的分类概率，默认为1。
* enable_mkldnn(bool): 是否开启MKLDNN，默认False。
* cpu_num_threads(int): 指定cpu线程数，默认设置为10。
* label_name_path(str): 指定一个表示所有的label name的文件路径。当用户使用自己训练的模型，可指定这一参数，打印结果时可以显示图像对应的类名称。若用户使用Paddle提供的inference model，则可不指定该参数，默认使用imagenet1k的label_name。
* pre_label_image(bool): 是否需要进行预标注。
* pre_label_out_idr(str): 进行预标注后，输出结果的文件路径。

### 3. 代码使用方法
* 用户使用自己指定的模型,需要指定模型路径参数model_file和参数params_file

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file='user-specified model path',
    params_file='parmas path', use_gpu=False, use_tensorrt=False)
image_file = '' # image_file 可指定为前缀是https的网络图片，也可指定为本地图片
result=clas.predict(image_file)
print(result)
```

* 用户使用PaddlePaddle训练好的inference model来预测，用户需要使用，初始化打印的模型的其中一个，并指定给model_name。
用户可以不指定model_file,模型会自动下载到当前目录,并保存在以model_name命名的文件夹中，避免下载不同模型的覆盖问题。

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50',use_gpu=False, use_tensorrt=False)
image_file = '' # image_file 可指定为前缀是https的网络图片，也可指定为本地图片
result=clas.predict(image_file)
print(result)
```

* 用户可以将image_file 指定为包含图片的文件夹路径，可以指定top_k参数

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50',use_gpu=False, use_tensorrt=False,top_k=5)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

* 用户可以指定--pre_label_image=True, --pre_label_out_idr= './output_pre_label/'，将图片复制到，以其top1对应的类别命名的文件夹中。

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50',use_gpu=False, use_tensorrt=False,top_k=5, pre_label_image=True,pre_label_out_idr='./output_pre_label/')
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

* 用户可以指定--label_name_path,作为用户自己训练模型的label_dict_file,格式应为(class_id<space>class_name<\n>)

```
0 tench, Tinca tinca
1 goldfish, Carassius auratus
2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
......
```

用户如果使用Paddle提供的inference model，则不需要提供label_name_path，会默认使用ppcls/utils/imagenet1k_label_list.txt。
如果用户希望使用自己的模型，则可以提供label_name_path，将label_name与结果一并输出。如果不提供将不会输出label_name信息。

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file= './inference.pdmodel',params_file = './inference.pdiparams',label_name_path='./ppcls/utils/imagenet1k_label_list.txt',use_gpu=False)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50',use_gpu=False)
image_file = '' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```
