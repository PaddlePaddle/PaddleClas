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
下图是使用的demo图片
![](../images/whl/demo.jpg)
结果展示
```
	>>> result
	[{'filename': '/Users/mac/Downloads/PaddleClas/docs/images/whl/demo.jpg', 'class_ids': array([8]), 'scores': array([0.9796774], dtype=float32), 'label_names': ['hen']}]
```


### 2. 代码使用方法
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

* 用户可以指定--label_name_path,作为用户自己训练模型的label_dict_file,格式应为(classid<space>class_name<\n>)
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