# paddleclas package

## Get started quickly

### install package

install by pypi
```bash
pip install paddleclas==2.0.0rc3
```

build own whl package and install
```bash
python3 setup.py bdist_wheel
pip3 install dist/paddleclas-x.x.x-py3-none-any.whl
```

### 1. Quick Start

* Assign `image_file='docs/images/whl/demo.jpg'`, Use inference model that Paddle provides `model_name='ResNet50'`

**Here is demo.jpg**

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
       2.6378802e-06], dtype=float32), 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'black grouse'], 'filename': 'docs/images/whl/demo.jpg'}
```

* Using command line interactive programming
```bash
paddleclas --model_name=ResNet50 --top_k=5 --image_file='docs/images/whl/demo.jpg'
```

```
    >>> result
    **********docs/images/whl/demo.jpg**********
    filename: docs/images/whl/demo.jpg; class id: 8, 7, 86, 82, 80; scores: 0.9797, 0.0203, 0.0000, 0.0000, 0.0000; label: hen, cock, partridge, ruffed grouse, partridge, Bonasa umbellus, black grouse
    Predict complete!
```

### 2. Definition of Parameters
* model_name(str): model's name. If not assigning `model_file`and`params_file`, you can assign this param. If using inference model based on ImageNet1k provided by Paddle, set as default='ResNet50'.
* image_file(str or numpy.ndarray): image's path. Support assigning single local image, internet image and folder containing series of images. Also Support numpy.ndarray, the channel order is [B, G, R].
* use_gpu(bool): Whether to use GPU or not, defalut=False。
* use_tensorrt(bool): whether to open tensorrt or not. Using it can greatly promote predict preformance, default=False.
* is_preprocessed(bool): Assign the image data has been preprocessed or not when the image_file is numpy.ndarray.
* resize_short(int): resize the minima between height and width into resize_short(int), default=256
* resize(int): resize image into resize(int), default=224.
* normalize(bool): whether normalize image or not, default=True.
* batch_size(int): batch number, default=1.
* model_file(str): path of inference.pdmodel. If not assign this param，you need assign `model_name` for downloading.
* params_file(str): path of inference.pdiparams. If not assign this param，you need assign `model_name` for downloading.
* ir_optim(bool): whether enable IR optimization or not, default=True.
* gpu_mem(int): GPU memory usages，default=8000。
* enable_profile(bool): whether enable profile or not,default=False.
* top_k(int): Assign top_k, default=1.
* enable_mkldnn(bool): whether enable MKLDNN or not, default=False.
* cpu_num_threads(int): Assign number of cpu threads, default=10.
* label_name_path(str): Assign path of label_name_dict you use. If using your own training model, you can assign this param. If using inference model based on ImageNet1k provided by Paddle, you may not assign this param.Defaults take ImageNet1k's label name.
* pre_label_image(bool): whether prelabel or not, default=False.
* pre_label_out_idr(str): If prelabeling, the path of output.

### 3. Different Usages of Codes

**We provide two ways to use: 1. Python interative programming 2. Bash command line programming**

* check `help` information
```bash
paddleclas -h
```

* Use user-specified model, you need to assign model's path `model_file` and parameters's path`params_file`

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file='the path of model file',
    params_file='the path of params file')
image_file = 'docs/images/whl/demo.jpg'
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_file='user-specified model path' --params_file='parmas path' --image_file='docs/images/whl/demo.jpg'
```

* Use inference model which PaddlePaddle provides to predict, you need to choose one of model proviede by PaddleClas to assign `model_name`. So there's no need to assign `model_file`. And the model you chosen will be download in `~/.paddleclas/`, which will be saved in folder named by `model_name`.

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = 'docs/images/whl/demo.jpg'
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/demo.jpg'
```

* You can assign input as format `numpy.ndarray` which has been preprocessed `image_file=np.ndarray`. Note that the image data must be three channel. If need To preprocess the image, the image channels order must be [B, G, R].

###### python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
image_file = cv2.imread("docs/images/whl/demo.jpg")
result=clas.predict(image_file)
```

* You can assign `image_file` as a folder path containing series of images.

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

* You can assign `--pre_label_image=True`, `--pre_label_out_idr= './output_pre_label/'`. Then images will be copied into folder named by top-1 class_id.

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50', pre_label_image=True, pre_label_out_idr='./output_pre_label/')
image_file = 'docs/images/whl/' # it can be image_file folder path which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/' --pre_label_image=True --pre_label_out_idr='./output_pre_label/'
```

* You can assign `--label_name_path` as your own label_dict_file, format should be as(class_id<space>class_name<\n>).

```
0 tench, Tinca tinca
1 goldfish, Carassius auratus
2 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
......
```

* If you use inference model that PaddleClas provides, you do not need assign `label_name_path`. Program will take `ppcls/utils/imagenet1k_label_list.txt` as defaults. If you hope using your own training model, you can provide `label_name_path` outputing 'label_name' and scores, otherwise no 'label_name' in output information.

###### python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_file= 'the path of model file', params_file = 'the path of params file', label_name_path='./ppcls/utils/imagenet1k_label_list.txt')
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
image_file = 'docs/images/whl/' # it can be directory which contains all of images you want to predict.
result=clas.predict(image_file)
print(result)
```

###### bash
```bash
paddleclas --model_name='ResNet50' --image_file='docs/images/whl/'
```
