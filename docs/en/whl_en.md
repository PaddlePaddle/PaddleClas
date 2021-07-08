# PaddleClas wheel package

## 1. Installation

* installing from pypi

```bash
pip3 install paddleclas==2.2.0
```

* build own whl package and install

```bash
python3 setup.py bdist_wheel
pip3 install dist/*
```


## 2. Quick Start
* Using the `ResNet50` model provided by PaddleClas, the following image(`'docs/images/whl/demo.jpg'`) as an example.

<div align="center">
<img src="../images/whl/demo.jpg"  width = "400" />
</div>

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs='docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

**Note**: `PaddleClas.predict()` is a `generator`. Therefore you need to use `next()` or `for` call it iteratively. It will perform a prediction by `batch_size` and return the prediction result(s) when called. Examples of returned results are as follows:

```
>>> result
[{'class_ids': [8, 7, 136, 80, 84], 'scores': [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], 'label_names': ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']}]
```

* CLI
```bash
paddleclas --model_name=ResNet50  --infer_imgs="docs/images/whl/demo.jpg"
```

```
>>> result
filename: docs/images/whl/demo.jpg, top-5, class_ids: [8, 7, 136, 80, 84], scores: [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], label_names: ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']
Predict complete!
```




## 3. Definition of Parameters

The following parameters can be specified in Command Line or used as parameters of the constructor when instantiating the PaddleClas object in Python.
* model_name(str): If using inference model based on ImageNet1k provided by Paddle, please specify the model's name by the parameter.
* inference_model_dir(str): Local model files directory, which is valid when `model_name` is not specified. The directory should contain `inference.pdmodel` and `inference.pdiparams`.
* infer_imgs(str): The path of image to be predicted, or the directory containing the image files, or the URL of the image from Internet.
* use_gpu(bool): Whether to use GPU or not, default by `True`.
* gpu_mem(int): GPU memory usages，default by `8000`。
* use_tensorrt(bool): Whether to open TensorRT or not. Using it can greatly promote predict preformance, default by `False`.
* enable_mkldnn(bool): Whether enable MKLDNN or not, default `False`.
* cpu_num_threads(int): Assign number of cpu threads, valid when `--use_gpu` is `False` and `--enable_mkldnn` is `True`, default by `10`.
* batch_size(int): Batch size, default by `1`.
* resize_short(int): Resize the minima between height and width into `resize_short`, default by `256`.
* crop_size(int): Center crop image to `crop_size`, default by `224`.
* topk(int): Print (return) the `topk` prediction results, default by `5`.
* class_id_map_file(str): The mapping file between class ID and label, default by `ImageNet1K` dataset's mapping.
* pre_label_image(bool): whether prelabel or not, default=False.
* save_dir(str): The directory to save the prediction results that can be used as pre-label, default by `None`, that is, not to save.

**Note**: If you want to use `Transformer series models`, such as `DeiT_***_384`, `ViT_***_384`, etc., please pay attention to the input size of model, and need to set `resize_short=384`, `resize=384`. The following is a demo.

* CLI:
```bash
from paddleclas import PaddleClas, get_default_confg
paddleclas --model_name=ViT_base_patch16_384 --infer_imgs='docs/images/whl/demo.jpg' --resize_short=384 --crop_size=384
```

* Python:
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ViT_base_patch16_384', resize_short=384, crop_size=384)
```

## 4. Usage

PaddleClas provides two ways to use:
1. Python interative programming;
2. Bash command line programming.

### 4.1 View help information

* CLI
```bash
paddleclas -h
```

### 4.2 Prediction using inference model provide by PaddleClas
You can use the inference model provided by PaddleClas to predict, and only need to specify `model_name`. In this case, PaddleClas will automatically download files of specified model and save them in the directory `~/.paddleclas/`.

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


### 4.3 Prediction using local model files
You can use the local model files trained by yourself to predict, and only need to specify `inference_model_dir`. Note that the directory must contain `inference.pdmodel` and `inference.pdiparams`.

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

### 4.4 Prediction by batch
You can predict by batch, only need to specify `batch_size` when `infer_imgs` is direcotry contain image files.

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


### 4.5 Prediction of Internet image
You can predict the Internet image, only need to specify URL of Internet image by `infer_imgs`. In this case, the image file will be downloaded and saved in the directory `~/.paddleclas/images/`.

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = 'https://raw.githubusercontent.com/paddlepaddle/paddleclas/release/2.2/docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='https://raw.githubusercontent.com/paddlepaddle/paddleclas/release/2.2/docs/images/whl/demo.jpg'
```


### 4.6 Prediction of NumPy.array format image
In Python code, you can predict the NumPy.array format image, only need to use the `infer_imgs` to transfer variable of image data. Note that the image data must be 3 channels.

* python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = cv2.imread("docs/images/whl/demo.jpg")
result=clas.predict(infer_imgs)
print(next(result))
```

### 4.7 Save the prediction result(s)
You can save the prediction result(s) as pre-label, only need to use `pre_label_out_dir` to specify the directory to save.

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


### 4.8 Specify the mapping between class id and label name
You can specify the mapping between class id and label name, only need to use `class_id_map_file` to specify the mapping file. PaddleClas uses ImageNet1K's mapping by default.

The content format of mapping file shall be:

```
class_id<space>class_name<\n>
```

For example:

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
infer_imgs = 'docs/images/whl/demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/whl/demo.jpg' --class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt'
```
