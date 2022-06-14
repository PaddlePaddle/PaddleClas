# PaddleClas wheel package

PaddleClas supports Python wheel package for prediction. At present, PaddleClas wheel supports image classification including ImagetNet1k models and PULC models, but does not support mainbody detection, feature extraction and vector retrieval.

---

## Catalogue

- [1. Installation](#1)
- [2. Quick Start](#2)
   - [2.1 ImageNet1k models](#2.1)
   - [2.2 PULC models](#2.2)
- [3. Definition of Parameters](#3)
- [4. More usage](#4)
   - [4.1 View help information](#4.1)
   - [4.2 Prediction using inference model provide by PaddleClas](#4.2)
   - [4.3 Prediction using local model files](#4.3)
   - [4.4 Prediction by batch](#4.4)
   - [4.5 Prediction of Internet image](#4.5)
   - [4.6 Prediction of `NumPy.array` format image](#4.6)
   - [4.7 Save the prediction result(s)](#4.7)
   - [4.8 Specify the mapping between class id and label name](#4.8)

<a name="1"></a>

## 1. Installation

* installing from pypi

```bash
pip3 install paddleclas==2.2.1
```

* build own whl package and install

```bash
python3 setup.py bdist_wheel
pip3 install dist/*
```

<a name="2"></a>

## 2. Quick Start

<a name="2.1"></a>

### 2.1 ImageNet1k models

Using the `ResNet50` model provided by PaddleClas, the following image(`'docs/images/inference_deployment/whl_demo.jpg'`) as an example.

![](../../images/inference_deployment/whl_demo.jpg)

* Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs='docs/images/inference_deployment/whl_demo.jpg'
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
paddleclas --model_name=ResNet50  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```

```
>>> result
filename: docs/images/inference_deployment/whl_demo.jpg, top-5, class_ids: [8, 7, 136, 80, 84], scores: [0.79368, 0.16329, 0.01853, 0.00959, 0.00239], label_names: ['hen', 'cock', 'European gallinule, Porphyrio porphyrio', 'black grouse', 'peacock']
Predict complete!
```

<a name="2.2"></a>

### 2.2 PULC models

PULC integrates various state-of-the-art algorithms such as backbone network, data augmentation and distillation, etc., and finally can automatically obtain a lightweight and high-precision image classification model.

PaddleClas provides a series of test cases, which contain demos of different scenes about people, cars, OCR, etc. Click [here](https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip) to download the data.

Prection using the PULC "Human Exists Classification" model provided by PaddleClas:

* Python

```python
import paddleclas
model = paddleclas.PaddleClas(model_name="person_exists")
result = model.predict(input_data="pulc_demo_imgs/person_exists/objects365_01780782.jpg")
print(next(result))
```

```
>>> result
[{'class_ids': [0], 'scores': [0.9955421453341842], 'label_names': ['nobody'], 'filename': 'pulc_demo_imgs/person_exists/objects365_01780782.jpg'}]
```

`Nobody` means there is no one in the image, `someone` means there is someone in the image. Therefore, the prediction result indicates that there is no one in the figure.

**Note**: `model.predict()` is a generator, so `next()` or `for` is needed to call it. This would to predict by batch that length is `batch_size`, default by 1. You can specify the argument `batch_size` and `model_name` when instantiating PaddleClas object, for example: `model = paddleclas.PaddleClas(model_name="person_exists",  batch_size=2)`. Please refer to [Supported Model List](#PULC_Models) for the supported model list.

* CLI

```bash
paddleclas --model_name=person_exists --infer_imgs=pulc_demo_imgs/person_exists/objects365_01780782.jpg
```

```
>>> result
class_ids: [0], scores: [0.9955421453341842], label_names: ['nobody'], filename: pulc_demo_imgs/person_exists/objects365_01780782.jpg
Predict complete!
```

**Note**: The "--infer_imgs" argument specify the image(s) to be predict, and you can also specify a directoy contains images. If use other model, you can specify the `--model_name` argument. Please refer to [Supported Model List](#PULC_Models) for the supported model list.

<a name="PULC_Models"></a>

**Supported Model List**

The name of PULC series models are as follows:

| Name | Intro |
| --- | --- |
| person_exists | Human Exists Classification |
| person_attribute | Pedestrian Attribute Classification |
| safety_helmet | Classification of Wheather Wearing Safety Helmet |
| traffic_sign | Traffic Sign Classification |
| vehicle_attribute | Vehicle Attribute Classification |
| car_exists | Car Exists Classification |
| text_image_orientation | Text Image Orientation Classification |
| textline_orientation | Text-line Orientation Classification |
| language_classification | Language Classification |

Please refer to [Human Exists Classification](../PULC/PULC_person_exists_en.md)、[Pedestrian Attribute Classification](../PULC/PULC_person_attribute_en.md)、[Classification of Wheather Wearing Safety Helmet](../PULC/PULC_safety_helmet_en.md)、[Traffic Sign Classification](../PULC/PULC_traffic_sign_en.md)、[Vehicle Attribute Classification](../PULC/PULC_vehicle_attribute_en.md)、[Car Exists Classification](../PULC/PULC_car_exists_en.md)、[Text Image Orientation Classification](../PULC/PULC_text_image_orientation_en.md)、[Text-line Orientation Classification](../PULC/PULC_textline_orientation_en.md)、[Language Classification](../PULC/PULC_language_classification_en.md) for more information about different scenarios.

<a name="3"></a>

## 3. Definition of Parameters

The following parameters can be specified in Command Line or used as parameters of the constructor when instantiating the PaddleClas object in Python.
* model_name(str): If using inference model based on ImageNet1k provided by Paddle, please specify the model's name by the parameter.
* inference_model_dir(str): Local model files directory, which is valid when `model_name` is not specified. The directory should contain `inference.pdmodel` and `inference.pdiparams`.
* infer_imgs(str): The path of image to be predicted, or the directory containing the image files, or the URL of the image from Internet.
* use_gpu(bool): Whether to use GPU or not.
* gpu_mem(int): GPU memory usages.
* use_tensorrt(bool): Whether to open TensorRT or not. Using it can greatly promote predict preformance.
* enable_mkldnn(bool): Whether enable MKLDNN or not.
* cpu_num_threads(int): Assign number of cpu threads, valid when `--use_gpu` is `False` and `--enable_mkldnn` is `True`.
* batch_size(int): Batch size.
* resize_short(int): Resize the minima between height and width into `resize_short`.
* crop_size(int): Center crop image to `crop_size`.
* topk(int): Print (return) the `topk` prediction results when Topk postprocess is used.
* threshold(float): The threshold of ThreshOutput when postprocess is used.
* class_id_map_file(str): The mapping file between class ID and label.
* save_dir(str): The directory to save the prediction results that can be used as pre-label.

**Note**: If you want to use `Transformer series models`, such as `DeiT_***_384`, `ViT_***_384`, etc., please pay attention to the input size of model, and need to set `resize_short=384`, `resize=384`. The following is a demo.

* CLI:
```bash
from paddleclas import PaddleClas, get_default_confg
paddleclas --model_name=ViT_base_patch16_384 --infer_imgs='docs/images/inference_deployment/whl_demo.jpg' --resize_short=384 --crop_size=384
```

* Python:
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ViT_base_patch16_384', resize_short=384, crop_size=384)
```

<a name="4"></a>

## 4. Usage

PaddleClas provides two ways to use:
1. Python interative programming;
2. Bash command line programming.

<a name="4.1"></a>

### 4.1 View help information

* CLI
```bash
paddleclas -h
```

<a name="4.2"></a>

### 4.2 Prediction using inference model provide by PaddleClas
You can use the inference model provided by PaddleClas to predict, and only need to specify `model_name`. In this case, PaddleClas will automatically download files of specified model and save them in the directory `~/.paddleclas/`.

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

### 4.3 Prediction using local model files
You can use the local model files trained by yourself to predict, and only need to specify `inference_model_dir`. Note that the directory must contain `inference.pdmodel` and `inference.pdiparams`.

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

<a name="4.5"></a>

### 4.5 Prediction of Internet image
You can predict the Internet image, only need to specify URL of Internet image by `infer_imgs`. In this case, the image file will be downloaded and saved in the directory `~/.paddleclas/images/`.

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

### 4.6 Prediction of NumPy.array format image
In Python code, you can predict the `NumPy.array` format image, only need to use the `infer_imgs` to transfer variable of image data. Note that the models in PaddleClas only support to predict 3 channels image data, and channels order is `RGB`.

* python
```python
import cv2
from paddleclas import PaddleClas
clas = PaddleClas(model_name='ResNet50')
infer_imgs = cv2.imread("docs/en/inference_deployment/whl_deploy_en.md")[:, :, ::-1]
result=clas.predict(infer_imgs)
print(next(result))
```

<a name="4.7"></a>

### 4.7 Save the prediction result(s)
You can save the prediction result(s) as pre-label, only need to use `pre_label_out_dir` to specify the directory to save.

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
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result=clas.predict(infer_imgs)
print(next(result))
```

* CLI
```bash
paddleclas --model_name='ResNet50' --infer_imgs='docs/images/inference_deployment/whl_demo.jpg' --class_id_map_file='./ppcls/utils/imagenet1k_label_list.txt'
```
