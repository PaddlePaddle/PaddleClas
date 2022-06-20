# PULC Quick Start

------

This document introduces the prediction using PULC series model based on PaddleClas wheel.

## Catalogue

- [1. Installation](#1)
  - [1.1 PaddlePaddle Installation](#11)
  - [1.2 PaddleClas wheel Installation](#12)
- [2. Quick Start](#2)
  - [2.1 Predicion with Command Line](#2.1)
  - [2.2 Predicion with Python](#2.2)
  - [2.3 Supported Model List](#2.3)
- [3. Summary](#3)

<a name="1"></a>

## 1. Installation

<a name="1.1"></a>

### 1.1 PaddlePaddle Installation

- Run the following command to install if CUDA9 or CUDA10 is available.

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- Run the following command to install if GPU device is unavailable.

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

Please refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) for more information about installation, for examples other versions.

<a name="1.2"></a>

### 1.2 PaddleClas wheel Installation

```bash
pip3 install paddleclas
```

<a name="2"></a>

## 2. Quick Start

PaddleClas provides a series of test cases, which contain demos of different scenes about people, cars, OCR, etc. Click [here](https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip) to download the data.

<a name="2.1"></a>

### 2.1 Predicion with Command Line

```
cd /path/to/pulc_demo_imgs
```

The prediction command:

```bash
paddleclas --model_name=person_exists --infer_imgs=pulc_demo_imgs/person_exists/objects365_01780782.jpg
```

Result:

```
>>> result
class_ids: [0], scores: [0.9955421453341842], label_names: ['nobody'], filename: pulc_demo_imgs/person_exists/objects365_01780782.jpg
Predict complete!
```
`Nobody` means there is no one in the image, `someone` means there is someone in the image. Therefore, the prediction result indicates that there is no one in the figure.

**Note**: The "--infer_imgs" argument specify the image(s) to be predict, and you can also specify a directoy contains images. If use other model, you can specify the `--model_name` argument. Please refer to [2.3 Supported Model List](#2.3) for the supported model list.

<a name="2.2"></a>

### 2.2 Predicion with Python

You can also use in Python:

```python
import paddleclas
model = paddleclas.PaddleClas(model_name="person_exists")
result = model.predict(input_data="pulc_demo_imgs/person_exists/objects365_01780782.jpg")
print(next(result))
```

The printed result information:

```
>>> result
[{'class_ids': [0], 'scores': [0.9955421453341842], 'label_names': ['nobody'], 'filename': 'pulc_demo_imgs/person_exists/objects365_01780782.jpg'}]
```

**Note**: `model.predict()` is a generator, so `next()` or `for` is needed to call it. This would to predict by batch that length is `batch_size`, default by 1. You can specify the argument `batch_size` and `model_name` when instantiating PaddleClas object, for example: `model = paddleclas.PaddleClas(model_name="person_exists",  batch_size=2)`. Please refer to [2.3 Supported Model List](#2.3) for the supported model list.

<a name="2.3"></a>

### 2.3 Supported Model List

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

<a name="3"></a>

## 3. Summary

The PULC series models have been verified to be effective in different scenarios about people, vehicles, OCR, etc. The ultra lightweight model can achieve the accuracy close to SwinTransformer model, and the speed is increased by 40+ times. And PULC also provides the whole process of dataset getting, model training, model compression and deployment. Please refer to [Human Exists Classification](PULC_person_exists_en.md)、[Pedestrian Attribute Classification](PULC_person_attribute_en.md)、[Classification of Wheather Wearing Safety Helmet](PULC_safety_helmet_en.md)、[Traffic Sign Classification](PULC_traffic_sign_en.md)、[Vehicle Attribute Classification](PULC_vehicle_attribute_en.md)、[Car Exists Classification](PULC_car_exists_en.md)、[Text Image Orientation Classification](PULC_text_image_orientation_en.md)、[Text-line Orientation Classification](PULC_textline_orientation_en.md)、[Language Classification](PULC_language_classification_en.md) for more information about different scenarios.
