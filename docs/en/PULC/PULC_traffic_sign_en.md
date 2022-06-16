# PULC Classification Model of Traffic Sign

------

## Catalogue

- [1. Introduction](#1)
- [2. Quick Start](#2)
    - [2.1 PaddlePaddle Installation](#2.1)
    - [2.2 PaddleClas Installation](#2.2)
    - [2.3 Prediction](#2.3)
- [3. Training, Evaluation and Inference](#3)
    - [3.1 Installation](#3.1)
    - [3.2 Dataset](#3.2)
      - [3.2.1 Dataset Introduction](#3.2.1)
      - [3.2.2 Getting Dataset](#3.2.2)
    - [3.3 Training](#3.3)
    - [3.4 Evaluation](#3.4)
    - [3.5 Inference](#3.5)
- [4. Model Compression](#4)
  - [4.1 SKL-UGI Knowledge Distillation](#4.1)
    - [4.1.1 Teacher Model Training](#4.1.1)
    - [4.1.2 Knowledge Distillation Training](#4.1.2)
- [5. SHAS](#5)
- [6. Inference Deployment](#6)
  - [6.1 Getting Paddle Inference Model](#6.1)
    - [6.1.1 Exporting Paddle Inference Model](#6.1.1)
    - [6.1.2 Downloading Inference Model](#6.1.2)
  - [6.2 Prediction with Python](#6.2)
    - [6.2.1 Image Prediction](#6.2.1)
    - [6.2.2 Images Prediction](#6.2.2)
  - [6.3 Deployment with C++](#6.3)
  - [6.4 Deployment as Service](#6.4)
  - [6.5 Deployment on Mobile](#6.5)
  - [6.6 Converting To ONNX and Deployment](#6.6)

<a name="1"></a>

## 1. Introduction

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of traffic sign using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in automatic driving, road monitoring, etc.

The following table lists the relevant indicators of the model. The first two lines means that using SwinTransformer_tiny and MobileNetV3_small_x0_35 as the backbone to training. The third to sixth lines means that the backbone is replaced by PPLCNet, additional use of EDA strategy and additional use of EDA strategy and SKL-UGI knowledge distillation strategy.

| Backbone | Top-1 Acc(%) | Latency(ms) | Size(M)| Training Strategy |
|-------|-----------|----------|---------------|---------------|
| SwinTranformer_tiny  | 98.11 | 89.45  | 111 | using ImageNet pretrained model |
| MobileNetV3_small_x0_35  | 93.88 | 3.01  | 3.9 | using ImageNet pretrained model |
| PPLCNet_x1_0  | 97.78 | 2.10  | 8.2 | using ImageNet pretrained model |
| PPLCNet_x1_0  | 97.84 | 2.10  | 8.2 | using SSLD pretrained model |
| PPLCNet_x1_0  | 98.14 | 2.10  | 8.2 | using SSLD pretrained model + EDA strategy  |
| <b>PPLCNet_x1_0<b>  | <b>98.35<b> | <b>2.10<b>  | <b>8.2<b> | using SSLD pretrained model + EDA strategy + SKL-UGI knowledge distillation strategy|

It can be seen that high accuracy can be getted when backbone is SwinTranformer_tiny, but the speed is slow. Replacing backbone with the lightweight model MobileNetV3_small_x0_35, the speed can be greatly improved, but the accuracy will be greatly reduced. Replacing backbone with faster backbone PPLCNet_x1_0, the accuracy is lower 3.9 percentage points than MobileNetv3_small_x0_35. At the same time, the speed can be more than 43% faster. After additional using the SSLD pretrained model, the accuracy can be improved by about 0.06 percentage points without affecting the inference speed. Further, additional using the EDA strategy, the accuracy can be increased by 0.3 percentage points. Finally, after additional using the SKL-UGI knowledge distillation, the accuracy can be further improved by 0.21 percentage points. At this point, the accuracy exceeds that of SwinTranformer_tiny, but the speed is more than 41 times faster. The training method and deployment instructions of PULC will be introduced in detail below.

**Note**:

* The Latency is tested on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz. The MKLDNN is enabled and the number of threads is 10.
* About PP-LCNet, please refer to [PP-LCNet Introduction](../models/PP-LCNet_en.md) and [PP-LCNet Paper](https://arxiv.org/abs/2109.15099).

<a name="2"></a>

## 2. Quick Start

<a name="2.1"></a>  

### 2.1 PaddlePaddle Installation

- Run the following command to install if CUDA9 or CUDA10 is available.

```bash
python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

- Run the following command to install if GPU device is unavailable.

```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

Please refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html) for more information about installation, for examples other versions.

<a name="2.2"></a>  

### 2.2 PaddleClas wheel Installation

The command of PaddleClas installation as bellow:

```bash
pip3 install paddleclas
```

<a name="2.3"></a>

### 2.3 Prediction

First, please click [here](https://paddleclas.bj.bcebos.com/data/PULC/pulc_demo_imgs.zip) to download and unzip to get the test demo images.

* Prediction with CLI

```bash
paddleclas --model_name=traffic_sign --infer_imgs=pulc_demo_imgs/traffic_sign/100999_83928.jpg
```

Results:

```
>>> result
class_ids: [182, 179, 162, 128, 24], scores: [0.98623, 0.01255, 0.00022, 0.00021, 0.00012], label_names: ['pl110', 'pl100', 'pl120', 'p26', 'pm10'], filename: pulc_demo_imgs/traffic_sign/100999_83928.jpg
Predict complete!
```

**Note**: If you want to test other images, only need to specify the `--infer_imgs` argument, and the directory containing images is also supported.

* Prediction in Python

```python
import paddleclas
model = paddleclas.PaddleClas(model_name="traffic_sign")
result = model.predict(input_data="pulc_demo_imgs/traffic_sign/100999_83928.jpg")
print(next(result))
```

**Note**: The `result` returned by `model.predict()` is a generator, so you need to use the `next()` function to call it or `for` loop to loop it. And it will predict with `batch_size` size batch and return the prediction results when called. The default `batch_size` is 1, and you also specify the `batch_size` when instantiating, such as `model = paddleclas.PaddleClas(model_name="traffic_sign",  batch_size=2)`. The result of demo above:

```
>>> result
[{'class_ids': [182, 179, 162, 128, 24], 'scores': [0.98623, 0.01255, 0.00022, 0.00021, 0.00012], 'label_names': ['pl110', 'pl100', 'pl120', 'p26', 'pm10'], 'filename': 'pulc_demo_imgs/traffic_sign/100999_83928.jpg'}]
```

<a name="3"></a>

## 3. Training, Evaluation and Inference

<a name="3.1"></a>  

### 3.1 Installation

Please refer to [Installation](../installation/install_paddleclas_en.md) to get the description about installation.

<a name="3.2"></a>

### 3.2 Dataset

<a name="3.2.1"></a>

#### 3.2.1 Dataset Introduction

All datasets used in this case are open source data. Train data is the subset of [MS-COCO](https://cocodataset.org/#overview) training data. And the validation data is the subset of [Object365](https://www.objects365.org/overview.html) training data. ImageNet_val is [ImageNet-1k](https://www.image-net.org/) validation data.

The dataset used in this case is based on the [Tsinghua-Tencent 100K dataset (CC-BY-NC license), TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/) randomly expanded and cropped according to the bounding box.

<a name="3.2.2"></a>  

#### 3.2.2 Getting Dataset

The processing to `TT00K` includes randomly expansion and cropping, details are shown below.

```python
def get_random_crop_box(xmin, ymin, xmax, ymax, img_height, img_width, ratio=1.0):
    h = ymax - ymin
    w = ymax - ymin

    xmin_diff = random.random() * ratio * min(w, xmin/ratio)
    ymin_diff = random.random() * ratio * min(h, ymin/ratio)
    xmax_diff = random.random() * ratio * min(w, (img_width-xmin-1)/ratio)
    ymax_diff = random.random() * ratio * min(h, (img_height-ymin-1)/ratio)

    new_xmin = round(xmin - xmin_diff)
    new_ymin = round(ymin - ymin_diff)
    new_xmax = round(xmax + xmax_diff)
    new_ymax = round(ymax + ymax_diff)

    return new_xmin, new_ymin, new_xmax, new_ymax
```

Some image of the processed dataset is as follows:

<div align="center">
<img src="../../images/PULC/docs/traffic_sign_data_demo.png"  width = "500" />
</div>

You can also download the data processed directly. And the process script file `deal.py` is also included.

```
cd path_to_PaddleClas
```

Enter the `dataset/` directory, download and unzip the dataset.

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/traffic_sign.tar
tar -xf traffic_sign.tar
cd ../
```

The datas under `traffic_sign` directory:

```
traffic_sign
├── train
│   ├── 0_62627.jpg
│   ├── 100000_89031.jpg
│   ├── 100001_89031.jpg
...
├── test
│   ├── 100423_2315.jpg
│   ├── 100424_2315.jpg
│   ├── 100425_2315.jpg
...
├── other
│   ├── 100603_3422.jpg
│   ├── 100604_3422.jpg
...
├── label_list_train.txt
├── label_list_test.txt
├── label_list_other.txt
├── label_list_train_for_distillation.txt
├── label_list_train.txt.debug
├── label_list_test.txt.debug
├── label_name_id.txt
├── deal.py
```

Where `train/` and `test/` are training set and validation set respectively. The `label_list_train.txt` and `label_list_test.txt` are label files of training data and validation data respectively. The file `label_list_train.txt.debug` and `label_list_test.txt.debug` are subset of `train_list.txt` and `val_list.txt` respectively. `other` would be used for SKL-UGI knowledge distillation, and its label file is `label_list_train_for_distillation.txt`.

**Note**:

* About the contents format of `label_list_train.txt` and `label_list_train.txt`, please refer to [Description about Classification Dataset in PaddleClas](../data_preparation/classification_dataset_en.md).
* About the `label_list_train_for_distillation.txt`, please refer to [Knowledge Distillation Label](../advanced_tutorials/distillation/distillation_en.md).

<a name="3.3"></a>

### 3.3 Training

The details of training config in `ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml`. The command about training as follows:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml
```

The best metric of validation data is between `98.0` and `98.2`. There would be fluctuations because the data size is small.

<a name="3.4"></a>

### 3.4 Evaluation

After training, you can use the following commands to evaluate the model.

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

Among the above command, the argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.

<a name="3.5"></a>

### 3.5 Inference

After training, you can use the model that trained to infer. Command is as follow:

```bash
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model
```

The results:

```
99603_17806.jpg:        class id(s): [216, 145, 49, 207, 169], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['pm20', 'pm30', 'pm40', 'pl25', 'pm15']
```

**Note**:

* Among the above command, argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.
* The default test image is `deploy/images/PULC/traffic_sign/99603_17806.jpg`. And you can test other image, only need to specify the argument `-o Infer.infer_imgs=path_to_test_image`.


<a name="4"></a>

## 4. Model Compression

<a name="4.1"></a>

### 4.1 SKL-UGI Knowledge Distillation

SKL-UGI is a simple but effective knowledge distillation algrithem proposed by PaddleClas.

<!-- todo -->
<!-- Please refer to [SKL-UGI](../advanced_tutorials/distillation/distillation_en.md) for more details. -->

<a name="4.1.1"></a>

#### 4.1.1 Teacher Model Training

Training the teacher model with hyperparameters specified in `ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml`. The command is as follow:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

The best metric of validation data is about `98.59%`. The best teacher model weight would be saved in file `output/ResNet101_vd/best_model.pdparams`.

<a name="4.1.2"></a>

#### 4.1.2 Knowledge Distillation Training

The training strategy, specified in training config file `ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0_distillation.yaml`, the teacher model is `ResNet101_vd`, the student model is `PPLCNet_x1_0` and the additional unlabeled training data is validation data of ImageNet1k. The command is as follow:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

The best metric is about `98.35%`. The best student model weight would be saved in file `output/DistillationModel/best_model_student.pdparams`.

<a name="5"></a>

## 5. Hyperparameters Searching

The hyperparameters used by [3.2 section](#3.2) and [4.1 section](#4.1) are according by `Hyperparameters Searching` in PaddleClas. If you want to get better results on your own dataset, you can refer to [Hyperparameters Searching](PULC_train_en.md#4) to get better hyperparameters.

**Note**: This section is optional. Because the search process will take a long time, you can selectively run according to your specific. If not replace the dataset, you can ignore this section.

<a name="6"></a>

## 6. Inference Deployment

<a name="6.1"></a>

### 6.1 Getting Paddle Inference Model

Paddle Inference is the original Inference Library of the PaddlePaddle, provides high-performance inference for server deployment. And compared with  directly based on the pretrained model, Paddle Inference can use tools to accelerate prediction, so as to achieve better inference performance. Please refer to [Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html) for more information.

Paddle Inference need Paddle Inference Model to predict. Two process provided to get Paddle Inference Model. If want to use the provided by PaddleClas, you can download directly, click [Downloading Inference Model](#6.1.2).
<a name="6.1.1"></a>

### 6.1.1 Exporting Paddle Inference Model

The command about exporting Paddle Inference Model is as follow:

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/PULC/traffic_sign/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model_student \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_traffic_sign_infer
```

After running above command, the inference model files would be saved in `deploy/models/PPLCNet_x1_0_traffic_sign_infer`, as shown below:

```
├── PPLCNet_x1_0_traffic_sign_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

**Note**: The best model is from knowledge distillation training. If knowledge distillation training is not used, the best model would be saved in `output/PPLCNet_x1_0/best_model.pdparams`.

<a name="6.1.2"></a>

### 6.1.2 Downloading Inference Model

You can also download directly.

```
cd deploy/models
# download the inference model and decompression
wget https://paddleclas.bj.bcebos.com/models/PULC/traffic_sign_infer.tar && tar -xf traffic_sign_infer.tar
```

After decompression, the directory `models` should be shown below.

```
├── traffic_sign_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="6.2"></a>

### 6.2 Prediction with Python

<a name="6.2.1"></a>  

#### 6.2.1 Image Prediction

Return the directory `deploy`:

```
cd ../
```

Run the following command to classify traffic sign about the image `./images/PULC/traffic_sign/99603_17806.jpg`.

```shell
# Use the following command to predict with GPU.
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign.yaml
# Use the following command to predict with CPU.
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign.yaml -o Global.use_gpu=False
```

The prediction results:

```
99603_17806.jpg:        class id(s): [216, 145, 49, 207, 169], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['pm20', 'pm30', 'pm40', 'pl25', 'pm15']
```

<a name="6.2.2"></a>  

#### 6.2.2 Images Prediction

If you want to predict images in directory, please specify the argument `Global.infer_imgs` as directory path by `-o Global.infer_imgs`. The command is as follow.

```shell
# Use the following command to predict with GPU. If want to replace with CPU, you can add argument -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/traffic_sign/inference_traffic_sign.yaml -o Global.infer_imgs="./images/PULC/traffic_sign/"
```

All prediction results will be printed, as shown below.

```
100999_83928.jpg:    class id(s): [182, 179, 162, 128, 24], score(s): [0.99, 0.01, 0.00, 0.00, 0.00], label_name(s): ['pl110', 'pl100', 'pl120', 'p26', 'pm10']
99603_17806.jpg:    class id(s): [216, 145, 49, 24, 169], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['pm20', 'pm30', 'pm40', 'pm10', 'pm15']
```

About the `label_name` details, please refer to `dataset/traffic_sign/report.pdf`.

<a name="6.3"></a>

### 6.3 Deployment with C++

PaddleClas provides an example about how to deploy with C++. Please refer to [Deployment with C++](../inference_deployment/cpp_deploy_en.md).

<a name="6.4"></a>

### 6.4 Deployment as Service

Paddle Serving is a flexible, high-performance carrier for machine learning models, and supports different protocol, such as RESTful, gRPC, bRPC and so on, which provides different deployment solutions for a variety of heterogeneous hardware and operating system environments. Please refer [Paddle Serving](https://github.com/PaddlePaddle/Serving) for more information.

PaddleClas provides an example about how to deploy as service by Paddle Serving. Please refer to [Paddle Serving Deployment](../inference_deployment/paddle_serving_deploy_en.md).

<a name="6.5"></a>

### 6.5 Deployment on Mobile

Paddle-Lite is an open source deep learning framework that designed to make easy to perform inference on mobile, embeded, and IoT devices. Please refer to [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) for more information.

PaddleClas provides an example of how to deploy on mobile by Paddle-Lite. Please refer to [Paddle-Lite deployment](../inference_deployment/paddle_lite_deploy_en.md).

<a name="6.6"></a>

### 6.6 Converting To ONNX and Deployment

Paddle2ONNX support convert Paddle Inference model to ONNX model. And you can deploy with ONNX model on different inference engine, such as TensorRT, OpenVINO, MNN/TNN, NCNN and so on. About Paddle2ONNX details, please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).

PaddleClas provides an example of how to convert Paddle Inference model to ONNX model by paddle2onnx toolkit and predict by ONNX model. You can refer to [paddle2onnx](../../../deploy/paddle2onnx/readme_en.md) for deployment details.
