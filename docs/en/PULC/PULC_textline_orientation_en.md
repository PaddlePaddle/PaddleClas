# PULC Classification Model of Textline Orientation

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

This case provides a way for users to quickly build a lightweight, high-precision and practical classification model of textline orientation using PaddleClas PULC (Practical Ultra Lightweight image Classification). The model can be widely used in character correction, character recognition, etc.

The following table lists the relevant indicators of the model. The first two lines means that using SwinTransformer_tiny and MobileNetV3_small_x0_35 as the backbone to training. The third to seventh lines means that the backbone is replaced by PPLCNet, additional use of EDA strategy and additional use of EDA strategy and SKL-UGI knowledge distillation strategy.

| Backbone | Top-1 Acc(%) | Latency(ms) | Size(M)| Training Strategy |
|-------|-----------|----------|---------------|---------------|
| SwinTranformer_tiny  | 93.61 | 89.64  | 111 |  using ImageNet pretrained model |
| MobileNetV3_small_x0_35  | 81.40 | 2.96  | 2.6 |  using ImageNet pretrained model |
| PPLCNet_x1_0  | 89.99 | 2.11  | 7.0 |  using ImageNet pretrained model |
| PPLCNet_x1_0*  | 94.06 | 2.68  | 7.0 | using ImageNet pretrained model |
| PPLCNet_x1_0*  | 94.11 | 2.68  | 7.0 | using SSLD pretrained model |
| <b>PPLCNet_x1_0**<b>  | <b>96.01<b> | <b>2.72<b>  | <b>7.0<b> | using SSLD pretrained model + EDA strategy  |
| PPLCNet_x1_0**  | 95.86 | 2.72  | 7.0 | using SSLD pretrained model + EDA strategy + SKL-UGI knowledge distillation strategy|

It can be seen that high accuracy can be getted when backbone is SwinTranformer_tiny, but the speed is slow. Replacing backbone with the lightweight model MobileNetV3_small_x0_35, the speed can be greatly improved, but the accuracy will be greatly reduced. Replacing backbone with faster backbone PPLCNet_x1_0, the accuracy is higher more 8.6 percentage points than MobileNetv3_small_x0_35. At the same time, the speed can be more than 10% faster. On this basis, by changing the resolution and stripe (refer to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)), the speed becomes 27% slower, but the accuracy can be improved by 4.5 percentage points. After additional using the SSLD pretrained model, the accuracy can be improved by about 0.05 percentage points without affecting the inference speed. Finally, additional using the EDA strategy, the accuracy can be increased by 1.9 percentage points. The training method and deployment instructions of PULC will be introduced in detail below.

**Note**:
* Backbone name without \* means the resolution is 224x224, and with \* means the resolution is 48x192 (h\*w). The stride of the network is changed to `[2, [2, 1], [2, 1], [2, 1]`. Please refer to [PaddleOCR]（ https://github.com/PaddlePaddle/PaddleOCR）for more details.
* Backbone name with \*\* means that the resolution is 80x160 (h\*w), and the stride of the network is changed to `[2, [2, 1], [2, 1], [2, 1]]`. This resolution is searched by [Hyperparameter Searching](pulc_train_en.md#4).
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
paddleclas --model_name=textline_orientation --infer_imgs=pulc_demo_imgs/textline_orientation/textline_orientation_test_0_0.png
```

Results:

```
>>> result
class_ids: [0], scores: [1.0], label_names: ['0_degree'], filename: pulc_demo_imgs/textline_orientation/textline_orientation_test_0_0.png
Predict complete!
```

**Note**: If you want to test other images, only need to specify the `--infer_imgs` argument, and the directory containing images is also supported.

* Prediction in Python

```python
import paddleclas
model = paddleclas.PaddleClas(model_name="textline_orientation")
result = model.predict(input_data="pulc_demo_imgs/textline_orientation/textline_orientation_test_0_0.png")
print(next(result))
```

**Note**: The `result` returned by `model.predict()` is a generator, so you need to use the `next()` function to call it or `for` loop to loop it. And it will predict with `batch_size` size batch and return the prediction results when called. The default `batch_size` is 1, and you also specify the `batch_size` when instantiating, such as `model = paddleclas.PaddleClas(model_name="textline_orientation",  batch_size=2)`. The result of demo above:

```
>>> result
[{'class_ids': [0], 'scores': [1.0], 'label_names': ['0_degree'], 'filename': 'pulc_demo_imgs/textline_orientation/textline_orientation_test_0_0.png'}]
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

The data used in this case come from internal data. If you want to experience the training process, you can use open source data, such as [ICDAR2019-LSVT](https://aistudio.baidu.com/aistudio/datasetdetail/8429).

<a name="3.2.2"></a>  

#### 3.2.2 Getting Dataset

Take ICDAR2019-LSVT for example, images with ID numbers from 0 to 1999 would be processed and used. After rotation, it is divided into class 0 or class 1. Class 0 means that the textline rotation angle is 0 degrees, and class 1 means 180 degrees.

- Training data: The images with ID number from 0 to 1799 are used as the training set. 3600 images in total.
- Evaluation data: The images with ID number from 1800 to 1999 are used as the evaluation set. 400 images in total.

Some image of the processed dataset is as follows:

![](../../images/PULC/docs/textline_orientation_data_demo.png)

And you can also download the data processed directly.

```
cd path_to_PaddleClas
```

Enter the `dataset/` directory, download and unzip the dataset.

```shell
cd dataset
wget https://paddleclas.bj.bcebos.com/data/PULC/textline_orientation.tar
tar -xf textline_orientation.tar
cd ../
```

The datas under `textline_orientation` directory:

```
├── 0
│   ├── img_0.jpg
│   ├── img_1.jpg
...
├── 1
│   ├── img_0.jpg
│   ├── img_1.jpg
...
├── train_list.txt
└── val_list.txt
```

其中 `0/` 和 `1/` 分别存放 0 类和 1 类的数据。`train_list.txt` 和 `val_list.txt` 分别为训练集和验证集的标签文件。

Where `0/` and `1/` are class 0 and class 1 data respectively. The `train_list.txt` and `val_list.txt` are label files of training data and validation data respectively.

**Note**:

* About the contents format of `train_list.txt` and `val_list.txt`, please refer to [Description about Classification Dataset in PaddleClas](../data_preparation/classification_dataset_en.md).

<a name="3.3"></a>

### 3.3 Training

The details of training config in `ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml`. The command about training as follows:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml
```

**Note**:

* Because the ICDAR2019-LSVT data set is different from the dataset used in the provided pretrained model. If you want to get higher accuracy, you can process [ICDAR2019-LSVT](https://aistudio.baidu.com/aistudio/datasetdetail/8429).

<a name="3.4"></a>

### 3.4 Evaluation

After training, you can use the following commands to evaluate the model.

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

Among the above command, the argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.

<a name="3.5"></a>

### 3.5 Inference

After training, you can use the model that trained to infer. Command is as follow:

```python
python3 tools/infer.py \
    -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

The results:

```
[{'class_ids': [0], 'scores': [1.0], 'file_name': 'deploy/images/PULC/textline_orientation/textline_orientation_test_0_0.png', 'label_names': ['0_degree']}]
```

**Note**:

* Among the above command, argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.
* The default test image is `deploy/images/PULC/textline_orientation/textline_orientation_test_0_0.png`. And you can test other image, only need to specify the argument `-o Infer.infer_imgs=path_to_test_image`.

<a name="4"></a>

## 4. Model Compression

<a name="4.1"></a>

### 4.1 SKL-UGI Knowledge Distillation

SKL-UGI is a simple but effective knowledge distillation algrithem proposed by PaddleClas.

<!-- todo -->
<!-- Please refer to [SKL-UGI](../advanced_tutorials/distillation/distillation_en.md) for more details. -->

<a name="4.1.1"></a>

#### 4.1.1 Teacher Model Training

Training the teacher model with hyperparameters specified in `ppcls/configs/PULC/textline_orientation/PPLCNet/PPLCNet_x1_0.yaml`. The command is as follow:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml \
        -o Arch.name=ResNet101_vd
```

The best metric of validation data is between `0.96` and `0.98`. The best teacher model weight would be saved in file `output/ResNet101_vd/best_model.pdparams`.

<a name="4.1.2"></a>

#### 4.1.2 Knowledge Distillation Training

The training strategy, specified in training config file `ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0_distillation.yaml`, the teacher model is `ResNet101_vd` and the student model is `PPLCNet_x1_0`. The command is as follow:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0_distillation.yaml \
        -o Arch.models.0.Teacher.pretrained=output/ResNet101_vd/best_model
```

The best metric is between `0.95` and `0.97`. The best student model weight would be saved in file `output/DistillationModel/best_model_student.pdparams`.

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
    -c ./ppcls/configs/PULC/textline_orientation/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_textline_orientation_infer
```

After running above command, the inference model files would be saved in `deploy/models/PPLCNet_x1_0_textline_orientation_infer`, as shown below:

```
├── PPLCNet_x1_0_textline_orientation_infer
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
# 下载 inference 模型并解压
wget https://paddleclas.bj.bcebos.com/models/PULC/textline_orientation_infer.tar && tar -xf textline_orientation_infer.tar
```

After decompression, the directory `models` should be shown below.

```
├── textline_orientation_infer
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

Run the following command to classify the rotation of image `./images/PULC/textline_orientation/objects365_02035329.jpg`.

```shell
# Use the following command to predict with GPU.
python3.7 python/predict_cls.py -c configs/PULC/textline_orientation/inference_textline_orientation.yaml
# Use the following command to predict with CPU.
python3.7 python/predict_cls.py -c configs/PULC/textline_orientation/inference_textline_orientation.yaml  -o Global.use_gpu=False
```

The prediction results:

```
textline_orientation_test_0_0.png:    class id(s): [0], score(s): [1.00], label_name(s): ['0_degree']
```

<a name="6.2.2"></a>  

#### 6.2.2 Images Prediction

If you want to predict images in directory, please specify the argument `Global.infer_imgs` as directory path by `-o Global.infer_imgs`. The command is as follow.

```shell
# Use the following command to predict with GPU. If want to replace with CPU, you can add argument -o Global.use_gpu=False
python3.7 python/predict_cls.py -c configs/PULC/textline_orientation/inference_textline_orientation.yaml -o Global.infer_imgs="./images/PULC/textline_orientation/"
```

All prediction results will be printed, as shown below.

```
textline_orientation_test_0_0.png:    class id(s): [0], score(s): [1.00], label_name(s): ['0_degree']
textline_orientation_test_0_1.png:    class id(s): [0], score(s): [1.00], label_name(s): ['0_degree']
textline_orientation_test_1_0.png:    class id(s): [1], score(s): [1.00], label_name(s): ['180_degree']
textline_orientation_test_1_1.png:    class id(s): [1], score(s): [1.00], label_name(s): ['180_degree']
```

Among the prediction results above, `0_degree` means that the rotation angle of the textline image is 0, and `180_degree` means that 180.

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
