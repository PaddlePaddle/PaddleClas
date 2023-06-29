# PP-LCNet Series

---

- [1. Introduction](#1)
    - [1.1 Model Introduction](#1.1)
    - [1.2 Model Details](#1.2)
    - [1.3 Result](#1.3)
- [2. Quick Start](#2)
    - [2.1 PaddlePaddle Installation](#2.1)
    - [2.2 PaddleClas Installation](#2.2)
    - [2.3 Prediction](#2.3)
- [3. Training, Evaluation and Inference](#3)
    - [3.1 Installation](#3.1)
    - [3.2 Dataset](#3.2)
    - [3.3 Training](#3.3)
      - [3.3.1 Train ImageNet](#3.3.1)
      - [3.3.2 Fine-tuning based on ImageNet weights](#3.3.2)
    - [3.4 Evaluation](#3.4)
    - [3.5 Inference](#3.5)
- [4. Inference Deployment](#4)
  - [4.1 Getting Paddle Inference Model](#4.1)
    - [4.1.1 Exporting Paddle Inference Model](#4.1.1)
    - [4.1.2 Downloading Inference Model](#4.1.2)
  - [4.2 Prediction with Python](#4.2)
    - [4.2.1 Image Prediction](#4.2.1)
    - [4.2.2 Images Prediction](#4.2.2)
  - [4.3 Deployment with C++](#4.3)
  - [4.4 Deployment as Service](#4.4)
  - [4.5 Deployment on Mobile](#4.5)
  - [4.6 Converting To ONNX and Deployment](#4.6)
- [4. Reference](#5)

<a name='1'></a>

## 1. Introduction

Recent years witnessed the emergence of many lightweight backbone networks. In past two years, in particular, there were abundant networks searched by NAS that either enjoy advantages on FLOPs or Params, or have an edge in terms of inference speed on ARM devices. However, few of them dedicated to specified optimization of Intel CPU, resulting their imperfect inference speed on the intel CPU side. Based on this, we specially design the backbone network PP-LCNet for Intel CPU devices with its acceleration library MKLDNN. Compared with other lightweight SOTA models, this backbone network can further improve the performance of the model without increasing the inference time, significantly outperforming the existing SOTA models.

<a name='1.2'></a>

### 1.2 Model Details

Build on extensive experiments, we found that many seemingly less time-consuming operations will increase the latency on Intel CPU-based devices, especially when the MKLDNN acceleration library is enabled. Finally, we summarized some strategies that can improve the accuracy of the model without increasing the latency and combined these four strategies to form PP-LCNet.

The overall structure of the network is shown in the figure below.
![](../../images/PP-LCNet/PP-LCNet.png)

<a name='1.3'></a>

### 1.3 Result

<a name="1.3.1"></a>
### 1.3.1 Image Classification

For image classification, ImageNet dataset is adopted. Compared with the current mainstream lightweight network, PP-LCNet can obtain faster inference speed with the same accuracy. When using Baidu’s self-developed SSLD distillation strategy, the accuracy is further improved, with the Top-1 Acc of ImageNet exceeding 80% at an inference speed of about 5ms on the Intel CPU side.

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|-------|-----------|----------|---------------|---------------|-------------|
| PPLCNet_x0_25  | 1.5 | 18  | 51.86 | 75.65 | 1.74 |
| PPLCNet_x0_35  | 1.6 | 29  | 58.09 | 80.83 | 1.92 |
| PPLCNet_x0_5   | 1.9 | 47  | 63.14 | 84.66 | 2.05 |
| PPLCNet_x0_75  | 2.4 | 99  | 68.18 | 88.30 | 2.29 |
| PPLCNet_x1_0     | 3.0 | 161 | 71.32 | 90.03 | 2.46 |
| PPLCNet_x1_5   | 4.5 | 342 | 73.71 | 91.53 | 3.19 |
| PPLCNet_x2_0     | 6.5 | 590 | 75.18 | 92.27 | 4.27 |
| PPLCNet_x2_5   | 9.0 | 906 | 76.60 | 93.00 | 5.39 |
| PPLCNet_x0_5_ssld | 1.9 | 47  | 66.10 | 86.46 | 2.05 |
| PPLCNet_x1_0_ssld | 3.0 | 161 | 74.39 | 92.09 | 2.46 |
| PPLCNet_x2_5_ssld | 9.0 | 906 | 80.82 | 95.33 | 5.39 |

where `_ssld` represents the model after using `SSLD distillation`. For details about `SSLD distillation`, see [SSLD distillation](../advanced_tutorials/distillation/distillation_en.md).

Performance comparison with other lightweight networks:

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|-------|-----------|----------|---------------|---------------|-------------|
| MobileNetV2_x0_25  | 1.5 | 34  | 53.21 | 76.52 | 2.47 |
| MobileNetV3_small_x0_35  | 1.7 | 15  | 53.03 | 76.37 | 3.02 |
| ShuffleNetV2_x0_33  | 0.6 | 24  | 53.73 | 77.05 | 4.30 |
| <b>PPLCNet_x0_25<b>  | <b>1.5<b> | <b>18<b>  | <b>51.86<b> | <b>75.65<b> | <b>1.74<b> |
| MobileNetV2_x0_5  | 2.0 | 99  | 65.03 | 85.72 | 2.85 |
| MobileNetV3_large_x0_35  | 2.1 | 41  | 64.32 | 85.46 | 3.68 |
| ShuffleNetV2_x0_5  | 1.4 | 43  | 60.32 | 82.26 | 4.65 |
| <b>PPLCNet_x0_5<b>   | <b>1.9<b> | <b>47<b>  | <b>63.14<b> | <b>84.66<b> | <b>2.05<b> |
| MobileNetV1_x1_0 | 4.3 | 578  | 70.99 | 89.68 | 3.38 |
| MobileNetV2_x1_0 | 3.5 | 327  | 72.15 | 90.65 | 4.26 |
| MobileNetV3_small_x1_25  | 3.6 | 100  | 70.67 | 89.51 | 3.95 |
| <b>PPLCNet_x1_0<b>     |<b> 3.0<b> | <b>161<b> | <b>71.32<b> | <b>90.03<b> | <b>2.46<b> |

We also test the inference speed of PPLCNet on other devices:

* Inference speed based on V100 GPU

| Models        | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) | FP32<br/>Batch Size=1\4<br/>(ms) | FP32<br/>Batch Size=8<br/>(ms) |
| ------------- | --------- | ----------------- | ---------------------------- | -------------------------------- | ------------------------------ |
| PPLCNet_x0_25 | 224       | 256               | 0.72                         | 1.17                             | 1.71                           |
| PPLCNet_x0_35 | 224       | 256               | 0.69                         | 1.21                             | 1.82                           |
| PPLCNet_x0_5  | 224       | 256               | 0.70                         | 1.32                             | 1.94                           |
| PPLCNet_x0_75 | 224       | 256               | 0.71                         | 1.49                             | 2.19                           |
| PPLCNet_x1_0  | 224       | 256               | 0.73                         | 1.64                             | 2.53                           |
| PPLCNet_x1_5  | 224       | 256               | 0.82                         | 2.06                             | 3.12                           |
| PPLCNet_x2_0  | 224       | 256               | 0.94                         | 2.58                             | 4.08                           |

* Inference speed based on SD855

| Models        | SD855 time(ms)<br>bs=1, thread=1 | SD855 time(ms)<br/>bs=1, thread=2 | SD855 time(ms)<br/>bs=1, thread=4 |
| ------------- | -------------------------------- | --------------------------------- | --------------------------------- |
| PPLCNet_x0_25 | 2.30                             | 1.62                              | 1.32                              |
| PPLCNet_x0_35 | 3.15                             | 2.11                              | 1.64                              |
| PPLCNet_x0_5  | 4.27                             | 2.73                              | 1.92                              |
| PPLCNet_x0_75 | 7.38                             | 4.51                              | 2.91                              |
| PPLCNet_x1_0  | 10.78                            | 6.49                              | 3.98                              |
| PPLCNet_x1_5  | 20.55                            | 12.26                             | 7.54                              |
| PPLCNet_x2_0  | 33.79                            | 20.17                             | 12.10                             |
| PPLCNet_x2_5  | 49.89                            | 29.60                             | 17.82                             |

<a name="1.3.2"></a>

### 1.3.2 Object Detection

For object detection, we adopt Baidu’s self-developed PicoDet, which focuses on lightweight object detection scenarios. The following table shows the comparison between the results of PP-LCNet and MobileNetV3 on the COCO dataset. PP-LCNet has an obvious advantage in both accuracy and speed.

| Backbone | mAP(%) | Latency(ms) |
|-------|-----------|----------|
MobileNetV3_large_x0_35 | 19.2 | 8.1 |
<b>PPLCNet_x0_5<b> | <b>20.3<b> | <b>6.0<b> |
MobileNetV3_large_x0_75 | 25.8 | 11.1 |
<b>PPLCNet_x1_0<b> | <b>26.9<b> | <b>7.9<b> |

<a name="1.3.3"></a>

### 1.3.3 Semantic Segmentation

For semantic segmentation, DeeplabV3+ is adopted. The following table presents the comparison between PP-LCNet and MobileNetV3 on the Cityscapes dataset, and PP-LCNet also stands out in terms of accuracy and speed.

| Backbone | mIoU(%) | Latency(ms) |
|-------|-----------|----------|
MobileNetV3_large_x0_5 | 55.42 | 135 |
<b>PPLCNet_x0_5<b> | <b>58.36<b> | <b>82<b> |
MobileNetV3_large_x0_75 | 64.53 | 151 |
<b>PPLCNet_x1_0<b> | <b>66.03<b> | <b>96<b> |


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

Please refer to [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/macos-pip_en.html) for more information about installation, for examples other versions.

<a name="2.2"></a>  

### 2.2 PaddleClas wheel Installation

The command of PaddleClas installation as bellow:

```  
pip3 install paddleclas
```

<a name="2.3"></a>

### 2.3 Prediction

* Prediction with CLI

```bash
paddleclas --model_name=PPLCNet_x1_0  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```

Results:
```
>>> result
class_ids: [8, 7, 86, 81, 85], scores: [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], label_names: ['hen', 'cock', 'partridge', 'ptarmigan', 'quail'], filename: docs/images/inference_deployment/whl_demo.jpg
Predict complete!
```  

**Note**: When replacing other scale models of PPLCNet, just replace `model_name`. For example, when changing the model at this time to `PPLCNet_x0_25`, you only need to change `--model_name=PPLCNet_x1_0` to `--model_name=PPLCNet_x0_25`.


* Prediction in Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='PPLCNet_x1_0')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result = clas.predict(infer_imgs)
print(next(result))
```

The result of demo above:

```
>>> result
[{'class_ids': [8, 7, 86, 81, 85], 'scores': [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], 'label_names': ['hen', 'cock', 'partridge', 'ptarmigan', 'quail'], 'filename': 'docs/images/inference_deployment/whl_demo.jpg'}]
```

**Note**: The result returned by model.predict() is a `generator`, so you need to use the `next()` function to call it or `for loop` to loop it. And it will predict with batch_size size batch and return the prediction results when called. The default batch_size is 1, and you also specify the batch_size when instantiating, such as `model = paddleclas.PaddleClas(model_name="PPLCNet_x1_0", batch_size=2)`.

<a name="3"></a>

## 3. Training, Evaluation and Inference

<a name="3.1"></a>  

### 3.1 Installation

Please refer to [Installation](../installation.md) to get the description about installation.

<a name="3.2"></a>

### 3.2 Dataset

Please prepare ImageNet-1k data at [ImageNet official website](https://www.image-net.org/).

Enter the `PaddleClas/` directory:

```
cd path_to_PaddleClas
```

Enter the `dataset/` directory, name the downloaded data `ILSVRC2012` , and the `ILSVRC2012` directory has the following data:

```
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

where `train/` and `val/` are the training set and validation set, respectively. `train_list.txt` and `val_list.txt` are the label files for the training set and validation set, respectively.

**Note:**
* About the contents format of `train_list.txt` and `val_list.txt`, please refer to [Description about Classification Dataset in PaddleClas](../data_preparation/classification_dataset_en.md).


<a name="3.3"></a>

### 3.3 Training

<a name="3.3.1"></a>

#### 3.3.1 Train ImageNet

The PPLCNet_x1_0 training configuration is provided in `ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml`, which can be started with the following script:  

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml
```


**Note:**

* The current model with the best accuracy will be saved in `output/PPLCNet_x1_0/best_model.pdparams`

<a name="3.3.2"></a>

#### 3.3.2 Fine-tuning based on ImageNet weights

If you are not training an ImageNet task, you need to change the configuration file and training method, such as reducing the learning rate, reducing the number of epochs, etc.

<a name="3.4"></a>

### 3.4 Evaluation

After training, you can use the following commands to evaluate the model.

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```
Among the above command, the argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.

<a name="3.5"></a>

### 3.5 Inference

After the model training is completed, the pre-trained model obtained from the training can be loaded for model prediction. A complete example is provided in the `tools/infer.py` of the model library, and the model prediction can be done by simply executing the following command:

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model
```

The results:
```
[{'class_ids': [8, 7, 86, 81, 85], 'scores': [0.91347, 0.03779, 0.0036, 0.00117, 0.00112], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ptarmigan', 'quail']}]
```

**Note**:

* Among the above command, argument `-o Global.pretrained_model="output/PPLCNet_x1_0/best_model"` specify the path of the best model weight file. You can specify other path if needed.


* The default test image is `docs/images/inference_deployment/whl_demo.jpg` ，And you can test other image, only need to specify the argument `-o Infer.infer_imgs=path_to_test_image`.

* The default output is the value of Top-5. If you want to output the value of Top-k, you can specify `-o Infer.PostProcess.topk=k`, where `k` is the value you specify.

* The default label mapping is based on the ImageNet dataset. If you change the dataset, you need to re-specify `Infer.PostProcess.class_id_map_file`. For the method of making the mapping file, please refer to `ppcls/utils/imagenet1k_label_list.txt`


<a name="4"></a>

## 4. Inference Deployment

<a name="4.1"></a>

### 4.1 Getting Paddle Inference Model

Paddle Inference is the original Inference Library of the PaddlePaddle, provides high-performance inference for server deployment. And compared with  directly based on the pretrained model, Paddle Inference can use tools to accelerate prediction, so as to achieve better inference performance. Please refer to [Paddle Inference](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html) for more information.

Paddle Inference need Paddle Inference Model to predict. Two process provided to get Paddle Inference Model. If want to use the provided by PaddleClas, you can download directly, click [Downloading Inference Model](#4.1.2).


<a name="4.1.1"></a>

### 4.1.1 Exporting Paddle Inference Model

The command about exporting Paddle Inference Model is as follow:

```bash
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml \
    -o Global.pretrained_model=output/PPLCNet_x1_0/best_model \
    -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_infer
```

After running above command, the inference model files would be saved in `deploy/models/PPLCNet_x1_0_infer`, as shown below:

```
├── PPLCNet_x1_0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```


<a name="4.1.2"></a>

### 4.1.2 Downloading Inference Model

You can also download directly.

```
cd deploy/models
# download the inference model and decompression
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar && tar -xf PPLCNet_x1_0_infer.tar
```

After decompression, the directory `models` should be shown below.
```
├── PPLCNet_x1_0_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="4.2"></a>

### 4.2 Prediction with Python


<a name="4.2.1"></a>  

#### 4.2.1 Image Prediction

Return the directory `deploy`:

```
cd ../
```

Run the following command to classify whether there are humans in the image `./images/ImageNet/ILSVRC2012_val_00000010.jpeg`.

```shell
# Use the following command to predict with GPU.
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer
# Use the following command to predict with CPU.
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer -o Global.use_gpu=False
```

The prediction results:

```
ILSVRC2012_val_00000010.jpeg:   class id(s): [153, 265, 204, 283, 229], score(s): [0.61, 0.11, 0.05, 0.03, 0.02], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'toy poodle', 'Lhasa, Lhasa apso', 'Persian cat', 'Old English sheepdog, bobtail']
```

<a name="4.2.2"></a>  

#### 4.2.2 Images Prediction

If you want to predict images in directory, please specify the argument `Global.infer_imgs` as directory path by `-o Global.infer_imgs`. The command is as follow.

```shell
# Use the following command to predict with GPU. If want to replace with CPU, you can add argument -o Global.use_gpu=False
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPLCNet_x1_0_infer -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:   class id(s): [153, 265, 204, 283, 229], score(s): [0.61, 0.11, 0.05, 0.03, 0.02], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'toy poodle', 'Lhasa, Lhasa apso', 'Persian cat', 'Old English sheepdog, bobtail']
ILSVRC2012_val_00010010.jpeg:   class id(s): [695, 551, 507, 531, 419], score(s): [0.11, 0.06, 0.03, 0.03, 0.03], label_name(s): ['padlock', 'face powder', 'combination lock', 'digital watch', 'Band Aid']
ILSVRC2012_val_00020010.jpeg:   class id(s): [178, 211, 209, 210, 236], score(s): [0.87, 0.03, 0.01, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Chesapeake Bay retriever', 'German short-haired pointer', 'Doberman, Doberman pinscher']
ILSVRC2012_val_00030010.jpeg:   class id(s): [80, 23, 93, 81, 99], score(s): [0.87, 0.01, 0.01, 0.01, 0.00], label_name(s): ['black grouse', 'vulture', 'hornbill', 'ptarmigan', 'goose']
```

<a name="4.3"></a>

### 4.3 Deployment with C++

PaddleClas provides an example about how to deploy with C++. Please refer to [Deployment with C++](../inference_deployment/cpp_deploy_en.md).

<a name="4.4"></a>

### 4.4 Deployment as Service

Paddle Serving is a flexible, high-performance carrier for machine learning models, and supports different protocol, such as RESTful, gRPC, bRPC and so on, which provides different deployment solutions for a variety of heterogeneous hardware and operating system environments. Please refer [Paddle Serving](https://github.com/PaddlePaddle/Serving) for more information.

PaddleClas provides an example about how to deploy as service by Paddle Serving. Please refer to [Paddle Serving Deployment](../inference_deployment/paddle_serving_deploy_en.md).

<a name="4.5"></a>

### 4.5 Deployment on Mobile

Paddle-Lite is an open source deep learning framework that designed to make easy to perform inference on mobile, embeded, and IoT devices. Please refer to [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) for more information.

PaddleClas provides an example of how to deploy on mobile by Paddle-Lite. Please refer to [Paddle-Lite deployment](../inference_deployment/paddle_lite_deploy_en.md).

<a name="4.6"></a>

### 4.6 Converting To ONNX and Deployment

Paddle2ONNX support convert Paddle Inference model to ONNX model. And you can deploy with ONNX model on different inference engine, such as TensorRT, OpenVINO, MNN/TNN, NCNN and so on. About Paddle2ONNX details, please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).

PaddleClas provides an example of how to convert Paddle Inference model to ONNX model by paddle2onnx toolkit and predict by ONNX model. You can refer to [paddle2onnx](../../../deploy/paddle2onnx/readme_en.md) for deployment details.

<a name="5"></a>
## 5. Reference

Reference to cite when you use PP-LCNet in a paper:
```
@misc{cui2021pplcnet,
      title={PP-LCNet: A Lightweight CPU Convolutional Neural Network},
      author={Cheng Cui and Tingquan Gao and Shengyu Wei and Yuning Du and Ruoyu Guo and Shuilong Dong and Bin Lu and Ying Zhou and Xueying Lv and Qiwen Liu and Xiaoguang Hu and Dianhai Yu and Yanjun Ma},
      year={2021},
      eprint={2109.15099},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
