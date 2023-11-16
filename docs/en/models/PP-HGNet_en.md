# PP-HGNet Series
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

<a name='1'></a>

## 1. Introduction

<a name='1.1'></a>

### 1.1 Model Introduction

PP-HGNet is a high-performance backbone network developed by the PaddleCV team that is more suitable for GPU devices. Compared with other SOTA models on GPU devices, this model has higher accuracy under the same latency. At the same latency, the model is 3.8 percentage points higher than the ResNet34-D model, 2.4 percentage points higher than the ResNet50-D model, and 4.7 percentage points higher than the ResNet50-D model after using the SSLD distillation strategy. At the same time, under the same accuracy, its latency is much smaller than the mainstream VisionTransformer model. We will release the technical report to arxiv recently, so stay tuned.

<a name='1.2'></a>

### 1.2 Model Details

The author of PP-HGNet analyzes and summarizes the current GPU-friendly networks for GPU devices, and uses 3x3 standard convolutions as much as possible (the highest computational density). Here, VOVNet is used as the base model, and the improvement points that are mainly beneficial to GPU acceleration will be integrated. In the end, under the same latency of PP-HGNet, the accuracy greatly surpasses other backbones

The overall structure of the PP-HGNet backbone network is as follows:

![](../../../images/PP-HGNet/PP-HGNet.png)

Among them, PP-HGNet is composed of multiple HG-Blocks. The details of HG-Blocks are as follows:

![](../../../images/PP-HGNet/PP-HGNet-block.png)

<a name='1.3'></a>

### 1.3 Result

The accuracy and latency of PP-HGNet are as follows:

| Model | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) | Pre-trained model download address | Inference model download address |
|:--: |:--: |:--: |:--: | :--: |:--: |
| PPHGNet_tiny      | 79.83 | 95.04 | 1.77 | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_tiny_pretrained.pdparams) | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_infer.tar) |
| PPHGNet_tiny_ssld  | 81.95 | 96.12 | 1.77 | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_tiny_ssld_pretrained.pdparams) | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_ssld_infer.tar) |
| PPHGNet_small     | 81.51| 95.82 | 2.52  | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_small_pretrained.pdparams) | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_small_infer.tar) |
| PPHGNet_small_ssld | 83.82| 96.81 | 2.52  | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_small_ssld_pretrained.pdparams) | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_small_ssld_infer.tar) |
| PPHGNet_base_ssld | 85.00| 97.35 | 5.97   | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_base_ssld_pretrained.pdparams) | [download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_base_ssld_infer.tar) |

**Note:**

* 1. `_ssld` represents the model after using `SSLD distillation`. For details about `SSLD distillation`, see [SSLD distillation](../advanced_tutorials/distillation/distillation_en.md).
* 2. More metrics and weights of PP-HGNet, so stay tuned.

The comparison between PP-HGNet and other models is as follows. The test machine is NVIDIA® Tesla® V100, the TensorRT engine is turned on, and the precision type is FP32. Under the same latency, the accuracy of PP-HGNet surpasses other SOTA CNN models, and in comparison with the SwinTransformer model, it is more than 2 times faster with higher accuracy.

| Model | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|:--: |:--: |:--: |:--: |
| ResNet34                 | 74.57      | 92.14       | 1.97        |
| ResNet34_vd              | 75.98      | 92.98       | 2.00        |
| EfficientNetB0           | 77.38      | 93.31       | 1.96        |
| <b>PPHGNet_tiny<b>       | <b>79.83<b> | <b>95.04<b> | <b>1.77<b> |
| <b>PPHGNet_tiny_ssld<b>  | <b>81.95<b> | <b>96.12<b> | <b>1.77<b> |
| ResNet50                 | 76.50      | 93.00       | 2.54        |
| ResNet50_vd              | 79.12      | 94.44       | 2.60        |
| ResNet50_rsb             | 80.40      |         |     2.54        |
| EfficientNetB1           | 79.15      | 94.41       | 2.88        |
| SwinTransformer_tiny     | 81.2      | 95.5       | 6.59        |
| <b>PPHGNet_small<b>      | <b>81.51<b>| <b>95.82<b> | <b>2.52<b>  |
| <b>PPHGNet_small_ssld<b> | <b>83.82<b>| <b>96.81<b> | <b>2.52<b>  |
| Res2Net200_vd_26w_4s_ssld| 85.13      | 97.42       | 11.45       |
| ResNeXt101_32x48d_wsl    | 85.37      | 97.69       | 55.07       |
| SwinTransformer_base     | 85.2       | 97.5        | 13.53       |  
| <b>PPHGNet_base_ssld<b> | <b>85.00<b>| <b>97.35<b> | <b>5.97<b>   |

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
paddleclas --model_name=PPHGNet_small  --infer_imgs="docs/images/inference_deployment/whl_demo.jpg"
```

Results:
```
>>> result
class_ids: [8, 7, 86, 82, 81], scores: [0.71479, 0.08682, 0.00806, 0.0023, 0.00121], label_names: ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'ptarmigan'], filename: docs/images/inference_deployment/whl_demo.jpg
Predict complete!
```  

**Note**: When replacing other scale models of PPHGNet, just replace `model_name`. For example, when changing the model at this time to `PPHGNet_tiny`, you only need to change `--model_name=PPHGNet_small` to `--model_name=PPHGNet_tiny`.


* Prediction in Python
```python
from paddleclas import PaddleClas
clas = PaddleClas(model_name='PPHGNet_small')
infer_imgs = 'docs/images/inference_deployment/whl_demo.jpg'
result = clas.predict(infer_imgs)
print(next(result))
```

The result of demo above:

```
>>> result
[{'class_ids': [8, 7, 86, 82, 81], 'scores': [0.77132, 0.05122, 0.00755, 0.00199, 0.00115], 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'ptarmigan'], 'filename': 'docs/images/inference_deployment/whl_demo.jpg'}]
```

**Note**: The result returned by model.predict() is a `generator`, so you need to use the `next()` function to call it or `for loop` to loop it. And it will predict with batch_size size batch and return the prediction results when called. The default batch_size is 1, and you also specify the batch_size when instantiating, such as `model = paddleclas.PaddleClas(model_name="PPHGNet_small", batch_size=2)`.

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

The PPHGNet_small training configuration is provided in `ppcls/configs/ImageNet/PPHGNet/PPHGNet_small.yaml`, which can be started with the following script:  

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPHGNet/PPHGNet_small.yaml
```


**Note:**

* The current model with the best accuracy will be saved in `output/PPHGNet_small/best_model.pdparams`

<a name="3.3.2"></a>

#### 3.3.2 Fine-tuning based on ImageNet weights

If you are not training an ImageNet task, you need to change the configuration file and training method, such as reducing the learning rate, reducing the number of epochs, etc.

<a name="3.4"></a>

### 3.4 Evaluation

After training, you can use the following commands to evaluate the model.

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/PPHGNet/PPHGNet_small.yaml \
    -o Global.pretrained_model=output/PPHGNet_small/best_model
```
Among the above command, the argument `-o Global.pretrained_model="output/PPHGNet_small/best_model"` specify the path of the best model weight file. You can specify other path if needed.

<a name="3.5"></a>

### 3.5 Inference

After the model training is completed, the pre-trained model obtained from the training can be loaded for model prediction. A complete example is provided in the `tools/infer.py` of the model library, and the model prediction can be done by simply executing the following command:

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/PPHGNet/PPHGNet_small.yaml \
    -o Global.pretrained_model=output/PPHGNet_small/best_model
```

The results:
```
[{'class_ids': [8, 7, 86, 82, 81], 'scores': [0.71479, 0.08682, 0.00806, 0.0023, 0.00121], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'ptarmigan']}]
```

**Note**:

* Among the above command, argument `-o Global.pretrained_model="output/PPHGNet_small/best_model"` specify the path of the best model weight file. You can specify other path if needed.


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
    -c ppcls/configs/ImageNet/PPHGNet/PPHGNet_small.yaml \
    -o Global.pretrained_model=output/PPHGNet_small/best_model \
    -o Global.save_inference_dir=deploy/models/PPHGNet_small_infer
```

After running above command, the inference model files would be saved in `deploy/models/PPHGNet_small_infer`, as shown below:

```
├── PPHGNet_small_infer
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
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_small_infer.tar && tar -xf PPHGNet_small_infer.tar
```

After decompression, the directory `models` should be shown below.
```
├── PPHGNet_small_infer
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
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNet_small_infer
# Use the following command to predict with CPU.
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNet_small_infer -o Global.use_gpu=False
```

The prediction results:

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 204], score(s): [0.50, 0.05, 0.02, 0.01, 0.01], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'Lhasa, Lhasa apso']
```

<a name="4.2.2"></a>  

#### 4.2.2 Images Prediction

If you want to predict images in directory, please specify the argument `Global.infer_imgs` as directory path by `-o Global.infer_imgs`. The command is as follow.

```shell
# Use the following command to predict with GPU. If want to replace with CPU, you can add argument -o Global.use_gpu=False
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNet_small_infer -o Global.infer_imgs=images/ImageNet/
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 204], score(s): [0.50, 0.05, 0.02, 0.01, 0.01], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'Lhasa, Lhasa apso']
ILSVRC2012_val_00010010.jpeg:    class id(s): [626, 622, 531, 487, 633], score(s): [0.68, 0.02, 0.02, 0.02, 0.02], label_name(s): ['lighter, light, igniter, ignitor', 'lens cap, lens cover', 'digital watch', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', "loupe, jeweler's loupe"]
ILSVRC2012_val_00020010.jpeg:    class id(s): [178, 211, 171, 246, 741], score(s): [0.82, 0.00, 0.00, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Italian greyhound', 'Great Dane', 'prayer rug, prayer mat']
ILSVRC2012_val_00030010.jpeg:    class id(s): [80, 83, 136, 23, 93], score(s): [0.84, 0.00, 0.00, 0.00, 0.00], label_name(s): ['black grouse', 'prairie chicken, prairie grouse, prairie fowl', 'European gallinule, Porphyrio porphyrio', 'vulture', 'hornbill']
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
