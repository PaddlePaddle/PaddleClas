# PP-HGNeV2 Series
---
- [1. Model Introduction](#1)
    - [1.1 Model Overview](#1.1)
    - [1.2 Model Details](#1.2)
    - [1.3 Experimental Precision](#1.3)
- [2. Model Training, Evaluation, and Prediction](#2)
    - [2.1 Environment Configuration](#2.1)
    - [2.2 Data Preparation](#2.2)
    - [2.3 Model Training](#2.3)
      - [2.3.1 Training from Scratch on ImageNet](#2.3.1)
      - [2.3.2 Fine-tuning Other Classification Tasks Based on ImageNet Weights](#2.3.2)
    - [2.4 Model Evaluation](#2.4)
    - [2.5 Model Prediction](#2.5)
- [3. Model Inference Deployment](#3)
  - [3.1 Inference Model Preparation](#3.1)
  - [3.2 Inference Based on Python Prediction Engine](#3.2)
    - [3.2.1 Predicting a Single Image](#3.2.1)
    - [3.2.2 Batch Prediction Based on a Folder](#3.2.2)
  - [3.3 Inference Based on C++ Prediction Engine](#3.3)
  - [3.4 Serving](#3.4)
  - [3.5 On-device Deployment](#3.5)
  - [3.6 Paddle2ONNX Model Conversion and Prediction](#3.6)

<a name="1"></a>

## 1. Model Introduction

<a name="1.1"></a>

### 1.1 Model Introduction

PP-HGNetV2 (High Performance GPU Network V2) is the next-generation version of PP-HGNet, independently developed by Baidu PaddlePaddle's vision team. Building upon PP-HGNet, it has undergone further optimization and improvement, ultimately achieving the utmost in "Accuracy-Latency Balance" on NVIDIA GPU devices, with significantly higher accuracy than other models with similar inference speeds. It demonstrates strong performance in tasks such as single-label classification, multi-label classification, object detection, and semantic segmentation. A comparison of its accuracy versus prediction time with common server-side models is shown in the figure below.

![](../../images/models/V100_benchmark/v100.fp32.bs1.main_fps_top1_s.png)

* The GPU evaluation environment is based on a V100 machine, running 2100 times under the FP32+TensorRT configuration (excluding the warmup time of the first 100 runs).

<a name="1.2"></a>

### 1.2 Model Details

The specific improvements of PP-HGNetV2 over PP-HGNet are as follows:

- Improved the stem part of the PPHGNet network by stacking more 2x2 convolution kernels to learn richer local features, and using a smaller number of channels to enhance the inference speed for high-resolution tasks such as object detection and semantic segmentation;
- Replaced the relatively redundant standard convolutional layers in the later stages of PP-HGNet with a combination of PW + DW5x5, resulting in a network with fewer parameters while achieving a larger receptive field and further improving accuracy;
- Added the LearnableAffineBlock module, which can significantly improve the accuracy of smaller models with minimal increase in the number of parameters and without affecting inference time;
- Restructured the stage distribution of the PP-HGNet network to cover models of different scales from B0-B6, thereby meeting the requirements of different tasks.

In addition to the above improvements, compared with other models provided by PaddleClas, PP-HGNetV2 provides [SSLD](https://arxiv.org/abs/2103.05959) pre-trained weights with higher accuracy and stronger generalization ability by default, which perform better in downstream tasks.

<a name="1.3"></a>

### 1.3 Model Accuracy

The accuracy, speed metrics, pre-trained weights, and inference model weights links of PP-HGNetV2 are as follows:

| Model | Top-1 Acc(\%)(stage-2) | Top-5 Acc(\%)(stage-2) | Latency(ms) | Download link for stage-1 pretrained model | Download link for stage-2 pretrained model | Download link for inference model (stage-2) |
|:--: |:--: |:--: |:--: | :--: |:--: |:--: |
| PPHGNetV2_B0     | 77.77 | 93.91 | 0.52 |[Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_stage1_pretrained.pdparams)| [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B0_ssld_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B0_ssld_infer.tar) |
| PPHGNetV2_B1     | 79.18 | 94.57 | 0.58 |[Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_stage1_pretrained.pdparams)| [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B1_ssld_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B1_ssld_infer.tar) |
| PPHGNetV2_B2     | 81.74 | 95.88 | 0.95 |[Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_stage1_pretrained.pdparams)| [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B2_ssld_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B2_ssld_infer.tar) |
| PPHGNetV2_B3     | 82.98 | 96.43 | 1.18 |[Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_stage1_pretrained.pdparams)| [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B3_ssld_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B3_ssld_infer.tar) |
| PPHGNetV2_B4     | 83.57 | 96.72 | 1.46 |[Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_stage1_pretrained.pdparams)| [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B4_ssld_pretrained.pdparams) | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B4_ssld_infer.tar) |
| PPHGNetV2_B5     | 84.75 | 97.32 | 2.84 |[Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_stage1_pretrained.pdparams)| [Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B5_ssld_pretrained.pdparams) | [Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B5_ssld_infer.tar) |
| PPHGNetV2_B6     | 86.30 | 97.84 | 5.29 |[Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_stage1_pretrained.pdparams)| [Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNetV2_B6_ssld_pretrained.pdparams) | [Download Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNetV2_B6_ssld_infer.tar) |

**Notes:**

* Test environment: V100, FP32+TensorRT8.5, BS=1;
* To achieve higher precision for downstream tasks, the PP-HGNetV2 series provides `SSLD` pre-trained weights. For an introduction to `SSLD` and its training methods, refer to the [SSLD paper](https://arxiv.org/abs/2103.05959) and [SSLD training](../../training/advanced/knowledge_distillation.md). The stage-1 weights provided here are obtained through data mining and distillation training using ImageNet1k+ImageNet22k in the stage-1 phase of `SSLD`, while the stage-2 weights are obtained through distillation fine-tuning using ImageNet1k in the stage-2 phase of `SSLD`. In practical scenarios, the stage-1 weights exhibit better generalization, and it is recommended to directly use the stage-1 weights for downstream task training.

<a name="2"></a>

## 2. Model Training, Evaluation, and Prediction

<a name="2.1"></a>

### 2.1 Environment Configuration

* Installation: Please refer to the document [Environment Preparation](../../installation.md) to configure the PaddleClas runtime environment.

<a name="2.2"></a>

### 2.2 Data Preparation

Please prepare the data related to ImageNet-1k on the [official ImageNet website](https://www.image-net.org/).

Enter the PaddleClas directory.

```
cd path_to_PaddleClas
```

Enter the `dataset/` directory, name the downloaded data as `ILSVRC2012`, and store it here. The `ILSVRC2012` directory contains the following data:

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

Here, `train/` and `val/` are the training set and validation set, respectively. `train_list.txt` and `val_list.txt` are the label files for the training set and validation set, respectively.

**Note:**

* For the format description of `train_list.txt` and `val_list.txt`, refer to the [PaddleClas Classification Dataset Format Description](../../training/single_label_classification/dataset.md#1-dataset-format-description).

<a name="2.3"></a>

### 2.3 Model Training

<a name="2.3.1"></a>

#### 2.3.1 Training ImageNet from Scratch

Training configurations for PPHGNetV2 models of different sizes are provided in `ppcls/configs/ImageNet/PPHGNetV2/`, allowing you to load the configuration for the corresponding model for training. For example, to train `PPHGNetV2_B4`, you can initiate the training with the following script:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
        -o Global.output_dir=./output/PPHGNetV2_B4 \
        -o Arch.pretrained=False
```

**Note:**

* The model with the current best precision will be saved in `output/PPHGNetV2_B4/best_model.pdparams`;
* This section only demonstrates how to train from scratch on the ImageNet dataset. The configuration does not use aggressive training strategies or knowledge distillation training strategies, so the precision obtained from training is lower than that in Section [1.3](#1.3). If you want to achieve the precision in Section [1.3](#1.3), you can refer to [SSLD Training](../../training/advanced/knowledge_distillation.md), configure the relevant data, and load the [stage-1 configuration](../../../../ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4_ssld_stage1.yaml) and [stage-2 configuration](../../../../PPHGNetV2_B4_ssld_stage2.yaml) for training.

<a name="2.3.2"></a>

#### 2.3.2 Fine-tuning Other Classification Tasks Based on ImageNet Weights

When fine-tuning the model, it is necessary to load the pre-trained weights and reduce the learning rate to avoid damaging the original weights. For example, to fine-tune and train `PPHGNetV2_B4`, you can initiate the training with the following script:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/train.py \
        -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
        -o Global.epochs=30 \
        -o Global.output_dir=./output/PPHGNetV2_B4 \
        -o Optimizer.lr.learning_rate=0.05
```

**Note:**

* `epochs` and `learning_rate` can be adjusted according to the actual situation;
* For better generalization, the default weights loaded here are the weights obtained from `SSLD` stage-1 training.

<a name="2.4"></a>

### 2.4 Model Evaluation

After training the model, you can evaluate the model metrics using the following command.

```shell
python tools/eval.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model
```

Here, `-o Global.pretrained_model="output/PPHGNetV2_B4/best_model"` specifies the path where the current best model weights are located. If you want to specify other weights, simply replace the corresponding path.

<a name="2.5"></a>

### 2.5 Model Prediction

After the model training is completed, the pre-trained model obtained from training can be loaded for model prediction. A complete example is provided in `tools/infer.py` of the model library. You can complete model prediction simply by executing the following command:

```shell
python tools/infer.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model
```

The output results are as follows:

```
[{'class_ids': [8, 7, 86, 82, 83], 'scores': [0.92473, 0.07478, 0.00025, 7e-05, 6e-05], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse (also known as partridge, Bonasa umbellus)', 'prairie chicken (also known as prairie grouse, prairie fowl)']}]
```

**Note:**

* Here, `-o Global.pretrained_model="output/PPHGNetV2_B4/best_model"` specifies the path where the current best model weights are located. If you want to specify other weights, simply replace the corresponding path.

* By default, predictions are made for `docs/images/inference_deployment/whl_demo.jpg`. You can also make predictions for other images by adding the field `-o Infer.infer_imgs=xxx`.

* By default, the Top-5 values are output. If you want to output the Top-k values, you can specify `-o Infer.PostProcess.topk=k`, where `k` is the value you specify.

* The default label mapping is based on the ImageNet dataset. If you change the dataset, you need to re-specify `Infer.PostProcess.class_id_map_file`. For the method of creating this mapping file, refer to `ppcls/utils/imagenet1k_label_list.txt`.

<a name="3"></a>

## 3. Model Inference Deployment

<a name="3.1"></a>

### 3.1 Preparation of Inference Model

Paddle Inference is PaddlePaddle's native inference library, which operates on the server side and cloud, providing high-performance inference capabilities. Compared with directly making predictions based on pre-trained models, Paddle Inference can use MKLDNN, CUDNN, and TensorRT to accelerate predictions, thereby achieving better inference performance. For more information about the Paddle Inference inference engine, refer to the [official Paddle Inference tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html).

Here, we provide a script for converting weights and models. By executing this script, you can obtain the corresponding inference model:

```shell
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/PPHGNetV2/PPHGNetV2_B4.yaml \
    -o Global.pretrained_model=output/PPHGNetV2_B4/best_model \
    -o Global.save_inference_dir=deploy/models/PPHGNetV2_B4_infer
```

After executing this script, the `PPHGNetV2_B4_infer` folder will be generated under `deploy/models/`, and the `models` folder should have the following file structure:

```
├── PPHGNetV2_B4_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="3.2"></a>

### 3.2 Inference based on Python prediction engine

<a name="3.2.1"></a>

#### 3.2.1 Predicting a Single Image

Return to the `deploy` directory:

```
cd ../
```

Run the following command to classify the image `./images/ImageNet/ILSVRC2012_val_00000010.jpeg`.

```shell
# Use the following command to make predictions using GPU
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer
# Use the following command to make predictions using CPU
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer -o Global.use_gpu=False
```

The output results are as follows.

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 265], score(s): [0.94, 0.03, 0.02, 0.00, 0.00], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'toy poodle']
```

<a name="3.2.2"></a>

#### 3.2.2 Batch Prediction Based on Folders

If you want to predict images within a folder, you can directly modify the `Global.infer_imgs` field in the configuration file, or modify the corresponding configuration through the `-o` parameter below.

```shell
# Use the following command to make predictions using GPU. If you want to use CPU for prediction, you can add -o Global.use_gpu=False after the command
python3 python/predict_cls.py -c configs/inference_cls.yaml -o Global.inference_model_dir=models/PPHGNetV2_B4_infer -o Global.infer_imgs=images/ImageNet/
```

The classification results of all images in the folder will be output in the terminal, as shown below.

```
ILSVRC2012_val_00000010.jpeg:    class id(s): [332, 153, 283, 338, 265], score(s): [0.94, 0.03, 0.02, 0.00, 0.00], label_name(s): ['Angora, Angora rabbit', 'Maltese dog, Maltese terrier, Maltese', 'Persian cat', 'guinea pig, Cavia cobaya', 'toy poodle']
ILSVRC2012_val_00010010.jpeg:    class id(s): [626, 487, 531, 622, 593], score(s): [0.81, 0.08, 0.03, 0.01, 0.01], label_name(s): ['lighter, light, igniter, ignitor', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'digital watch', 'lens cap, lens cover', 'harmonica, mouth organ, harp, mouth harp']
ILSVRC2012_val_00020010.jpeg:    class id(s): [178, 211, 246, 236, 181], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['Weimaraner', 'vizsla, Hungarian pointer', 'Great Dane', 'Doberman, Doberman pinscher', 'Bedlington terrier']
ILSVRC2012_val_00030010.jpeg:    class id(s): [80, 83, 23, 8, 81], score(s): [1.00, 0.00, 0.00, 0.00, 0.00], label_name(s): ['black grouse', 'prairie chicken, prairie grouse, prairie fowl', 'vulture', 'hen', 'ptarmigan']
```

<a name="3.3"></a>

### 3.3 Inference Based on C++ Prediction Engine

PaddleClas provides examples of inference based on the C++ prediction engine. You can refer to [Server-side C++ Prediction](../../deployment/image_classification/cpp/linux.md) to complete the corresponding inference deployment. If you are using the Windows platform, you can refer to [Compilation Guide Based on Visual Studio 2019 Community CMake](../../deployment/image_classification/cpp/windows.md) to complete the compilation of the prediction library and model prediction.

<a name="3.4"></a>

### 3.4 Serving

Paddle Serving provides high-performance, flexible, and easy-to-use industrial-grade online inference services. Paddle Serving supports multiple protocols such as RESTful, gRPC, and bRPC, and offers inference solutions in various heterogeneous hardware and operating system environments. For more information about Paddle Serving, refer to the [Paddle Serving code repository](https://github.com/PaddlePaddle/Serving).

PaddleClas provides examples of model serving deployment based on Paddle Serving. You can refer to [Model Serving Deployment](../../deployment/image_classification/paddle_serving.md) to complete the corresponding deployment work.

<a name="3.5"></a>

### 3.5 On-device Deployment

Paddle Lite is a high-performance, lightweight, flexible, and easily extensible deep learning inference framework designed to support multiple hardware platforms, including mobile, embedded, and server-side platforms. For more information about Paddle Lite, please refer to the [Paddle Lite code repository](https://github.com/PaddlePaddle/Paddle-Lite).

PaddleClas provides examples of on-device model deployment based on Paddle Lite. You can refer to [On-device Deployment](../../deployment/image_classification/paddle_lite.md) to complete the corresponding deployment tasks.

<a name="3.6"></a>

### 3.6 Paddle2ONNX Model Conversion and Prediction

Paddle2ONNX supports converting PaddlePaddle model formats to ONNX model formats. Through ONNX, Paddle models can be deployed to various inference engines, including TensorRT/OpenVINO/MNN/TNN/NCNN, as well as other inference engines or hardware that support the ONNX open-source format. For more information about Paddle2ONNX, refer to the [Paddle2ONNX code repository](https://github.com/PaddlePaddle/Paddle2ONNX).

PaddleClas provides examples of using Paddle2ONNX to convert inference models to ONNX models and perform inference predictions. You can refer to [Paddle2ONNX Model Conversion and Prediction](../../deployment/image_classification/paddle2onnx.md) to complete the corresponding deployment tasks.