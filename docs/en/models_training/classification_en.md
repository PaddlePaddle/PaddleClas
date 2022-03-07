# Image Classification

------

Image Classification is a fundamental task that classifies the image by semantic information and assigns it to a specific label. Image Classification is the foundation of Computer Vision tasks, such as object detection, image segmentation, object tracking and behavior analysis. Image Classification has comprehensive applications, including face recognition and smart video analysis in the security and protection field, traffic scenario recognition in the traffic field, image retrieval and electronic photo album classification in the internet industry, and image recognition in the medical industry.

Generally speaking, Image Classification attempts to comprehend an entire image as a whole by feature engineering and assigns labels by a classifier. Hence, how to extract the features of image is the essential part. Before deep learning, the most used classification method is the Bag of Words model. However, Image Classification based on deep learning can learn the hierarchical feature description by supervised and unsupervised learning, replacing the manually image feature selection. Recently, Convolution Neural Network in deep learning has an awesome performance in the image field. CNN uses the pixel information as the input to get the all information of images. Additionally, since the model uses convolution to extract features,  and  the output is classification result. Thus, this kind of end-to-end method achieves ideal performance and is applied widely.

Image Classification is a very basic but important field in the subject of computer vision. Its research results have always influenced the development of computer vision and even deep learning. Image classification has many sub-fields, such as multi-label image classification and fine-grained image classification. Here is only a brief description of single-label image classification.

See [here](../algorithm_introduction/image_classification_en.md) for the detailed introduction of image classification algorithms.

## Catalogue

- [1. Dataset Introduction](#1)
  - [1.1 ImageNet-1k](#1.1)
  - [1.2 CIFAR-10/CIFAR-100](#1.2)
- [2. Image Classification Process](#2)
  - [2.1 Data and Its Preprocessing](#2.1)
  - [2.2 Prepare the Model](#2.2)
  - [2.3 Train the Model](#2.3)
  - [2.4 Evaluate the Model](#2.4)
- [3. Application Methods](#3)
  - [3.1 Training and Evaluation on CPU or Single GPU](#3.1)
    - [3.1.1 Model Training](#3.1.1)
    - [3.1.2 Model Finetuning](#3.1.2)
    - [3.1.3 Resume Training](#3.1.3)
    - [3.1.4 Model Evaluation](#3.1.4)
  - [3.2 Training and Evaluation on Linux+ Multi-GPU](#3.2)
    - [3.2.1 Model Training](#3.2.1)
    - [3.2.2 Model Finetuning](#3.2.2)
    - [3.2.3 Resume Training](#3.2.3)
    - [3.2.4 Model Evaluation](#3.2.4)
  - [3.3 Use the Pre-trained Model to Predict](#3.3)
  - [3.4 Use the Inference Model to Predict](#3.4)

<a name="1"></a>

## 1. Dataset Introduction

<a name="1.1"></a>

### 1.1 ImageNet-1k

The ImageNet is a large-scale visual database for the research of visual object recognition. More than 14 million images have been annotated manually to point out objects in the picture in this project, and at least more than 1 million images provide bounding box. ImageNet-1k is a subset of the ImageNet dataset, which contains 1000 categories. The training set contains 1281167 image data, and the validation set contains 50,000 image data. Since 2010, the ImageNet project has held an image classification competition every year, which is the ImageNet Large-scale Visual Recognition Challenge (ILSVRC). The dataset used in the challenge is ImageNet-1k. So far, ImageNet-1k has become one of the most important data sets for the development of computer vision, and it promotes the development of the entire computer vision. The initialization models of many computer vision downstream tasks are based on the weights trained on this dataset.

<a name="1.2"></a>

### 1.2 CIFAR-10/CIFAR-100

The CIFAR-10 dataset consists of 60,000 color images in 10 categories, with an image resolution of 32x32, and each category has 6000 images, including 5000 in the training set and 1000 in the validation set. 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships and trucks. The CIFAR-100 data set is an extension of CIFAR-10. It consists of 60,000 color images in 100 classes, with an image resolution of 32x32, and each class has 600 images, including 500 in the training set and 100 in the validation set. Researchers can try different algorithms quickly because these two data sets are small in scale. These two datasets are also commonly used data sets for testing the quality of models in the image classification field.

<a name="2"></a>

## 2. Image Classification Process

The prepared training data is preprocessed and then passed through the image classification model. The output of the model and the real label are used in a cross-entropy loss function. This loss function describes the convergence direction of the model.  Then the corresponding gradient descent for the final loss function is calculated and returned to the model,  which update the weight of the model by optimizers. Finally, an image classification model can be obtained.

<a name="2.1"></a>

### 2.1 Data Preprocessing

The quality and quantity of data often determine the performance of a model. In the field of image classification, data includes images and labels. In most cases, labeled data is scarce, so the amount of data is difficult to reach the level of saturation of the model. In order to enable the model to learn more image features, a lot of image transformation or data augmentation is required before the image enters the model, so as to ensure the diversity of input image data and ensure that the model has better generalization capabilities. PaddleClas provides standard image transformation for training ImageNet-1k, and also provides 8 data augmentation methods. For related codes, please refer to [data preprocess](../../../ppcls/data/preprocess)，The configuration file refer to [Data Augmentation Configuration File](../../../ppcls/configs/ImageNet/DataAugment). For related algorithms, please refer to [data augment algorithms](../algorithm_introduction/DataAugmentation_en.md).

<a name="2.2"></a>

### 2.2 Prepare the Model

After the data is determined, the model often determines the upper limit of the final accuracy. In the field of image classification, classic models emerge in an endless stream. PaddleClas provides 35 series and a total of 164 ImageNet pre-trained models. For specific accuracy, speed and other indicators, please refer to [Backbone Network Introduction](../algorithm_introduction/ImageNet_models_en.md).

<a name="2.3"></a>

### 2.3 Train

After preparing the data and model, you can start training the model and update the parameters of the model. After many iterations, a trained model can finally be obtained for image classification tasks. The training process of image classification requires a lot of experience and involves the setting of many hyperparameters. PaddleClas provides a series of [training tuning methods](./train_strategy_en.md), which can quickly help you obtain a high-precision model.

PaddleClas support training with VisualDL to visualize the metric. VisualDL is a visualization analysis tool of PaddlePaddle, provides a variety of charts to show the trends of parameters, and visualizes model structures, data samples, histograms of tensors, PR curves , ROC curves and high-dimensional data distributions. It enables users to understand the training process and the model structure more clearly and intuitively so as to optimize models efficiently. For more information, please refer to [VisualDL](../others/VisualDL_en.md).

<a name="2.4"></a>

### 2.4 Evaluation

After a model is trained, the evaluation results of the model on the validation set can determine the performance of the model. The evaluation index is generally Top1-Acc or Top5-Acc. The higher the index, the better the model performance.

<a name="3"></a>

## 3. Application Methods

Please refer to [Installation](../installation/install_paddleclas_en.md) to setup environment at first, and prepare flower102 dataset by following the instruction mentioned in the [Quick Start](../quick_start/quick_start_classification_new_user_en.md).

So far, PaddleClas supports the following training/evaluation environments:

```
└── CPU/Single GPU
    ├── Linux
    └── Windows

└── Multi card GPU
    └── Linux
```

<a name="3.1"></a>

### 3.1 Training and Evaluation on CPU or Single GPU

If training and evaluation are performed on CPU or single GPU, it is recommended to use the `tools/train.py` and `tools/eval.py`. For training and evaluation in multi-GPU environment on Linux, please refer to [3.2 Training and evaluation on Linux+GPU](#3.2).

<a name="3.1.1"></a>

#### 3.1.1 Model Training

After preparing the configuration file, The training process can be started in the following way.

```shell
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=False \
    -o Global.device=gpu
```

Among them, `-c` is used to specify the path of the configuration file, `-o` is used to specify the parameters needed to be modified or added, `-o Arch.pretrained=False` means to not using pre-trained models. `-o Global.device=gpu` means to use GPU for training. If you want to use the CPU for training, you need to set `Global.device` to `cpu`.

Of course, you can also directly modify the configuration file to update the configuration. For specific configuration parameters, please refer to [Configuration Document](config_description_en.md).

The output log examples are as follows:

- If mixup or cutmix is used in training, top-1 and top-k (default by 5) will not be printed in the log:

  ```
  ...
  [Train][Epoch 3/20][Avg]CELoss: 6.46287, loss: 6.46287
  ...
  [Eval][Epoch 3][Avg]CELoss: 5.94309, loss: 5.94309, top1: 0.01961, top5: 0.07941
  ...
  ```

- If mixup or cutmix is not used during training, in addition to the above information, top-1 and top-k (The default is 5) will also be printed in the log:

  ```
  ...
  [Train][Epoch 3/20][Avg]CELoss: 6.12570, loss: 6.12570, top1: 0.01765, top5: 0.06961
  ...
  [Eval][Epoch 3][Avg]CELoss: 5.40727, loss: 5.40727, top1: 0.07549, top5: 0.20980
  ...
  ```

During training, you can view loss changes in real time through `VisualDL`, see [VisualDL](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/en/extension/VisualDL_en.md) for details.

<a name="3.1.2"></a>

#### 3.1.2 Model Finetuning

After correcting config file, you can load pretrained model  weight to finetune. The command is as follows:

```shell
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=True \
    -o Global.device=gpu
```

Among them,`Arch.pretrained` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file. You can also set it into `True` to use pretrained weights that trained in ImageNet1k.

We also provide a lot of pre-trained models trained on the ImageNet-1k dataset. For the model list and download address, please refer to the [model library overview](../algorithm_introduction/ImageNet_models_en.md).

<a name="3.1.3"></a>

#### 3.1.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```shell
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
    -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter.

**Note**:

- The `-o Global.checkpoints` parameter does not need to include the suffix of the checkpoints. The above training command will generate the checkpoints as shown below during the training process. If you want to continue training from the epoch `5`, Just set the `Global.checkpoints` to `../output/MobileNetV3_large_x1_0/epoch_5`, PaddleClas will automatically fill in the `pdopt` and `pdparams` suffixes. Files in the output directory are structured as follows：

  ```
  output
  ├── MobileNetV3_large_x1_0
  │   ├── best_model.pdopt
  │   ├── best_model.pdparams
  │   ├── best_model.pdstates
  │   ├── epoch_1.pdopt
  │   ├── epoch_1.pdparams
  │   ├── epoch_1.pdstates
      .
      .
      .
  ```

<a name="3.1.4"></a>

#### 3.1.4 Model Evaluation

The model evaluation process can be started as follows.

```shell
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

The above command will use `./configs/quick_start/MobileNetV3_large_x1_0.yaml` as the configuration file to evaluate the model `./output/MobileNetV3_large_x1_0/best_model`. You can also set the evaluation by changing the parameters in the configuration file, or you can update the configuration with the `-o` parameter, as shown above.

Some of the configurable evaluation parameters are described as follows:

- `Arch.name`：Model name
- `Global.pretrained_model`：The path of the model file to be evaluated

**Note：** When loading the model to be evaluated, you only need to specify the path of the model file stead of the suffix. PaddleClas will automatically add the `.pdparams` suffix, such as [3.1.3 Resume Training](#3.1.3).

When loading the model to be evaluated, you only need to specify the path of the model file stead of the suffix. PaddleClas will automatically add the `.pdparams` suffix, such as [3.1.3 Resume Training](../models_training/classification_en.md#3.1.3).

<a name="3.2"></a>

### 3.2 Training and Evaluation on Linux+ Multi-GPU

If you want to run PaddleClas on Linux with GPU, it is highly recommended to use `paddle.distributed.launch` to start the model training script(`tools/train.py`) and evaluation script(`tools/eval.py`), which can start on multi-GPU environment more conveniently.

<a name="3.2.1"></a>

#### 3.2.1 Model Training

The training process can be started in the following way. `paddle.distributed.launch` specifies the GPU running card number by setting `gpus`:

```shell
# PaddleClas initiates multi-card multi-process training via launch

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml
```

The format of output log information is the same as above, see [3.1.1 Model training](#3.1.1) for details.

<a name="3.2.2"></a>

#### 3.2.2 Model Finetuning

After configuring the yaml file, you can finetune it by loading the pretrained weights. The command is as below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Arch.pretrained=True
```

Among them, `Arch.pretrained` is set to `True` or `False`. It also can be used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

There contains a lot of examples of model finetuning in the [new user version](../quick_start/quick_start_classification_new_user_en.md) and [professional version](../quick_start/quick_start_classification_professional_en.md) of PaddleClas Trial in 30 mins. You can refer to this tutorial to finetune the model on a specific dataset.

<a name="3.2.3"></a>

#### 3.2.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
        -o Global.device=gpu
```

The configuration file does not need to be modified. You only need to add the `Global.checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter as described in [3.1.3 Resume training](#3.1.3).

<a name="3.2.4"></a>

#### 3.2.4 Model Evaluation

The model evaluation process can be started as follows.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/eval.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

About parameter description, see [3.1.4 Model evaluation](#3.1.4) for details.

<a name="3.3"></a>

### 3.3 Use the Pre-trained Model to Predict

After the training is completed, you can predict by using the pre-trained model obtained by the training. A complete example is provided in `tools/infer/infer.py` of the model library, run the following command to conduct model prediction:

```
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

Parameters：

- `Infer.infer_imgs`：The path of the image file or folder to be predicted.
- `Global.pretrained_model`：Weight file path, such as`./output/MobileNetV3_large_x1_0/best_model`

<a name="3.4"></a>

### 3.4 Use the Inference Model to Predict

By exporting the inference model，PaddlePaddle supports inference using prediction engines, which will be introduced next. Firstly, you should export inference model using `tools/export_model.py`.

```shell
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=output/MobileNetV3_large_x1_0/best_model
```

Among them, `Global.pretrained_model` parameter is used to specify the model file path that does not need to include the file suffix name.（such as [3.1.3 Resume Training](#3.1.3)）。

The above command will generate the model structure file (`inference.pdmodel`) and the model weight file (`inference.pdiparams`), and then the inference engine can be used for inference:

Go to the deploy directory:

```
cd deploy
```

Using inference engine to inference. Because the mapping file of ImageNet1k dataset is `class_id_map_file` by default, here it should be set to None.

```shell
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/flowers102/jpg/image_00001.jpg \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None
```

Among them：

- `Global.infer_imgs`：The path of the image file to be predicted.
- `Global.inference_model_dir`：Model structure file path, such as `../inference/`.
- `Global.use_tensorrt`：Whether to use the TesorRT, default by `False`.
- `Global.use_gpu`：Whether to use the GPU, default by `True`.
- `Global.enable_mkldnn`：Wheter to use `MKL-DNN`, default by `False`. It is valid when `Global.use_gpu` is `False`.
- `Global.use_fp16`：Whether to enable FP16, default by `False`.

Note: If you want to use `Transformer` series models, such as `DeiT_***_384`, `ViT_***_384`, etc.，please pay attention to the input size of model, and need to set `resize_short=384`, `resize=384`.

If you want to evaluate the speed of the model, it is recommended to enable TensorRT to accelerate for GPU, and MKL-DNN for CPU.
