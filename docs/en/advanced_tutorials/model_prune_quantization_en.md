# Model Quantization and Pruning

Complex models are conducive to better model performance, but they may also lead to certain redundancy. This section presents ways to streamline the model, including model quantization (quantization training and offline quantization) and model pruning.

Model quantization reduces the full precision to a fixed number of points to lower the redundancy and achieve the purpose of simplifying the model computation and improving model inference performance. Model quantization can reduce the size of model parameters by converting its precision from FP32 to Int8 without losing model precision, followed by accelerated computation, creating a quantized model with more speed advantages when deployed on mobile devices.

Model pruning decreases the number of model parameters by cutting out the unimportant convolutional kernels in the CNN, thus bringing down the computational complexity.

This tutorial explains how to use PaddleSlim, PaddlePaddle's model compression library, for PaddleClas compression, i.e., pruning and quantization. [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) integrates a variety of common and leading model compression functions such as model pruning, quantization (including quantization training and offline quantization), distillation, and neural network search. If you are interested, please follow us and learn more.

To start with, you are recommended to learn [PaddleClas Training](../models_training/classification_en.md) and [PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html), see [Model Pruning and Quantization Algorithms](../algorithm_introduction/model_prune_quantization_en.md) for related pruning and quantization methods.

------

## Catalogue

- [1. Prepare the Environment](#1)
  - [1.1 Install PaddleSlim](#1.1)
  - [1.2 Prepare the Trained Model](#1.2)
- [2. Quick Start](#2)
  - [2.1 Model Quantization](#2.1)
    - [2.1.1 Online Quantization Training](#2.1.1)
    - [2.1.2 Offline Quantization](#2.1.2)
  - [2.2 Model Pruning](#2.2)
- [3. Export the Model](#3)
- [4. Deploy the Model](#4)
- [5. Hyperparameter Training](#5)

<a name="1"></a>

## 1. Prepare the Environment

Once a model has been trained, you can adopt quantization or pruning to further compress the model size and speed up the inference.

Five steps are included：

1. Install PaddleSlim
2. Prepare the trained the model
3. Compress the model
4. Export quantized inference model
5. Inference and deployment of the quantized model

<a name="1.1"></a>

### 1.1 Install PaddleSlim

- You can adopt pip install for installation.

```
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- You can also install it from the source code with the latest features of PaddleSlim.

```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python3.7 setup.py install
```

<a name="1.2"></a>

### 1.2 Prepare the Trained Model

PaddleClas offers a list of trained [models](../models/models_intro_en.md). If the model to be quantized is not in the list, you need to follow the [regular training](../models_training/classification_en.md) method to get the trained model.

<a name="2"></a>

## 2. Quick Start

Go to PaddleClas root directory

```shell
cd PaddleClas
```

Related code for `slim` training has been integrated under `ppcls/engine/`, and the offline quantization code can be found in `deploy/slim/quant_post_static.py`.

<a name="2.1"></a>

### 2.1 Model Quantization

Quantization training includes offline and online training. Online quantitative training, the more effective one, requires loading a pre-trained model, which can be quantized after defining the strategy.

<a name="2.1.1"></a>

#### 2.1.1 Online Quantization Training

Try the following command：

- CPU/Single GPU

Take CPU for example, if you use GPU, change the `cpu` to `gpu`.

```
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_quantization.yaml -o Global.device=cpu
```

The parsing of the `yaml` file is described in [reference document](../models_training/config_description_en.md). For accuracy, the `pretrained model` has already been adopted by the `yaml` file.

- Launch in single-machine multi-card/ multi-machine multi-card mode

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -c ppcls/configs/slim/ResNet50_vd_quantization.yaml
```

<a name="2.1.2"></a>

#### 2.1.2 Offline Quantization

**Note**: Currently, the `inference model` exported from the trained model is a must for offline quantization. See the [tutorial](../inference_deployment/export_model_en.md) for general export of the  `inference model`.

Normally, offline quantization may lose more accuracy.

After generating the `inference model`, the offline quantization is run as follows:

```shell
python3.7 deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

The `inference model` is stored in`Global.save_inference_dir`.

Successfully executed, the `quant_post_static_model` folder is created in the `Global.save_inference_dir`, where the generated offline quantization models are stored and can be deployed directly without re-exporting the models.

<a name="2.2"></a>

### 2.2 Model Pruning

Trying the following command：

- CPU/Single GPU

Take CPU for example, if you use GPU, change the `cpu` to `gpu`.

```shell
python3.7 tools/train.py -c ppcls/configs/slim/ResNet50_vd_prune.yaml -o Global.device=cpu
```

- Launch in single-machine single-card/ single-machine multi-card/ multi machine multi-card mode

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
      tools/train.py \
      -c ppcls/configs/slim/ResNet50_vd_prune.yaml
```

<a name="3"></a>

## 3. Export the Model

Having obtained the saved model after online quantization training and pruning, it can be exported as an inference model for inference deployment. Here we take model pruning as an example:

```
python3.7 tools/export.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
    -o Global.pretrained_model=./output/ResNet50_vd/best_model \
    -o Global.save_inference_dir=./inference
```

<a name="4"></a>

## 4. Deploy the Model

The exported model can be deployed directly using inference, please refer to [inference deployment](../inference_deployment/).

You can also use PaddleLite's opt tool to convert the inference model to a mobile model for its mobile deployment. Please refer to [Mobile Model Deployment](../inference_deployment/paddle_lite_deploy_en.md) for more details.

<a name="5"></a>

## 5. Hyperparameter Training

- For quantization and pruning training, it is recommended to load the pre-trained model obtained from conventional training to accelerate the convergence of quantization training.
- For quantization training, it is recommended to modify the initial learning rate to `1/20~1/10` of the conventional training and the number of training epochs to `1/5~1/2`, while adding Warmup to the learning rate strategy. Please make no other modifications to the configuration information.
- For pruning training, the hyperparameter configuration is recommended to remain the same as the regular training.
