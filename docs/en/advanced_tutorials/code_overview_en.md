# Code Overview

## Catalogue

- [1. Overview of Code and Content](#1)
- [2. Training Module](#2)
  - [2.1 Data](#2.1)
  - [2.2 Model Structure](#2.2)
  - [2.3 Loss Function](#2.3)
  - [2.4 Optimizer, Learning Rate Decay, and Weight Decay](#2.4)
  - [2.5 Evaluation During Training](#2.5)
  - [2.6 Model Saving](#2.6)
  - [2.7 Model Pruning and Quantification](#2.7)
- [3. Codes and Methods for Inference and Deployment](#3)


<a name="1"></a>
## 1. Overview of Code and Content

The main code and content structure of PaddleClas are as follows:

- benchmark: shell scripts to test the speed metrics of different models in PaddleClas, such as single-card training speed metrics, multi-card training speed metrics, etc.
- dataset: datasets and the scripts used to process datasets. The scripts are responsible for processing the dataset into a suitable format for Dataloader.
- deploy: code for deployment, including deployment tools, which support python/cpp inference, Hub Serveing, Paddle Lite, Slim offline quantification and other deployment methods.
- ppcls: code for training and evaluation which is the main body of the PaddleClas framework. It also contains configuration files, and specific code of model training, evaluation, inference, dynamic to static export, etc.
- tools: entry functions and scripts for training, evaluation, inference, and dynamic to static export.
- The requirements.txt file is adopted to install the dependencies for PaddleClas. Use pip for upgrading, installation, and application.
- test_tipc: TIPC tests of PaddleClas models from training to prediction to verify that whether each function works properly.


<a name="2"></a>
## 2. Training Module

Modules of training deep learning model mainly contains data, model structure, loss function,
strategies such as optimizer, learning rate decay, and weight decay strategy, etc., which are explained below.


<a name="2.1"></a>
### 2.1 Data

For supervised tasks, the training data generally contains the raw data and its annotation.
In a single-label-based image classification task, the raw data refers to the image data,
while the annotation is the class to which the image data belongs. In PaddleClas, a label file,
in the following format, is required for training,
with each row containing one training sample and separated by a separator (space by default),
representing the image path and the class label respectively.

```
train/n01440764/n01440764_10026.JPEG 0
train/n01440764/n01440764_10027.JPEG 0
```

`ppcls/data/dataloader/common_dataset.py` contains the `CommonDataset` class inherited from `paddle.io.Dataset`,
which is a dataset class that can index and fetch a given sample by a key value.
Dataset classes such as `ImageNetDataset`, `LogoDataset`, `CommonDataset`, etc. are all inherited from this class.

The raw image needs to be preprocessed before training.
The standard data preprocessing during training contains
`DecodeImage`, `RandCropImage`, `RandFlipImage`, `NormalizeImage`, and `ToCHWImage`.
The data preprocessing is mainly in the `transforms` field, which is presented in a list,
and then converts the data in order, as reflected in the configuration file below.

```yaml
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```

PaddleClas also contains `AutoAugment`, `RandAugment`, and other data augmentation methods,
which can also be configured in the configuration file and thus added to the data preprocessing of the training.
Each data augmentation and process method is implemented as a class for easy migration and reuse.
For more specific implementation of data processing, please refer to the code under `ppcls/data/preprocess/ops/`.

You can also use methods such as mixup or cutmix to augment the data that make up a batch.
PaddleClas integrates `MixupOperator`, `CutmixOperator`, `FmixOperator`, and other batch-based data augmentation methods,
which can be configured by deploying the mix parameter in the configuration file.
For code implementation, please refer to `ppcls/data/preprocess /batch_ops/batch_operators.py`.

In image classification, the data post-processing is mainly `argmax` operation, which is not elaborated here.


<a name="2.2"></a>
### 2.2 Model Structure

The model in the configuration file is structured as follows:

```yaml
Arch:
  name: ResNet50
  class_num: 1000
  pretrained: False
  use_ssld: False
```

`Arch.name`: the name of the model

`Arch.pretrained`: whether to add a pre-trained model

`Arch.use_ssld`: whether to use a pre-trained model based on `SSLD` knowledge distillation.

All model names are defined in `ppcls/arch/backbone/__init__.py`.

Correspondingly, the model object is created in `ppcls/arch/__init__.py` with the `build_model` method.

```python
def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, model_type)(**config)
    return arch
```


<a name="2.3"></a>
### 2.3 Loss Function

PaddleClas implement `CELoss` , `JSDivLoss`, `TripletLoss`, `CenterLoss` and other loss functions, all defined in `ppcls/loss`.

In the `ppcls/loss/__init__.py` file, `CombinedLoss` is used to construct and combine loss functions.
The loss functions and calculation methods required in different training strategies are disparate,
and the following factors are considered by PaddleClas in the construction of the loss function.

1. whether to use label smooth
2. whether to use mixup or cutmix
3. whether to use distillation method for training
4. whether to train metric learning

User can specify the type and weight of the loss function in the configuration file,
such as adding TripletLossV2 to the training, the configuration file is as follows:

```yaml
Loss:
  Train:
    - CELoss:
        weight: 1.0
    - TripletLossV2:
        weight: 1.0
        margin: 0.5
```


<a name="2.4"></a>
### 2.4 Optimizer, Learning Rate Decay, and Weight Decay

In image classification tasks, `Momentum` is a commonly used optimizer,
and several optimizer strategies such as `Momentum`, `RMSProp`, `Adam`, and `AdamW` are provided in PaddleClas.

The weight decay strategy is a common regularization method, mainly adopted to prevent model overfitting.
Two weight decay strategies, `L1Decay` and `L2Decay`, are provided in PaddleClas.

Learning rate decay is an essential training method for accuracy improvement in image classification tasks.
PaddleClas currently supports `Cosine`, `Piecewise`, `Linear`, and other learning rate decay strategies.

In the configuration file, the optimizer, weight decay,
and learning rate decay strategies can be configured with the following fields.

```yaml
Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.1
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0001
```

Employ `build_optimizer` in `ppcls/optimizer/__init__.py` to create the optimizer and learning rate objects.

```python
def build_optimizer(config, epochs, step_each_epoch, parameters):
    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)
    logger.debug("build lr ({}) success..".format(lr))
    # step2 build regularization
    if 'regularizer' in config and config['regularizer'] is not None:
        reg_config = config.pop('regularizer')
        reg_name = reg_config.pop('name') + 'Decay'
        reg = getattr(paddle.regularizer, reg_name)(**reg_config)
    else:
        reg = None
    logger.debug("build regularizer ({}) success..".format(reg))
    # step3 build optimizer
    optim_name = config.pop('name')
    if 'clip_norm' in config:
        clip_norm = config.pop('clip_norm')
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    else:
        grad_clip = None
    optim = getattr(optimizer, optim_name)(learning_rate=lr,
                                           weight_decay=reg,
                                           grad_clip=grad_clip,
                                           **config)(parameters=parameters)
    logger.debug("build optimizer ({}) success..".format(optim))
    return optim, lr
```

Different optimizers and weight decay strategies are implemented as classes,
which can be found in the file `ppcls/optimizer/optimizer.py`.
Different learning rate decay strategies can be found in the file `ppcls/optimizer/learning_rate.py`.


<a name="2.5"></a>
### 2.5 Evaluation During Training

When training the model, you can set the interval of model saving,
or you can evaluate the validation set every several epochs so that the model with the best accuracy can be saved.
Follow the examples below to configure.

```
Global:
  save_interval: 1 # epoch interval of model saving
  eval_during_train: True # whether evaluate during training
  eval_interval: 1 # epoch interval of evaluation
```


<a name="2.6"></a>
### 2.6 Model Saving

The model is saved through the `paddle.save()` function of the Paddle framework.
The dynamic graph version of the model is saved in the form of a dictionary to facilitate further training.
The specific implementation is as follows:

```python
def save_model(program, model_path, epoch_id, prefix='ppcls'):
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    logger.info(
        logger.coloring("Already save model in {}".format(model_path), "HEADER"))
```

When saving, there are two things to keep in mind:

1. Only save the model on node 0, otherwise, if all nodes save models to the same path,
a file conflict may occur during multi-card training when multiple nodes write files,
preventing the final saved model from being loaded correctly.
2. Optimizer parameters also need to be saved to facilitate subsequent loading of breakpoints for training.


<a name="2.7"></a>
### 2.7 Model Pruning and Quantification

If you want to conduct compression training, please configure with the following fields.

1. Model pruning：

```yaml
Slim:
  prune:
    name: fpgm
    pruned_ratio: 0.3
```

2. Model quantification：

```yaml
Slim:
  quant:
    name: pact
```
For details of the training method, see [Pruning and Quantification Application](model_prune_quantization_en.md),
and the algorithm is described in [Pruning and Quantification algorithms](model_prune_quantization_en.md).


<a name="3"></a>
## 3. Codes and Methods for Inference and Deployment

- If you wish to quantify the classification model offline, please refer to
[Model Pruning and Quantification Tutorial](model_prune_quantization_en.md) for offline quantification.
- If you wish to use python for server-side deployment,
please refer to [Python Inference Tutorial](../inference_deployment/python_deploy_en.md).
- If you wish to use cpp for server-side deployment,
please refer to [Cpp Inference Tutorial](../inference_deployment/cpp_deploy_en.md).
- If you wish to deploy the classification model as a service,
please refer to the [Hub Serving Inference Deployment Tutorial](../inference_deployment/paddle_hub_serving_deploy_en.md).
- If you wish to use classification models for inference on mobile,
please refer to [PaddleLite Inference Deployment Tutorial](../inference_deployment/paddle_lite_deploy_en.md)
- If you wish to use the whl package for inference of classification models,
please refer to [whl Package Inference](../inference_deployment/whl_deploy_en.md) .
