# Trial in 30mins(professional)

Here is a quick start tutorial for professional users to use PaddleClas on the Linux operating system. The main content is based on the CIFAR-100 data set. You can quickly experience the training of different models, experience loading different pre-trained models, experience the SSLD knowledge distillation solution, and experience data augmentation. Please refer to [Installation Guide](../installation/install_paddleclas_en.md) to configure the operating environment and clone PaddleClas code.

------

## Catalogue

- [1. Data and model preparation](#1)
  - [1.1 Data preparation](#1.1)
    - [1.1.1 Prepare CIFAR100](#1.1.1)
- [2. Model training](#2)
  - [2.1 Single label training](#2.1)
    - [2.1.1 Training without loading the pre-trained model](#2.1.1)
    - [2.1.2 Transfer learning](#2.1.2)
- [3. Data Augmentation](#3)
  - [3.1 Data augmentation-Mixup](#3.1)
- [4. Knowledge distillation](#4)
- [5. Model evaluation and inference](#5)
  - [5.1 Single-label classification model evaluation and inference](#5.1)
    - [5.1.1 Single-label classification model evaluation](#5.1.1)
    - [5.1.2 Single-label classification model prediction](#5.1.2)
    - [5.1.3 Single-label classification uses inference model for model inference](#5.1.3)

<a name="1"></a>

## 1. Data and model preparation

<a name="1.1"></a>

### 1.1 Data preparation


* Enter the PaddleClas directory.

```
cd path_to_PaddleClas
```

<a name="1.1.1"></a> 

#### 1.1.1 Prepare CIFAR100

* Enter the `dataset/` directory, download and unzip the CIFAR100 dataset.

```shell
cd dataset
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/CIFAR100.tar
tar -xf CIFAR100.tar
cd ../
```

<a name="2"></a>

## 2. Model training

<a name="2.1"></a> 

### 2.1 Single label training

<a name="2.1.1"></a> 

#### 2.1.1 Training without loading the pre-trained model

* Based on the ResNet50_vd model, the training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR"
```

The highest accuracy of the validation set is around 0.415.

Here, multiple GPUs are used for training. If only one GPU is used, please specify the GPU with the `CUDA_VISIBLE_DEVICES` setting, and specify the GPU with the `--gpus` setting, the same below. For example, to train with only GPU 0:

```shell
export CUDA_VISIBLE_DEVICES=0
python3 -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Optimizer.lr.learning_rate=0.01
```

* **Notice**:

* The GPUs specified in `--gpus` can be a subset of the GPUs specified in `CUDA_VISIBLE_DEVICES`.
* Since the initial learning rate and batch-size need to maintain a linear relationship, when training is switched from 4 GPUs to 1 GPU, the total batch-size is reduced to 1/4 of the original, and the learning rate also needs to be reduced to 1/4 of the original, so changed the default learning rate from 0.04 to 0.01.


<a name="2.1.2"></a> 


#### 2.1.2 Transfer learning

* Based on ImageNet1k classification pre-training model ResNet50_vd_pretrained (accuracy rate 79.12%) for fine-tuning, the training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

The highest accuracy of the validation set is about 0.718. After loading the pre-trained model, the accuracy of the CIFAR100 data set has been greatly improved, with an absolute accuracy increase of 30%.

* Based on ImageNet1k classification pre-training model ResNet50_vd_ssld_pretrained (accuracy rate of 82.39%) for fine-tuning, the training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True \
        -o Arch.use_ssld=True
```

In the final CIFAR100 verification set, the top-1 accuracy is 0.73. Compared with the fine-tuning of the pre-trained model with a top-1 accuracy of 79.12%, the top-1 accuracy of the new data set can be increased by 1.2% again.

* Replace the backbone with MobileNetV3_large_x1_0 for fine-tuning, the training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV3_large_x1_0_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR" \
        -o Arch.pretrained=True
```

The highest accuracy of the validation set is about 0.601, which is nearly 12% lower than ResNet50_vd.

<a name="3"></a>


## 3. Data Augmentation

PaddleClas contains many data augmentation methods, such as Mixup, Cutout, RandomErasing, etc. For specific methods, please refer to [Data augmentation chapter](../algorithm_introduction/DataAugmentation_en.md)。

<a name="3.1"></a> 

### 3.1 Data augmentation-Mixup

Based on the training method in [Data Augmentation Chapter](../algorithm_introduction/DataAugmentation_en.md) in Section 3.3, combined with Mixup's data augmentation method for training, the specific training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/ResNet50_vd_mixup_CIFAR100_finetune.yaml \
        -o Global.output_dir="output_CIFAR"

```


The final accuracy on the CIFAR100 verification set is 0.73, and the use of data augmentation can increase the model accuracy by about 1.2% again.


* **Note**

* For other data augmentation configuration files, please refer to the configuration files in `ppcls/configs/ImageNet/DataAugment/`.
* The number of epochs for training CIFAR100 is small, so the accuracy of the validation set may fluctuate by about 1%.

<a name="4"></a>


## 4. Knowledge distillation


PaddleClas includes a self-developed SSLD knowledge distillation scheme. For specific content, please refer to [Knowledge Distillation Chapter](../algorithm_introduction/knowledge_distillation_en.md). This section will try to use knowledge distillation technology to train the MobileNetV3_large_x1_0 model. Here we use the ResNet50_vd model trained in section 2.1.2 as the teacher model for distillation. First, save the ResNet50_vd model trained in section 2.1.2 to the specified directory. The script is as follows.

```shell
mkdir pretrained
cp -r output_CIFAR/ResNet50_vd/best_model.pdparams  ./pretrained/
```

The model name, teacher model and student model configuration, pre-training address configuration, and freeze_params configuration in the configuration file are as follows, where the two values in `freeze_params_list` represent whether the teacher model and the student model freeze parameter training respectively.

```yaml
Arch:
  name: "DistillationModel"
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - True
  - False
  models:
    - Teacher:
        name: ResNet50_vd
        pretrained: "./pretrained/best_model"
    - Student:
        name: MobileNetV3_large_x1_0
        pretrained: True
```

The loss configuration is as follows, where the training loss is the cross entropy of the output of the student model and the teacher model, and the validation loss is the cross entropy of the output of the student model and the true label.

```yaml
Loss:
  Train:
    - DistillationCELoss:
        weight: 1.0
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
```

The final training script is shown below.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/R50_vd_distill_MV3_large_x1_0_CIFAR100.yaml \
        -o Global.output_dir="output_CIFAR"

```


In the end, the accuracy on the CIFAR100 validation set was 64.4%. Using the teacher model for knowledge distillation, the accuracy of MobileNetV3 increased by 4.3%.

* **Note**

* In the distillation process, the pre-trained model used by the teacher model is the training result on the CIFAR100 dataset, and the student model uses the MobileNetV3_large_x1_0 pre-trained model with an accuracy of 75.32% on the ImageNet1k dataset.
  * The distillation process does not need to use real labels, so more unlabeled data can be used. In the process of use, you can generate fake `train_list.txt` from unlabeled data, and then merge it with the real `train_list.txt`, You can experience it yourself based on your own data.

<a name="5"></a>

## 5. Model evaluation and inference

<a name="5.1"></a> 

### 5.1 Single-label classification model evaluation and inference

<a name="5.1.1"></a> 

#### 5.1.1 Single-label classification model evaluation

After training the model, you can use the following commands to evaluate the accuracy of the model.

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model="output_CIFAR/ResNet50_vd/best_model"
```

<a name="5.1.2"></a> 

#### 5.1.2 Single-label classification model prediction

After the model training is completed, the pre-trained model obtained by the training can be loaded for model prediction. A complete example is provided in `tools/infer.py`, the model prediction can be completed by executing the following command:

```python
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Infer.infer_imgs=./dataset/CIFAR100/test/0/0001.png \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```

<a name="5.1.3"></a> 

#### 5.1.3 Single-label classification uses inference model for model inference

We need to export the inference model, PaddlePaddle supports the use of prediction engines for inference. Here, we will introduce how to use the prediction engine for inference:
First, export the trained model to inference model:

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/ResNet50_vd_CIFAR100.yaml \
    -o Global.pretrained_model=output_CIFAR/ResNet50_vd/best_model
```

* By default, `inference.pdiparams`, `inference.pdmodel` and `inference.pdiparams.info` files will be generated in the `inference` folder.

Use prediction engines for inference:

Enter the deploy directory:

```bash
cd deploy
```

Change the `inference_cls.yaml` file. Since the resolution used for training CIFAR100 is 32x32, the relevant resolution needs to be changed. The image preprocessing in the final configuration file is as follows:

```yaml
PreProcess:
  transform_ops:
    - ResizeImage:
        resize_short: 36
    - CropImage:
        size: 32
    - NormalizeImage:
        scale: 0.00392157
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
```

Execute the command to make predictions. Since the default `class_id_map_file` is the mapping file of the ImageNet dataset, you need to set None here.

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/CIFAR100/test/0/0001.png \
    -o PostProcess.Topk.class_id_map_file=None
```
