# Configuration Instruction

------

## Introdction

The parameters in the PaddleClas configuration file(`ppcls/configs/*.yaml`)are described for you to customize or modify the hyperparameter configuration more quickly.

## Details

### Catalogue

- [1. Classification model](#1)
  - [1.1 Global Configuration](#1.1)
  - [1.2 Architecture](#1.2)
  - [1.3 Loss function](#1.3)
  - [1.4 Optimizer](#1.4)
  - [1.5 Data reading module(DataLoader)](#1.5)
      - [1.5.1 dataset](#1.5.1)
      - [1.5.2 sampler](#1.5.2)
      - [1.5.3 loader](#1.5.3)
  - [1.6 Evaluation metric](#1.6)
  - [1.7 Inference](#1.7)
- [2. Distillation model](#2)
  - [2.1 Architecture](#2.1)
  - [2.2 Loss function](#2.2)
  - [2.3 Evaluation metric](#2.3)
- [3. Recognition model](#3)
  - [3.1 Architechture](#3.1)
  - [3.2 Evaluation metric](#3.2)
  
  
<a name="1"></a>
### 1. Classification model

Here the configuration of `ResNet50_vd` on`ImageNet-1k`is used as an example to explain the each parameter in detail. [Configure Path](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml).

<a name="1.1"></a>
#### 1.1 Global Configuration

| Parameter name     | Specific meaning                                        | Defult value     | Optional value    |
| ------------------ | ------------------------------------------------------- | ---------------- | ----------------- |
| checkpoints        | Breakpoint model path for resuming training             | null             | str               |
| pretrained_model   | Pre-trained model path                                  | null             | str               |
| output_dir         | Save model path                                         | "./output/"      | str               |
| save_interval      | How many epochs to save the model at each interval      | 1                | int               |
| eval_during_train  | Whether to evaluate at training                         | True             | bool              |
| eval_interval      | How many epochs to evaluate at each interval            | 1                | int               |
| epochs             | Total number of epochs in training                      |                  | int               |
| print_batch_step   | How many mini-batches to print out at each interval     | 10               | int               |
| use_visualdl       | Whether to visualize the training process with visualdl | False            | bool              |
| image_shape        | Image size                                              | [3，224，224]    | list, shape: (3,) |
| save_inference_dir | Inference model save path                               | "./inference"    | str               |
| eval_mode          | Model of eval                                           | "classification" | "retrieval"       |

**Note**：The http address of pre-trained model can be filled in the `pretrained_model`

<a name="1.2"></a>
#### 1.2 Architecture

| Parameter name | Specific meaning  | Defult value | Optional value        |
| -------------- | ----------------- | ------------ | --------------------- |
| name           | Model Arch name   | ResNet50     | PaddleClas model arch |
| class_num      | Category number   | 1000         | int                   |
| pretrained     | Pre-trained model | False        | bool， str            |

**Note**: Here pretrained can be set to True or False, so does the path of the weights. In addition, the pretrained is disabled when Global.pretrained_model is also set to the corresponding path.

<a name="1.3"></a>
#### 1.3 Loss function

| Parameter name | Specific meaning                            | Defult value | Optional value         |
| -------------- | ------------------------------------------- | ------------ | ---------------------- |
| CELoss         | cross-entropy loss function                 | ——           | ——                     |
| CELoss.weight  | The weight of CELoss in the whole Loss      | 1.0          | float                  |
| CELoss.epsilon | The epsilon value of label_smooth in CELoss | 0.1          | float，between 0 and 1 |

<a name="1.4"></a>
#### 1.4 Optimizer

| Parameter name    | Specific meaning                 | Defult value | Optional value                                     |
| ----------------- | -------------------------------- | ------------ | -------------------------------------------------- |
| name              | optimizer method name            | "Momentum"   | Other optimizer including "RmsProp"                |
| momentum          | momentum value                   | 0.9          | float                                              |
| lr.name           | method of dropping learning rate | "Cosine"     | Other dropping methods of "Linear" and "Piecewise" |
| lr.learning_rate  | initial value of learning rate   | 0.1          | float                                              |
| lr.warmup_epoch   | warmup rounds                    | 0            | int，such as 5                                           |
| regularizer.name  | regularization method name       | "L2"         | ["L1", "L2"]                                       |
| regularizer.coeff | regularization factor            | 0.00007      | float                                              |

**Note**：The new parameters may be different when `lr.name`  is different , as when `lr.name=Piecewise`, the following parameters need to be added:

```
  lr:
    name: Piecewise
    learning_rate: 0.1
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
```

Referring to [learning_rate.py](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/optimizer/learning_rate.py) for adding method and parameters.

<a name="1.5"></a>
#### 1.5 Data reading module(DataLoader)

<a name="1.5.1"></a>
##### 1.5.1 dataset

| Parameter name      | Specific meaning                     | Defult value                        | Optional value                 |
| ------------------- | ------------------------------------ | ----------------------------------- | ------------------------------ |
| name                | The name of the class to read the data                   | ImageNetDataset                     | VeriWild and other Dataet type |
| image_root          | The path where the dataset is stored | ./dataset/ILSVRC2012/               | str                            |
| cls_label_path      | data label list                      | ./dataset/ILSVRC2012/train_list.txt | str                            |
| transform_ops       | data preprocessing for single images | ——                                  | ——                             |
| batch_transform_ops | Data preprocessing for batch images  | ——                                  | ——                             |

The parameter meaning of transform_ops:

| Function name  | Parameter name | Specific meaning      |
| -------------- | -------------- | --------------------- |
| DecodeImage    | to_rgb         | data to RGB           |
|                | channel_first  | image data by CHW     |
| RandCropImage  | size           | Random crop           |
| RandFlipImage  |                | Random flip           |
| NormalizeImage | scale          | Normalize scale value |
|                | mean           | Normalize mean value  |
|                | std            | normalized variance   |
|                | order          | Normalize order       |
| CropImage      | size           | crop size             |
| ResizeImage    | resize_short   | resize by short edge  |

The parameter meaning of batch_transform_ops:

| Function name | Parameter name | Specific meaning                        |
| ------------- | -------------- | --------------------------------------- |
| MixupOperator | alpha          | Mixup parameter value，the larger the value, the stronger the augment |

<a name="1.5.2"></a>
##### 1.5.2 sampler

| Parameter name | Specific meaning                                             | Default value           | Optional value                                     |
| -------------- | ------------------------------------------------------------ | ----------------------- | -------------------------------------------------- |
| name           | sampler type                                                 | DistributedBatchSampler | DistributedRandomIdentitySampler and other Sampler |
| batch_size     | batch size                                                   | 64                      | int                                                |
| drop_last      | Whether to drop the last data that does reach the batch-size | False                   | bool                                               |
| shuffle        | whether to shuffle the data                                  | True                    | bool                                               |
<a name="1.5.3"></a>
##### 1.5.3 loader

| Parameter name    | Specific meaning             | Default meaning | Optional meaning |
| ----------------- | ---------------------------- | --------------- | ---------------- |
| num_workers       | Number of data read threads  | 4               | int              |
| use_shared_memory | Whether to use shared memory | True            | bool             |

<a name="1.6"></a>
#### 1.6 Evaluation metric

| Parameter name | Specific meaning | Default meaning | Optional meaning |
| -------------- | ---------------- | --------------- | ---------------- |
| TopkAcc        | TopkAcc          | [1, 5]          | list, int        |

<a name="1.7"></a>
#### 1.7 Inference

| Parameter name                | Specific meaning                  | Default meaning                       | Optional meaning |
| ----------------------------- | --------------------------------- | ------------------------------------- | ---------------- |
| infer_imgs                    | Image address to be inferred      | docs/images/whl/demo.jpg              | str              |
| batch_size                    | batch size                        | 10                                    | int              |
| PostProcess.name              | Post-process name                 | Topk                                  | str              |
| PostProcess.topk              | topk value                        | 5                                     | int              |
| PostProcess.class_id_map_file | mapping file of class id and name | ppcls/utils/imagenet1k_label_list.txt | str              |

**Note**：The interpretation of `transforms` in the Infer module refers to the interpretation of`transform_ops`in the dataset in the data reading module.

<a name="2"></a>
### 2. Distillation model

**Note**：Here the training configuration of `MobileNetV3_large_x1_0` on `ImageNet-1k` distilled MobileNetV3_small_x1_0 is used as an example to explain the meaning of each parameter in detail. [Configure path](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/Distillation/mv3_large_x1_0_distill_mv3_small_x1_0.yaml). Only parameters that are distinct from the classification model are introduced here.

<a name="2.1"></a>
#### 2.1 Architecture

| Parameter name     | Specific meaning                                          | Default meaning        | Optional meaning                   |
| ------------------ | --------------------------------------------------------- | ---------------------- | ---------------------------------- |
| name               | model arch name                                           | DistillationModel      | ——                                 |
| class_num          | category number                                           | 1000                   | int                                |
| freeze_params_list | freeze_params_list                                        | [True, False]          | list                               |
| models             | model list                                                | [Teacher, Student]     | list                               |
| Teacher.name       | teacher model name                                        | MobileNetV3_large_x1_0 | PaddleClas model                   |
| Teacher.pretrained | teacher model pre-trained weights                         | True                   | Boolean or pre-trained weight path |
| Teacher.use_ssld   | whether teacher model pretrained weights are ssld weights | True                   | Boolean                            |
| infer_model_name   | type of the model being inferred                          | Student                | Teacher                            |

**Note**：

1. list is represented in yaml as follows:

```
  freeze_params_list:
  - True
  - False
```

2.Student's parameters are similar and will not be repeated.

<a name="2.2"></a>
#### 2.2  Loss function

| Parameter name                      | Specific meaning                                             | Default meaning | Optional meaning |
| ----------------------------------- | ------------------------------------------------------------ | --------------- | ---------------- |
| DistillationCELoss                  | Distillation's cross-entropy loss function                   | ——              | ——               |
| DistillationCELoss.weight           | Loss weight                                                  | 1.0             | float            |
| DistillationCELoss.model_name_pairs | ["Student", "Teacher"]                                       | ——              | ——               |
| DistillationGTCELoss.weight         | Distillation's cross-entropy loss function of model and true Label | ——              | ——               |
| DistillationGTCELos.weight          | Loss weight                                                  | 1.0             | float            |
| DistillationCELoss.model_names      | Model names with real label for cross-entropy                | ["Student"]     | ——               |

<a name="2.3"></a>
#### 2.3 Evaluation metric

| Parameter name                | Specific meaning    | Default meaning              | Optional meaning |
| ----------------------------- | ------------------- | ---------------------------- | ---------------- |
| DistillationTopkAcc           | DistillationTopkAcc | including model_key and topk | ——               |
| DistillationTopkAcc.model_key | the evaluated model | "Student"                    | "Teacher"        |
| DistillationTopkAcc.topk      | Topk value          | [1, 5]                       | list, int        |

**Note**： `DistillationTopkAcc` has the same meaning as `TopkAcc`, except that it is only used in distillation tasks.

<a name="3"></a>
### 3. Recognition model

**Note**：The training configuration of`ResNet50` on`LogoDet-3k` is used here as an example to explain the meaning of each parameter in detail. [configure path](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/Logo/ResNet50_ReID.yaml). Only parameters that are distinct from the classification model are presented here.

<a name="3.1"></a>
#### 3.1 Architechture

| Parameter name         | Specific meaning                                             | Default meaning             | Optional meaning                                             |
| ---------------------- | ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| name                   | Model arch                                                   | "RecModel"                  | ["RecModel"]                                                 |
| infer_output_key       | inference output value                                       | “feature”                   | ["feature", "logits"]                                        |
| infer_add_softmax      | softmaxwhether to add softmax to infercne                    | False                       | [True, False]                                                |
| Backbone.name          | Backbone name                                                | ResNet50_last_stage_stride1 | other backbone provided by PaddleClas                        |
| Backbone.pretrained    | Backbone pre-trained model                                   | True                        | Boolean value or pre-trained model path                      |
| BackboneStopLayer.name | The name of the output layer in Backbone                     | True                        | The`full_name`of the feature output layer in Backbone        |
| Neck.name              | The name of the Neck part                                    | VehicleNeck                 | the dictionary structure to be passed in, the specific input parameters for the Neck network layer |
| Neck.in_channels       | Input dimension size of the Neck part                        | 2048                        | the size is the same as BackboneStopLayer.name               |
| Neck.out_channels      | Output the dimension size of the Neck part, i.e. feature dimension size | 512                         | int                                                          |
| Head.name              | Network Head part nam                                        | CircleMargin                | Arcmargin. Etc                                               |
| Head.embedding_size    | Feature dimension size                                       | 512                         | Consistent with Neck.out_channels                            |
| Head.class_num         | number of classes                                            | 3000                        | int                                                          |
| Head.margin            | margin value in CircleMargin                                 | 0.35                        | float                                                        |
| Head.scale             | scale value in CircleMargin                                  | 64                          | int                                                          |

**Note**：

1.In PaddleClas, the `Neck` part is the connection part between Backbone and embedding layer, and `Head` part is the connection part between embedding layer and classification layer.。

2.`BackboneStopLayer.name` can be obtained by visualizing the model, visualization can be referred to [Netron](https://github.com/lutzroeder/netron) or [visualdl](https://github.com/PaddlePaddle/VisualDL).

3.Calling tools/export_model.py will convert the model weights to inference model, where the infer_add_softmax parameter will control whether to add the Softmax activation function afterwards, the code default is True (the last output layer in the classification task will be connected to the Softmax activation function). In the recognition task, the activation function is not required for the feature layer, so it should be set to False here.



<a name="3.2"></a>
#### 3.2 Evaluation metric

| Parameter name | Specific meaning            | Default meaning | Optional meaning |
| -------------- | --------------------------- | --------------- | ---------------- |
| Recallk        | Recall rate                 | [1, 5]          | list, int        |
| mAP            | Average retrieval precision | None            | None             |
