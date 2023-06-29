
# Knowledge Distillation Practice

## Contents

- [1. Introduction](#1)
    - [1.1 Introduction to Knowledge Distillation](#1.1)
        - [1.1.1 Response based distillation](#1.1.1)
        - [1.1.2 Feature based distillation](#1.1.2)
        - [1.1.3 Relation based distillation](#1.1.3)
    - [1.2 Knowledge Distillation Algorithms Supported by PaddleClas](#1.2)
        - [1.2.1 SSLD](#1.2.1)
        - [1.2.2 DML](#1.2.2)
        - [1.2.3 UDML](#1.2.3)
        - [1.2.4 AFD](#1.2.4)
        - [1.2.5 DKD](#1.2.5)
        - [1.2.6 DIST](#1.2.6)
        - [1.2.7 MGD](#1.2.7)
        - [1.2.8 WSL](#1.2.8)
        - [1.2.9 SKD](#1.2.9)
        - [1.2.10 PEFD](#1.2.10)
- [2. Usage](#2)
    - [2.1 Environment Configuration](#2.1)
    - [2.2 Data Preparation](#2.2)
    - [2.3 Model Training](#2.3)
    - [2.4 Model Evaluation](#2.4)
    - [2.5 Model Prediction](#2.5)
    - [2.6 Model Export & Inference](#2.6)
- [3. References](#3)

<a name="1"></a>

## 1. Introduction

<a name="1.1"></a>

### 1.1 Introduction to Knowledge Distillation

In recent years, deep neural network has been proved to be an effective method to solve problems in computer vision, natural language processing and other fields. By constructing an appropriate neural network and training it, the model performance will basically exceed traditional algorithms.

When data is sufficient, the model performance can be significantly improved by increasing the number of parameters through appropriate construction of the network, but this increases the model complexity. Large models are expensive to deploy in actual scenarios.

Redundancy exists in deep neural networks. At present, there are several methods to compress the model to reduce its parameter amount, e.g. pruning, quantization, knowledge distillation, etc. Knowledge distillation refers to the method that helps the training process of a smaller network (student) under the supervision of a larger network (teacher), so as to ensure the small model can obtain relatively large performance improvement and even obtain accuracy similar to large models without increasing parameters.

Knowledge distillation methods can be divided into three different categories: Response based distillation, Feature based distillation, Relation based distillation. Detailed introduction is as follows.

<a name='1.1.1'></a>

#### 1.1.1 Response based distillation

Knowledge distillation (KD) was first proposed by Hinton, who introduced KL divergence to the training loss function in addition to the cross entropy between the output logits and the ground truth labels. The accuracy of models trained with KD exceeds the accuracy of the same models trained only using ground truth loss. It should be noted that a larger teacher model needs to be trained first to guide the training process of student models.

PaddleClas proposed a simple yet effective SSLD algorithm [6], removing the dependence on ground truth labels. Combined with a large number of unlabeled data, the accuracy of pretrained models obtained from distillation on 18 models was improved by 3+%.

The aforementioned standard distillation method uses large models as teacher models to guide students to improve the performance. Later, Deep Mutual Learning (DML) distillation method [7] was proposed, i.e., two models with the same architecture learn from each other. Compared with KD and other knowledge distillation algorithms that rely on large teacher models, DML is independent of large teacher models. Such training process is simpler and more efficient.

<a name='1.1.2'></a>

#### 1.1.2 Feature based distillation

Heo et al porposed OverHaul of Feature Distillation [8], in which feature map distance between the teacher and the student is calculated as distillation loss. Features of the student are transformed to match the shape of the teacher's features so that the distance can be computed.

Knowledge distillation methods based on feature map distance can be combined with response-based knowledge distillation algorithms mentioned in `3.1`, i.e., the student's outputs and its middle feature maps are supervised simultaneously. For DML, such combination is even simpler, since the student's features can be aligned with the teacher's features without transformation. This method is used in PP-OCRv2, improving the accuracy of OCR models significantly.

<a name='1.1.3'></a>

#### 1.1.3 Relation based distillation

Papers in [1.1.1](#1.1.1) and [1.1.2](#1.1.2) mainly consider the outputs and middle feature maps of student and teacher. These distillation algorithms focus on individual outputs and do not consider relations between individuals.

Park et al proposed RKD [10], a distillation algorithm based on relations. In RKD, mutual relations of data examples is considered, and two loss functions are used, the distance-wise distillation loss and angle-wise distillation loss.


The algorithm proposed in this paper, Relational Knowledge Distillation (RKD), transfers the structured relations between the output results obtained from the teacher model to the student model. Unlike the previous algorithms, which only focus on individual output results, the RKD algorithm uses two loss functions: distance-wise distillation loss and angle-wise distillation loss. In the final distillation loss function, both KD loss and RKD loss are considered. The final accuracy is better than that obtained by KD loss distillation only.

<a name='1.2'></a>

### 1.2 Knowledge Distillation Algorithms in PaddleClas

<a name='1.2.1'></a>

#### 1.2.1 SSLD

##### 1.2.1.1 Introduction to SSLD

Paper:

> [Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones](https://arxiv.org/abs/2103.05959)
>
> Cheng Cui, Ruoyu Guo, Yuning Du, Dongliang He, Fu Li, Zewu Wu, Qiwen Liu, Shilei Wen, Jizhou Huang, Xiaoguang Hu, Dianhai Yu, Errui Ding, Yanjun Ma
>
> arxiv, 2021

SSLD is a simple semi-supervised distillation method proposed by Baidu in 2021. By designing an improved JS divergence as the loss function and combining the data mining strategy based on ImageNet22k dataset, the accuracy of the 18 backbone network models was improved by more than 3% on average.

<!-- For more information about the principle, model zoo and usage of SSLD, please refer to: [Introduction to SSLD](ssld_en.md). -->


##### 1.2.1.2 Configuration of SSLD

The SSLD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationDMLLoss` as the training loss.

```yaml
# model architecture
Arch:
  name: "DistillationModel"    # model name, here distillation model is used
  class_num: &class_num 1000   # number of classes, 1000 for ImageNet1k
  pretrained_list:             # list of pretrained models, leave blank because specified in sub-models
  freeze_params_list:          # list of freezed params, networks of correspondent index are fixed if set to True
  - True
  - False
  infer_model_name: "Student"  # export Student sub-network when exporting model
  models:                      # list of sub-networks
    - Teacher:                 # teacher model
        name: ResNet50_vd      # model name
        class_num: *class_num  # number of classes
        pretrained: True       # pretrained model path, download official pretrained model if set to True
        use_ssld: True         # whether SSLD pretrained model is used (higher accuracy)
    - Student:                 # student model
        name: PPLCNet_x2_5     # model name
        class_num: *class_num  # number of classes
        pretrained: False      # pretrained model path, can be bool or string. Set to False here. Student model does not load the pretrained model by default

# loss function config for traing/eval process
Loss:                           # loss function
  Train:                        # list of training losses
    - DistillationDMLLoss:      # distillation DMLLoss. DMLLoss is encapsulated to support loss function of distillation (in dict)
        weight: 1.0             # weight of loss
        model_name_pairs:       # model pair used to compute loss. Here loss function between Student and Teacher is computed
        - ["Student", "Teacher"]
  Eval:                         # evaluation loss
    - CELoss:
        weight: 1.0
```

<a name='1.2.2'></a>

#### 1.2.2 DML

##### 1.2.2.1 Introduction to DML

Paper:

> [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html)
>
> Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu
>
> CVPR, 2018

In the DML paper, the process of distillation does not depend on a teacher model. Two models with the same architecture learn from each other and calculate the KL divergence of each other's logits.


Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | PPLCNet_x2_5 | [PPLCNet_x2_5.yaml](../../../ppcls/configs/ImageNet/PPLCNet/PPLCNet_x2_5.yaml) | 74.93% | - |
| DML | PPLCNet_x2_5 | [PPLCNet_x2_5_dml.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_dml.yaml) | 76.68%(**+1.75%**) | - |


* Note: Complete PPLCNet_x2_5 The model have been trained for 360 epochs. For comparison, both baseline and DML have been trained for 100 epochs. Therefore, the accuracy is lower than the model (76.60%) opened on the official website.


##### 1.2.2.2 Configuration of DML

The DML configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. Both models need to be updated. In the `Loss` field, you need to define `DistillationDMLLoss` (JS-Div between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.

```yaml
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  pretrained_list:
  freeze_params_list:        # mutual learning, params of both models are not freezed
  - False
  - False
  models:
    - Teacher:
        name: PPLCNet_x2_5   # mutual learning, so pretrained models are not loaded for both models
        class_num: *class_num
        pretrained: False
    - Student:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False

Loss:
  Train:
    - DistillationGTCELoss:    # CE loss with ground truth labels needs to be computed for both models because pretrained models are not loaded
        weight: 1.0
        model_names: ["Student", "Teacher"]
    - DistillationDMLLoss:
        weight: 1.0
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.3'></a>

#### 1.2.3 UDML

##### 1.2.3.1 Introduction to UDML


UDML is a teacher-free knowledge distillation algorithm proposed by PaddleCV group. It is improved based on DML. In addition to the outputs, it also considers the middle layers features in the distillation process, so as to further improve the accuracy of knowledge distillation. For more information about UDML and its application, please refer to: [PP-ShiTu paper](https://arxiv.org/abs/2111.00775) and [PP-OCRv3 paper](https://arxiv.org/abs/2109.03144).



Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | PPLCNet_x2_5 | [PPLCNet_x2_5.yaml](../../../ppcls/configs/ImageNet/PPLCNet/PPLCNet_x2_5.yaml) | 74.93% | - |
| UDML | PPLCNet_x2_5 | [PPLCNet_x2_5_dml.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_udml.yaml) | 76.74%(**+1.81%**) | - |


##### 1.2.3.2 Configuration of UDML


```yaml
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - False
  - False
  models:
    - Teacher:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False
        # return_patterns means that in addition to the output logits, the middle feature maps with the corresponding names will also be returned
        return_patterns: ["blocks3", "blocks4", "blocks5", "blocks6"]
    - Student:
        name: PPLCNet_x2_5
        class_num: *class_num
        pretrained: False
        return_patterns: ["blocks3", "blocks4", "blocks5", "blocks6"]

# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
       weight: 1.0
       key: logits
       model_names: ["Student", "Teacher"]
    - DistillationDMLLoss:
        weight: 1.0
        key: logits
        model_name_pairs:
        - ["Student", "Teacher"]
    - DistillationDistanceLoss:  # distance loss based on features. Here l2 loss is used to calculate the distance between block5s
        weight: 1.0
        key: "blocks5"
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - CELoss:
        weight: 1.0
```

**Note(:** `return_patterns` are specified in the network above. The function of returning middle layer features is based on TheseusLayer.

<!-- TODO(gaotingquan) -->
<!-- For more information about usage of TheseusLayer, please refer to: [Usage of TheseusLayer](theseus_layer.md). -->


<a name='1.2.4'></a>

#### 1.2.4 AFD

##### 1.2.4.1 Introduction to AFD

Paper:


> [Show, attend and distill: Knowledge distillation via attention-based feature matching](https://arxiv.org/abs/2102.02973)
>
> Mingi Ji, Byeongho Heo, Sungrae Park
>
> AAAI, 2018

AFD proposes to use attention based meta network to learn the relative similarity between features in the distillation process, and apply the identified similarity relation to control the distillation intensity of all possible feature maps pairs.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| AFD | ResNet18 | [resnet34_distill_resnet18_afd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_afd.yaml) | 71.68%(**+0.88%**) | - |

Note: In order to keep alignment with the training configuration in the paper, the number of training iterations is set to be 100 epochs, so the baseline accuracy is lower than the open source model accuracy in PaddleClas (71.0%).

##### 1.2.4.2 Configuration of AFD

The AFD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters. In the `Loss` field, you need to define `DistillationKLDivLoss` (KL-Div between student and teacher), `AFDLoss` (AFD loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.

```yaml
Arch:
  name: "DistillationModel"
  pretrained_list:
  freeze_params_list:
  models:
    - Teacher:
        name: AttentionModel # contains several serial networks. The following networks take outputs of previous networks as input
        pretrained_list:
        freeze_params_list:
          - True
          - False
        models:
          # basic network of AttentionModel
          - ResNet34:
              name: ResNet34
              pretrained: True
              # return_patterns means that in addition to the output logits, the middle feature maps with the corresponding names will also be returned
              return_patterns: &t_keys ["blocks[0]", "blocks[1]", "blocks[2]", "blocks[3]",
                                        "blocks[4]", "blocks[5]", "blocks[6]", "blocks[7]",
                                        "blocks[8]", "blocks[9]", "blocks[10]", "blocks[11]",
                                        "blocks[12]", "blocks[13]", "blocks[14]", "blocks[15]"]
          # transformation network of AttentionModel. It transforms the sub-networks in the basic network
          - LinearTransformTeacher:
              name: LinearTransformTeacher
              qk_dim: 128
              keys: *t_keys
              t_shapes: &t_shapes [[64, 56, 56], [64, 56, 56], [64, 56, 56], [128, 28, 28],
                                   [128, 28, 28], [128, 28, 28], [128, 28, 28], [256, 14, 14],
                                   [256, 14, 14], [256, 14, 14], [256, 14, 14], [256, 14, 14],
                                   [256, 14, 14], [512, 7, 7], [512, 7, 7], [512, 7, 7]]

    - Student:
        name: AttentionModel
        pretrained_list:
        freeze_params_list:
          - False
          - False
        models:
          - ResNet18:
              name: ResNet18
              pretrained: False
              return_patterns: &s_keys ["blocks[0]", "blocks[1]", "blocks[2]", "blocks[3]",
                                        "blocks[4]", "blocks[5]", "blocks[6]", "blocks[7]"]
          - LinearTransformStudent:
              name: LinearTransformStudent
              qk_dim: 128
              keys: *s_keys
              s_shapes: &s_shapes [[64, 56, 56], [64, 56, 56], [128, 28, 28], [128, 28, 28],
                                   [256, 14, 14], [256, 14, 14], [512, 7, 7], [512, 7, 7]]
              t_shapes: *t_shapes

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
        key: logits
    - DistillationKLDivLoss:  # distillation KL-Div loss, features are extracted according to names in model_name_pairs to calculate loss
        weight: 0.9           # weight of loss
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 4
        key: logits
    - AFDLoss:                # AFD loss
        weight: 50.0
        model_name_pair: ["Student", "Teacher"]
        student_keys: ["bilinear_key", "value"]
        teacher_keys: ["query", "value"]
        s_shapes: *s_shapes
        t_shapes: *t_shapes
  Eval:
    - CELoss:
        weight: 1.0
```

**Note(:** `return_patterns` are specified in the network above. The function of returning middle layer features is based on TheseusLayer.

<!-- TODO(gaotingquan) -->
<!-- For more information about usage of TheseusLayer, please refer to: [Usage of TheseusLayer](theseus_layer.md). -->

<a name='1.2.5'></a>

#### 1.2.5 DKD

##### 1.2.5.1 Introduction to DKD

Paper:


> [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
>
> Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang
>
> CVPR, 2022

DKD reformulates the classical KD loss into two parts, i.e., target class knowledge distillation (TCKD) and non-target class knowledge distillation (NCKD). The effect of the two parts is studied separately, and their weights can be adjusted independently, improving the accuracy and flexibility of distillation.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| DKD | ResNet18 | [resnet34_distill_resnet18_dkd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_dkd.yaml) | 72.59%(**+1.79%**) | - |


##### 1.2.5.2 Configuration of DKD

The DKD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationDKDLoss` (DKD loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.


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
        name: ResNet34
        pretrained: True

    - Student:
        name: ResNet18
        pretrained: False

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationDKDLoss:
        weight: 1.0
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 1
        alpha: 1.0
        beta: 1.0
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.6'></a>

#### 1.2.6 DIST

##### 1.2.6.1 Introduction to DIST

Paper:


> [Knowledge Distillation from A Stronger Teacher](https://arxiv.org/pdf/2205.10536v1.pdf)
>
> Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu
>
> 2022, under review

When using the KD method for distillation, as the accuracy of the teacher model is improved, the effect of distillation is often difficult to improve simultaneously. This paper proposes the DIST method, which uses the Pearson correlation coefficient to represent the difference between the student model and the teacher model, instead of the default KL-divergence in the distillation process, so as to ensure that the model can learn more accurate correlation information.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| DIST | ResNet18 | [resnet34_distill_resnet18_dist.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_dist.yaml) | 71.99%(**+1.19%**) | - |


##### 1.2.6.2 Configuration of DIST

The DIST configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationDISTLoss` (DIST loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.


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
        name: ResNet34
        pretrained: True

    - Student:
        name: ResNet18
        pretrained: False

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationDISTLoss:
        weight: 2.0
        model_name_pairs:
        - ["Student", "Teacher"]
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.7'></a>

#### 1.2.7 MGD

##### 1.2.7.1 Introduction to MGD

Paper:


> [Masked Generative Distillation](https://arxiv.org/abs/2205.01529)
>
> Zhendong Yang, Zhe Li, Mingqi Shao, Dachuan Shi, Zehuan Yuan, Chun Yuan
>
> ECCV 2022

This method performs distillation on the feature map. In the process of distillation, random masks are applied to the features, and the students are forced to use some features to generate all the features of the teacher model, so as to improve the representation ability of the student model. MGD achieve state-of-the-art performance on the feature distillation task, and has been widely verified to be effective in tasks such as detection and segmentation.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| MGD | ResNet18 | [resnet34_distill_resnet18_mgd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_mgd.yaml) | 71.86%(**+1.06%**) | - |


##### 1.2.7.2 Configuration of MGD

The MGD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationPairLoss` (MGD loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.

```yaml
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - True
  - False
  infer_model_name: "Student"
  models:
    - Teacher:
        name: ResNet34
        class_num: *class_num
        pretrained: True
        return_patterns: &t_stages ["blocks[2]", "blocks[6]", "blocks[12]", "blocks[15]"]
    - Student:
        name: ResNet18
        class_num: *class_num
        pretrained: False
        return_patterns: &s_stages ["blocks[1]", "blocks[3]", "blocks[5]", "blocks[7]"]

# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationPairLoss:
        weight: 1.0
        model_name_pairs: [["Student", "Teacher"]] # calculate mgdloss for Student and Teacher
        name: "loss_mgd"
        base_loss_name: MGDLoss # MGD loss, the following are parameters of 'MGD loss'
        s_key: "blocks[7]"   # feature map used to calculate MGD loss in student model
        t_key: "blocks[15]"  # feature map used to calculate MGD loss in teacher model
        student_channels: 512   # channel num for stduent feature map
        teacher_channels: 512   # channel num for teacher feature map
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.8'></a>

#### 1.2.8 WSL

##### 1.2.8.1 Introduction to WSL

Paper:


> [Rethinking Soft Labels For Knowledge Distillation: A Bias-variance Tradeoff Perspective](https://arxiv.org/abs/2102.0650)
>
> Helong Zhou, Liangchen Song, Jiajie Chen, Ye Zhou, Guoli Wang, Junsong Yuan, Qian Zhang
>
> ICLR, 2021

Weighted Soft Labels (WSL) loss function assigns weights to the KD Loss of each sample according to the CE Loss ratio of the teacher model and the student model with respect to the ground-truth labels. If the student model predicts a certain sample better than the teacher model, a smaller weight will be assigned to the sample. The method is simple and effective. It enables the weight of each sample to be adaptively adjusted, thereby improving the distillation accuracy.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| WSL | ResNet18 | [resnet34_distill_resnet18_wsl.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_wsl.yaml) | 72.23%(**+1.43%**) | - |


##### 1.2.8.2 Configuration of WSL

The WSL configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationWSLLoss` (WSL loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.


```yaml
# model architecture
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
        name: ResNet34
        pretrained: True

    - Student:
        name: ResNet18
        pretrained: False

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationWSLLoss:
        weight: 2.5
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 2
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.9'></a>

#### 1.2.9 SKD

##### 1.2.9.1 Introduction to SKD

Paper:


> [Reducing the Teacher-Student Gap via Spherical Knowledge Disitllation](https://arxiv.org/abs/2010.07485)
>
> Jia Guo, Minghao Chen, Yao Hu, Chen Zhu, Xiaofei He, Deng Cai
>
> 2022, under review

Due to the limited capacity of the student, student performance would unexpectedly drop when distilling from an oversized teacher. Spherical Knowledge Distillation (SKD) explicitly eliminates the gap of confidence between teacher and student, so as to ease the capacity gap problem. SKD achieves a significant improvement over previous SOTA in distilling ResNet18 on ImageNet1k.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| SKD | ResNet18 | [resnet34_distill_resnet18_skd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_skd.yaml) | 72.84%(**+2.04%**) | - |


##### 1.2.9.2 Configuration of SKD

The SKD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationSKDLoss` (SKD loss between student and teacher). It should be noted that SKD loss includes KL div loss with teacher and CE loss with ground truth labels. Therefore, `DistillationGTCELoss` does not need to be defined.


```yaml
# model architecture
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
        name: ResNet34
        pretrained: True

    - Student:
        name: ResNet18
        pretrained: False

  infer_model_name: "Student"


# loss function config for traing/eval process
Loss:
  Train:
    - DistillationSKDLoss:
        weight: 1.0
        model_name_pairs: [["Student", "Teacher"]]
        temperature: 1.0
        multiplier: 2.0
        alpha: 0.9
  Eval:
    - CELoss:
        weight: 1.0
```

<a name='1.2.10'></a>

#### 1.2.10 PEFD

##### 1.2.10.1 Introduction to PEFD

Paper:


> [Improved Feature Distillation via Projector Ensemble](https://arxiv.org/pdf/2210.15274.pdf)
>
> Yudong Chen, Sen Wang, Jiajun Liu, Xuwei Xu, Frank de Hoog, Zi Huang
>
> NeurIPS 2022

PEFD uses an ensemble of multiple projectors to transform student's features before applying the feature distillation loss, so as to prevent the student from overfitting the teacher's features and further improve the performance of feature distillation.

Performance on ImageNet1k is shown below.

| Strategy | Backbone | Config | Top-1 acc | Download Link |
| --- | --- | --- | --- | --- |
| baseline | ResNet18 | [ResNet18.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet18.yaml) | 70.8% | - |
| PEFD | ResNet18 | [resnet34_distill_resnet18_pefd.yaml](../../../ppcls/configs/ImageNet/Distillation/resnet34_distill_resnet18_pefd.yaml) | 72.23%(**+1.43%**) | - |


##### 1.2.10.2 Configuration of PEFD

The PEFD configuration is shown below. In the `Arch` field, you need to define both the student model and the teacher model. The teacher model has fixed parameters, and the pretrained parameters are loaded. In the `Loss` field, you need to define `DistillationPairLoss` (PEFD loss between student and teacher) and `DistillationGTCELoss` (CE loss with ground truth labels) as the training loss.


```yaml
# model architecture
Arch:
  name: "DistillationModel"
  class_num: &class_num 1000
  # if not null, its lengths should be same as models
  pretrained_list:
  # if not null, its lengths should be same as models
  freeze_params_list:
  - True
  - False
  infer_model_name: "Student"
  models:
    - Teacher:
        name: ResNet34
        class_num: *class_num
        pretrained: True
        return_patterns: &t_stages ["avg_pool"]
    - Student:
        name: ResNet18
        class_num: *class_num
        pretrained: False
        return_patterns: &s_stages ["avg_pool"]

# loss function config for traing/eval process
Loss:
  Train:
    - DistillationGTCELoss:
        weight: 1.0
        model_names: ["Student"]
    - DistillationPairLoss:
        weight: 25.0
        base_loss_name: PEFDLoss
        model_name_pairs: [["Student", "Teacher"]]
        s_key: "avg_pool"
        t_key: "avg_pool"
        name: "loss_pefd"
        student_channel: 512
        teacher_channel: 512
  Eval:
    - CELoss:
        weight: 1.0
```

<a name="2"></a>

## 2. Training, Evaluation and Prediction

<a name="2.1"></a>  

### 2.1 Environment Configuration

* Installation: Please refer to [Installation Tutorial](../installation.md) to configure the running environment.

<a name="2.2"></a>

### 2.2 Data Preparation

Please prepare the ImageNet-1k dataset on [ImageNet website](https://www.image-net.org/).


Enter PaddleClas directory.

```
cd path_to_PaddleClas
```

Enter `dataset/` directory, name the downloaded data `ILSVRC2012` and store it here. The `ILSVRC2012` directory contains the following data:

```
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
├── train_list.txt
...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
├── val_list.txt
```

where `train/` and `val/` are training set and validation set, respectively. `train_list.txt` and `val_list.txt` are label files for training set and validation set, respectively.


If unlabeled data similar to the training set scenario is included, they can also be organized in the same way as the training set labels. Place the file in the same directory as the currently labeled dataset, and mark its label value as 0. Suppose the organized tag file is named `train_list_unlabel.txt`, you can use the following command to generate a label file for SSLD training.

```shell
cat train_list.txt train_list_unlabel.txt > train_list_all.txt
```


**Note:**

* For more information about the format of `train_list.txt` and `val_list.txt`, you may refer to [Format Description of PaddleClas Classification Dataset](../data_preparation/classification_dataset_en.md#1dataset-format) .


<a name="2.3"></a>

### 2.3 Model Training


In this section, the process of model training, evaluation and prediction of knowledge distillation algorithm will be introduced using the SSLD knowledge distillation algorithm as an example. The configuration file is [PPLCNet_x2_5_ssld.yaml](../../../ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml). You can use the following command to complete the model training.


```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml
```

<a name="2.4"></a>

### 2.4 Model Evaluation

After training the model, the following command can be used to evaluate the performance of the model:

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model
```

where `-o Global.pretrained_model="output/DistillationModel/best_model"` specifies the path of the current optimal weights. If you need to specify other weights, you can simply replace the path.

<a name="2.5"></a>

### 2.5 Model Prediction

After training is completed, the trained model can be loaded for prediction. A complete example is provided in `tools/infer.py`. You can use the model for prediction by executing the following command:

```python
python3 tools/infer.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=output/DistillationModel/best_model
```

The outputs are as follows:

```
[{'class_ids': [8, 7, 86, 82, 21], 'scores': [0.87908, 0.12091, 0.0, 0.0, 0.0], 'file_name': 'docs/images/inference_deployment/whl_demo.jpg', 'label_names': ['hen', 'cock', 'partridge', 'ruffed grouse, partridge, Bonasa umbellus', 'kite']}]
```


**Note:**

* Here `-o Global.pretrained_model="output/ResNet50/best_model"` specifies the path of the current optimal weights. If you need to specify other weights, you can simply replace the path.

* Image `docs/images/inference_deployment/whl_demo.jpg` is predicted by default. You can also predict other images by adding a field `-o Infer.infer_imgs=xxx`.


<a name="2.6"></a>

### 2.6 Model Export & Inference


PaddleInference is a native inference library for PaddlePaddle, which can be used on servers and clouds to provide high-performance inference. PaddleInference can use MKLDNN, CUDNN, and TensorRT to accelerate model inference, thereby achieving better performance compared with inference based directly on the trained model. For more information about PaddleInference, please refer to [Paddle Inference Tutorial](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html).

The model needs to be exported before inference. For models obtained from knowledge distillation, `-o Global.infer_model_name=Student` should be specified when exporting to indicate the model to be exported is the student model. The complete command is shown below.

```shell
python3 tools/export_model.py \
    -c ppcls/configs/ImageNet/Distillation/PPLCNet_x2_5_ssld.yaml \
    -o Global.pretrained_model=./output/DistillationModel/best_model \
    -o Arch.infer_model_name=Student
```

3 files will be generated in `inference` directory: `inference.pdiparams`, `inference.pdiparams.info` and `inference.pdmodel`.

For more information about model inference, please refer to: [Python Inference](../inference_deployment/python_deploy_en.md).


<a name="3"></a>

## 3. References

[1] Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[2] Bagherinezhad H, Horton M, Rastegari M, et al. Label refinery: Improving imagenet classification through label progression[J]. arXiv preprint arXiv:1805.02641, 2018.

[3] Yalniz I Z, Jégou H, Chen K, et al. Billion-scale semi-supervised learning for image classification[J]. arXiv preprint arXiv:1905.00546, 2019.

[4] Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation strategies from data[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2019: 113-123.

[5] Touvron H, Vedaldi A, Douze M, et al. Fixing the train-test resolution discrepancy[C]//Advances in Neural Information Processing Systems. 2019: 8250-8260.

[6] Cui C, Guo R, Du Y, et al. Beyond Self-Supervision: A Simple Yet Effective Network Distillation Alternative to Improve Backbones[J]. arXiv preprint arXiv:2103.05959, 2021.

[7] Zhang Y, Xiang T, Hospedales T M, et al. Deep mutual learning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4320-4328.

[8] Heo B, Kim J, Yun S, et al. A comprehensive overhaul of feature distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 1921-1930.

[9] Du Y, Li C, Guo R, et al. PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System[J]. arXiv preprint arXiv:2109.03144, 2021.

[10] Park W, Kim D, Lu Y, et al. Relational knowledge distillation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3967-3976.

[11] Zhao B, Cui Q, Song R, et al. Decoupled Knowledge Distillation[J]. arXiv preprint arXiv:2203.08679, 2022.

[12] Ji M, Heo B, Park S. Show, attend and distill: Knowledge distillation via attention-based feature matching[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(9): 7945-7952.

[13] Huang T, You S, Wang F, et al. Knowledge Distillation from A Stronger Teacher[J]. arXiv preprint arXiv:2205.10536, 2022.
