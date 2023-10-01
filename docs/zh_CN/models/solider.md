# Solider

-----
## 目录

- [1. 模型介绍](#1)
- [2. 精度、模型](#2)

<a name='1'></a>

## 1. 模型介绍

Solider是一个语义可控的自监督学习框架，可以从大量未标记的人体图像中学习一般的人类表征，从而最大限度地有利于下游以人类为中心的任务。与已有的自监督学习方法不同，该方法利用人体图像中的先验知识建立伪语义标签，并将更多的语义信息引入到学习的表示中。同时，不同的下游任务往往需要不同比例的语义信息和外观信息，单一的学习表示不能满足所有需求。为了解决这一问题，Solider引入了一种带有语义控制器的条件网络，可以满足下游任务的不同需求。[论文地址](https://arxiv.org/abs/2303.17602)。

<a name='2'></a>

## 2. 精度、模型

| Task                                              | Dataset     | Swin Tiny ([Link](链接：https://pan.baidu.com/s/1Buzo4fNt_HvDmTidUVRbuQ?pwd=yzys <br/>提取码：yzys)) | Swin Small ([Link](链接：https://pan.baidu.com/s/1d11UIQAu01zWQz-3Gv4n9g?pwd=ofbe <br/>提取码：ofbe)) | Swin Base ([Link](链接：https://pan.baidu.com/s/1KrRZRNaS3E8z7dXbXe9AJg?pwd=0z72 <br/>提取码：0z72)) |
| ------------------------------------------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Person Re-identification (mAP/R1) w/o re-ranking  | Market1501  | 91.6/96.1                                                    | 93.3/96.6                                                    | 93.9/96.9                                                    |
|                                                   | MSMT17      | 67.4/85.9                                                    | 76.9/90.8                                                    | 77.1/90.7                                                    |
| Person Re-identification (mAP/R1) with re-ranking | Market1501  | 95.3/96.6                                                    | 95.4/96.4                                                    | 95.6/96.7                                                    |
|                                                   | MSMT17      | 81.5/89.2                                                    | 86.5/91.7                                                    | 86.5/91.7                                                    |
| Attribute Recognition (mA)                        | PETA_ZS     | 74.37                                                        | 76.21                                                        | 76.43                                                        |
|                                                   | RAP_ZS      | 74.23                                                        | 75.95                                                        | 76.42                                                        |
|                                                   | PA100K      | 84.14                                                        | 86.25                                                        | 86.37                                                        |
| Person Search (mAP/R1)                            | CUHK-SYSU   | 94.9/95.7                                                    | 95.5/95.8                                                    | 94.9/95.5                                                    |
|                                                   | PRW         | 56.8/86.8                                                    | 59.8/86.7                                                    | 59.7/86.8                                                    |
| Pedestrian Detection (MR-2)                       | CityPersons | 10.3/40.8                                                    | 10.0/39.2                                                    | 9.7/39.4                                                     |
| Human Parsing (mIOU)                              | LIP         | 57.52                                                        | 60.21                                                        | 60.50                                                        |
| Pose Estimation (AP/AR)                           | COCO        | 74.4/79.6                                                    | 76.3/81.3                                                    | 76.6/81.5                                                    |

[1]：基于  LUPerson 数据集预训练

<a name='3'></a>
