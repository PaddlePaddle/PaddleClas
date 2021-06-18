# Vehicle Recognition

This part mainly includes two parts: vehicle fine-grained classification and vehicle Reid.

The goal of fine-grained classification is to recognize images belonging to multiple subordinate categories of a super-category, e.g., different species of animals/plants, different models of cars, different kinds of retail products. Obviously, fine-grained vehicle classification is to classify different sub categories of vehicles.

Vehicle ReID aims to re-target vehicle images across non-overlapping camera views given a query image. It has many practical applications, such as for analyzing and managing the traffic flows in Intelligent Transport System. In this process, how to extract robust features is particularly important.

In this document, the same training scheme is used to try the two application respectively.

## 1 Pipeline

See the pipline of [feature learning](./feature_learning_en.md) for details.

The config file of Vehicle ReID: [ResNet50_ReID.yaml](../../../ppcls/configs/Vehicle/ResNet50_ReID.yaml).

The config file of Vehicle fine-grained classification：[ResNet50.yaml](../../../ppcls/configs/Vehicle/ResNet50.yaml).

 The details are as follows.

### 1.1 Data Augmentation

Different from classification, this part mainly uses the following methods:

- `Resize` to 224. Especially for ReID, the vehicle image is already croped using bbox by detector. So if `CenterCrop` is used, more vehicle information will be lost.
- [AugMix](https://arxiv.org/abs/1912.02781v1)：Simulation of lighting changes, camera position changes and other real scenes.
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)：Simulate  occlusion.

### 1.2 Backbone

使用`ResNet50`作为backbone，同时做了如下修改：

 Using `ResNet50` as  backbone, and make the following modifications:

- Last stage stride = 1, keep the size of the final output feature map to 14x14. At the cost of increasing a small amount of calculation, the ability of feature expression is greatly improved.

code：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

### 1.3 Neck

In order to reduce the complexity of calculating feature distance in inferencne, an embedding convolution layer is added, and the feature dimension is 512.

### 1.4 Metric Learning Losses

- In vehicle ReID，[SupConLoss](../../../ppcls/loss/supconloss.py) , [ArcLoss](../../../ppcls/arch/gears/arcmargin.py) are used. The weight ratio of two losses is 1:1.
- In vehicle fine-grained classification, [TtripLet Loss](../../../ppcls/loss/triplet.py), [ArcLoss](../../../ppcls/arch/gears/arcmargin.py) are used. The weight ratio of two losses is 1:1.

## Experiment

### 2.1 Vehicle ReID

<img src="../../images/recognition/vehicle/cars.JPG" style="zoom:50%;" />

This method is used in VERI-Wild dataset. This dataset was captured in a large CCTV monitoring system in an unrestricted scenario for a month (30 * 24 hours). The system consists of 174 cameras, which are distributed in large area of more than 200 square kilometers. The original vehicle image set contains 12 million vehicle images. After data cleaning and labeling, 416314 images and 40671 vehicle ids are collected. [See the paper for details]( https://github.com/PKU-IMRE/VERI-Wild).

|         **Methods**          | **Small** |           |           |
| :--------------------------: | :-------: | :-------: | :-------: |
|                              |    mAP    |   Top1    |   Top5    |
| Strong baesline(Resnet50)[1] |   76.61   |   90.83   |   97.29   |
|    HPGN(Resnet50+PGN)[2]     |   80.42   |   91.37   |     -     |
|   GLAMOR(Resnet50+PGN)[3]    |   77.15   |   92.13   |   97.43   |
|      PVEN(Resnet50)[4]       |   79.8    |   94.01   |   98.06   |
|    SAVER(VAE+Resnet50)[5]    |   80.9    |   93.78   |   97.93   |
|    PaddleClas  baseline1     |   65.6    |   92.37   |   97.23   |
|    PaddleClas  baseline2     |   80.09   | **93.81** | **98.26** |

 Baseline1 is the released, and baseline2 will be released soon.

### 2.2 Vehicle Fine-grained Classification

 In this applications, we use [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) as train dataset.

![](../../images/recognition/vehicle/CompCars.png)

The images in the dataset mainly come from the network and monitoring  data. The network data includes 163 automobile manufacturers and 1716  automobile models, which includes **136726** full vehicle images and **27618** partial vehicle images. The network car data includes the information of  bounding box, perspective and five  attributes (maximum speed, displacement, number of doors, number of  seats and car type) for vehicles. The monitoring data includes  **50000** front view images.

 It is worth noting that this dataset needs to generate labels  according to its own needs. For example, in this demo, vehicles of the  same model produced in different years are regarded as the same  category. Therefore, the total number of categories is 431.

|           **Methods**           | Top1 Acc  |
| :-----------------------------: | :-------: |
|        ResNet101-swp[6]         |   97.6%   |
|      Fine-Tuning DARTS[7]       |   95.9%   |
|       Resnet50 + COOC[8]        |   95.6%   |
|             A3M[9]              |   95.4%   |
| PaddleClas  baseline (ResNet50) | **97.1**% |

## 3 References

[1] Bag of Tricks and a Strong Baseline for Deep Person Re-Identification.CVPR workshop 2019.

[2] Exploring Spatial Significance via Hybrid Pyramidal Graph Network for Vehicle Re-identification. In arXiv preprint arXiv:2005.14684

[3] GLAMORous: Vehicle Re-Id in Heterogeneous Cameras Networks with Global and Local Attention. In arXiv preprint arXiv:2002.02256

[4] Parsing-based view-aware embedding network for vehicle re-identification. CVPR 2020.

[5] The Devil is in the Details: Self-Supervised Attention for Vehicle Re-Identification. In ECCV 2020.

[6] Deep CNNs With Spatially Weighted Pooling for Fine-Grained Car Recognition. IEEE Transactions on Intelligent Transportation Systems, 2017.

[7] Fine-Tuning DARTS for Image Classification. 2020.

[8] Fine-Grained Vehicle Classification with Unsupervised Parts Co-occurrence Learning. 2018

[9] Attribute-Aware Attention Model for Fine-grained Representation Learning. 2019.
