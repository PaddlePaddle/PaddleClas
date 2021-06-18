# Product Recognition

Product recogniton is now widely used . The way of shopping by taking a photo has been adopted by many people. And the unmanned settlement platform has entered the major supermarkets,  which is also supported by product recognition technology. The technology is about the process of "product detection + product identification". The product detection module is responsible for detecting potential product areas, and the product identification model is responsible for identifying the main body detected by the product detection module. The recognition module uses the retrieval method to get the similarity rank of product in database and the query image . This document mainly introduces the feature extraction part of product pictures.

## 1 Pipeline

See the pipline of [feature learning](./feature_learning_en.md) for details.

The config file: [ResNet50_vd_Aliproduct.yaml](../../../ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml)

 The details are as follows.

### 1.1 Data Augmentation

- `RandomCrop`: 224x224
- `RandomFlip`
- `Normlize`:  normlize images to 0~1

### 1.2 Backbone

Using `ResNet50_vd` as the backbone, whicle is pretrained on ImageNet.

### 1.3 Neck

 A 512 dimensional embedding FC layer without batchnorm and activation is used.

### 1.4 Metric Learning Losses

 At present, `CELoss` is used. In order to obtain more robust  features, other loss will be used for training in the future. Please  look forward to it.

## 2 Experiment

 This scheme is tested on Aliproduct [1] dataset. This dataset is an open source dataset of Tianchi competition, which is the largest open source product data set at present. It has more than 50000 identification categories and about 2.5 million training pictures.

 On this data, the single model Top1 Acc: 85.67%.

## 3 References

[1] Weakly Supervised Learning with Side Information for Noisy Labeled Images. ECCV, 2020.
