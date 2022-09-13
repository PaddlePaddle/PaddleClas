# Feature Extraction

## Catalogue

- [1.Introduction](#1)
- [2.Network Structure](#2)
- [3.General Recognition Models](#3)
- [4.Customized Feature Extraction](#4)
    - [4.1 Data Preparation](#4.1)
    - [4.2 Model Training](#4.2)
    - [4.3 Model Evaluation](#4.3)
    - [4.4 Model Inference](#4.4)

<a name="1"></a>

## 1. Abstract

Feature extraction plays a key role in image recognition, which serves to transform the input image into a fixed dimensional feature vector for subsequent [vector search](./vector_search_en.md). Good features boast great similarity preservation, i.e., in the feature space, pairs of images with high similarity should have higher feature similarity (closer together), and pairs of images with low similarity should have less feature similarity (further apart). [Deep Metric Learning](../algorithm_introduction/metric_learning_en.md) is applied to explore how to obtain features with high representational power through deep learning.

<a name="2"></a>

## 2. Introduction


In order to customize the image recognition task flexibly, the whole network is divided into Backbone, Neck, Head, and Loss. The figure below illustrates the overall structure:

![img](../../images/feature_extraction_framework_en.png)

Functions of the above modules :

- **Backbone**: Specifies the backbone network to be used. It is worth noting that the ImageNet-based pre-training model provided by PaddleClas has an output of 1000 for the last layer, which demands for customization according to the required feature dimensions.
- **Neck**: Used for feature augmentation and feature dimension transformation. Here it can be a simple Linear Layer for feature dimension transformation, or a more complex FPN structure for feature augmentation.
- **Head**: Used to transform features into logits. In addition to the common Fc Layer, cosmargin, arcmargin, circlemargin and other modules are all available choices.
- **Loss**: Specifies the Loss function to be used. It is designed as a combined form to facilitate the combination of Classification Loss and Pair_wise Loss.

<a name="3"></a>

## 3. Methods

#### 3.1 Backbone

The Backbone part adopts [PP-LCNetV2_base](../models/PP-LCNetV2.md), which is based on `PPLCNet_V1`, including Rep strategy, PW convolution, Shortcut, activation function improvement, SE module improvement After several optimization points, the final classification accuracy is similar to `PPLCNet_x2_5`, and the inference delay is reduced by 40%<sup>*</sup>. During the experiment, we made appropriate improvements to `PPLCNetV2_base`, so that it can achieve higher performance in recognition tasks while keeping the speed basically unchanged, including: removing `ReLU` and ` at the end of `PPLCNetV2_base` FC`, change the stride of the last stage (RepDepthwiseSeparable) to 1.

**Note:** <sup>*</sup>The inference environment is based on Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz hardware platform, OpenVINO inference platform.

#### 3.2 Neck

We use [BN Neck](../../../ppcls/arch/gears/bnneck.py) to standardize each dimension of the features extracted by Backbone, reducing difficulty of optimizing metric learning loss and identification  loss simultaneously.

#### 3.3 Head

We use [FC Layer](../../../ppcls/arch/gears/fc.py) as the classification head to convert features into logits for classification loss.

#### 3.4 Loss

We use [Cross entropy loss](../../../ppcls/loss/celoss.py) and [TripletAngularMarginLoss](../../../ppcls/loss/tripletangularmarginloss.py), and we improved the original TripletLoss(TriHard Loss), replacing the optimization objective from L2 Euclidean space to cosine space, adding a hard distance constraint between anchor and positive/negtive, so the generalization ability of the model is improved. For detailed configuration files, see [GeneralRecognitionV2_PPLCNetV2_base.yaml](../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L63-77).

#### 3.5 Data Augmentation

We consider that the object may rotate to a certain extent and can not maintain an upright state in real scenes, so we add an appropriate [random rotation](../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L117) in the data augmentation to improve the retrieval performance in real scenes.

<a name="4"></a>

## 4. Experimental

We reasonably expanded and optimized the original training data, and finally used a summary of the following 17 public datasets:

| Dataset                | Data Amount | Number of Categories |  Scenario   |                                     Dataset Address                                     |
| :--------------------- | :---------: | :------------------: | :---------: | :-------------------------------------------------------------------------------------: |
| Aliproduct             |   2498771   |        50030         | Commodities |      [Address](https://retailvisionworkshop.github.io/recognition_challenge_2020/)      |
| GLDv2                  |   1580470   |        81313         |  Landmark   |               [address](https://github.com/cvdfoundation/google-landmark)               |
| VeRI-Wild              |   277797    |        30671         |  Vehicles   |                    [Address](https://github.com/PKU-IMRE/VERI-Wild)                     |
| LogoDet-3K             |   155427    |         3000         |    Logo     |              [Address](https://github.com/Wangjing1551/LogoDet-3K-Dataset)              |
| SOP                    |    59551    |        11318         | Commodities |              [Address](https://cvgl.stanford.edu/projects/lifted_struct/)               |
| Inshop                 |    25882    |         3997         | Commodities |            [Address](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)             |
| bird400                |    58388    |         400          |    birds    |          [address](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)          |
| 104flows               |    12753    |         104          |   Flowers   |              [Address](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)              |
| Cars                   |    58315    |         112          |  Vehicles   |            [Address](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)            |
| Fashion Product Images |    44441    |          47          |  Products   | [Address](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) |
| flowerrecognition      |    24123    |          59          |   flower    |         [address](https://www.kaggle.com/datasets/aymenktari/flowerrecognition)         |
| food-101               |   101000    |         101          |    food     |         [address](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)          |
| fruits-262             |   225639    |         262          |   fruits    |            [address](https://www.kaggle.com/datasets/aelchimminut/fruits262)            |
| inaturalist            |   265213    |         1010         |   natural   |           [address](https://github.com/visipedia/inat_comp/tree/master/2017)            |
| indoor-scenes          |    15588    |          67          |   indoor    |       [address](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)       |
| Products-10k           |   141931    |         9691         |  Products   |                       [Address](https://products-10k.github.io/)                        |
| CompCars               |    16016    |         431          |  Vehicles   |     [Address](http://​​​​​​http://ai.stanford.edu/~jkrause/cars/car_dataset.html​)      |
| **Total**              |   **6M**    |       **192K**       |      -      |                                            -                                            |

The final model accuracy metrics are shown in the following table:

| Model                  | Latency (ms) | Storage (MB) | product<sup>*</sup> |      | Aliproduct |      | VeRI-Wild |      | LogoDet-3k |      | iCartoonFace |      | SOP      |           | Inshop |          | gldv2 |          | imdb_face |          | iNat |          | instre |          | sketch |          | sop |     |
| :--------------------- | :----------- | :----------- | :------------------ | :--- | ---------- | ---- | --------- | ---- | ---------- | ---- | ------------ | ---- | -------- | --------- | ------ | -------- | ----- | -------- | --------- | -------- | ---- | -------- | ------ | -------- | ------ | -------- | --- | --- |
|                        |              |              | recall@1            | mAP  | recall@1   | mAP  | recall@1  | mAP  | recall@1   | mAP  | recall@1     | mAP  | recall@1 | mrecall@1 | mAP    | recall@1 | mAP   | recall@1 | mAP       | recall@1 | mAP  | recall@1 | mAP    | recall@1 | mAP    | recall@1 | mAP |
| PP-ShiTuV1_general_rec | 5.0          | 34           | 65.9                | 54.3 | 83.9       | 83.2 | 88.7      | 60.1 | 86.1       | 73.6 |              | 50.4 | 27.9     | 9.5       | 97.6   | 90.3     |
| PP-ShiTuV2_general_rec | 6.1          | 19           | 73.7                | 61.0 | 84.2       | 83.3 | 87.8      | 68.8 | 88.0       | 63.2 | 53.6         | 27.5 |          | 71.4      | 39.3   | 15.6     | 98.3  | 90.9     |

*The product dataset is a dataset made to verify the generalization performance of PP-ShiTu, and all the data are not present in the training and testing sets. The data contains 7 major categories (cosmetics, landmarks, wine, watches, cars, sports shoes, beverages) and 250 subcategories. When testing, use the labels of 250 small classes for testing; the sop dataset comes from [GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval](https://arxiv.org/abs/2111.13122), which can be regarded as " SOP" dataset.
* Pre-trained model address: [general_PPLCNetV2_base_pretrained_v1.0.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams)
* The evaluation metrics used are: `Recall@1` and `mAP`
* The CPU specific information of the speed test machine is: `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz`
* The evaluation conditions of the speed indicator are: MKLDNN is turned on, and the number of threads is set to 10

<a name="5"></a>

## 5. Custom Feature Extraction

Custom feature extraction refers to retraining the feature extraction model according to your own task.

Based on the `GeneralRecognitionV2_PPLCNetV2_base.yaml` configuration file, the following describes the main four steps: 1) data preparation; 2) model training; 3) model evaluation; 4) model inference

<a name="5.1"></a>

### 5.1 Data Preparation

First you need to customize your own dataset based on the task. Please refer to [Dataset Format Description](../data_preparation/recognition_dataset.md) for the dataset format and file structure.

After the preparation is complete, it is necessary to modify the content related to the data configuration in the configuration file, mainly including the path of the dataset and the number of categories. As is as shown below:

- Modify the number of classes:
  ```yaml
  Head:
    name: FC
    embedding_size: *feat_dim
    class_num: 192612 # This is the number of classes
    weight_attr:
      initializer:
        name: Normal
        std: 0.001
    bias_attr: False
  ```
- Modify the training dataset configuration:
  ```yaml
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ # Here is the directory where the train dataset is located
      cls_label_path: ./dataset/train_reg_all_data_v2.txt # Here is the path of the label file corresponding to the train dataset
      relabel: True
  ```
- Modify the query data configuration in the evaluation dataset:
  ```yaml
  Query:
    dataset:
      name: VeriWild
      image_root: ./dataset/Aliproduct/ # Here is the directory where the query dataset is located
      cls_label_path: ./dataset/Aliproduct/val_list.txt # Here is the path of the label file corresponding to the query dataset
  ```
- Modify the gallery data configuration in the evaluation dataset:
  ```yaml
  Gallery:
    dataset:
      name: VeriWild
      image_root: ./dataset/Aliproduct/ # This is the directory where the gallery dataset is located
      cls_label_path: ./dataset/Aliproduct/val_list.txt # Here is the path of the label file corresponding to the gallery dataset
  ```

<a name="5.2"></a>

### 5.2 Model training

Model training mainly includes the starting training and restoring training from checkpoint

- Single machine and single card training
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml
  ```
- Single machine multi-card training
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml
  ```
**Notice:**
The online evaluation method is used by default in the configuration file. If you want to speed up the training, you can turn off the online evaluation function, just add `-o Global.eval_during_train=False` after the above scripts.

After training, the final model files `latest.pdparams`, `best_model.pdarams` and the training log file `train.log` will be generated in the output directory. Among them, `best_model` saves the best model under the current evaluation index, and `latest` is used to save the latest generated model, which is convenient to resume training from the checkpoint when training task is interrupted. Training can be resumed from a checkpoint by adding `-o Global.checkpoint="path_to_resume_checkpoint"` to the end of the above training scripts, as shown below.

- Single machine and single card checkpoint recovery training
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.checkpoint="output/RecModel/latest"
  ```
- Single-machine multi-card checkpoint recovery training
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/train.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.checkpoint="output/RecModel/latest"
  ```

<a name="5.3"></a>

### 5.3 Model Evaluation

In addition to the online evaluation of the model during training, the evaluation program can also be started manually to obtain the specified model's accuracy metrics.

- Single Card Evaluation
  ```shell
  export CUDA_VISIBLE_DEVICES=0
  python3.7 tools/eval.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.pretrained_model="output/RecModel/best_model"
  ```

- Multi Card Evaluation
  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" \
  tools/eval.py \
  -c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
  -o Global.pretrained_model="output/RecModel/best_model"
  ```
**Note:** Multi Card Evaluation is recommended. This method can quickly obtain the metric cross all the data by using multi-card parallel computing, which can speed up the evaluation.

<a name="5.4"></a>

### 5.4 Model Inference

The inference process consists of two steps: 1) Export the inference model; 2) Model inference to obtain feature vectors

#### 5.4.1 Export inference model

First, you need to convert the `*.pdparams` model file into inference format. The conversion script is as follows.
```shell
python3.7 tools/export_model.py \
-c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
-o Global.pretrained_model="output/RecModel/best_model"
```
The generated inference model is located in the `PaddleClas/inference` directory by default, which contains three files, `inference.pdmodel`, `inference.pdiparams`, `inference.pdiparams.info`.
Where `inference.pdmodel` is used to store the structure of the inference model, `inference.pdiparams` and `inference.pdiparams.info` are used to store parameter information related to the inference model.

#### 5.4.2 Get feature vector

Use the inference model converted in the previous step to convert the input image into corresponding feature vector. The inference script is as follows.

```shell
cd deploy
python3.7 python/predict_rec.py \
-c configs/inference_rec.yaml \
-o Global.rec_inference_model_dir="../inference"
```
The resulting feature output format is as follows:

```log
wangzai.jpg: [-7.82453567e-02 2.55877394e-02 -3.66694555e-02 1.34572461e-02
  4.39076796e-02 -2.34078392e-02 -9.49947070e-03 1.28221214e-02
  5.53947650e-02 1.01355985e-02 -1.06436480e-02 4.97181974e-02
 -2.21862812e-02 -1.75557341e-02 1.55848479e-02 -3.33278324e-03
 ...
 -3.40284109e-02 8.35561901e-02 2.10910216e-02 -3.27066667e-02]
```

In most cases, just getting the features may not meet the users' requirements. If you want to go further on the image recognition task, you can refer to the document [Vector Search](./vector_search.md).

<a name="6"></a>

## 6. Summary

As a key part of image recognition, the feature extraction module has a lot of points for improvement in the network structure and the the loss function. Different datasets have their own characteristics, such as person re-identification, commodity recognition, face recognition. According to these characteristics, the academic community has proposed various methods, such as PCB, MGN, ArcFace, CircleLoss, TripletLoss, etc., which focus on the ultimate goal of increasing the gap between classes and reducing the gap within classes, so as to make a retrieval model robust enough in most scenes.

<a name="7"></a>

## 7. References

1. [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf)
2. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
