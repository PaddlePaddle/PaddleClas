## PP-ShiTuV2 Image Recognition System

## Content

- [PP-ShiTuV2 Introduction](#pp-shituv2-introduction)
  - [Dataset](#dataset)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Model Inference](#model-inference)
  - [Model Deployment](#model-deployment)
- [Module introduction](#module-introduction)
  - [Mainbody Detection](#mainbody-detection)
  - [Feature Extraction](#feature-extraction)
    - [Dataset](#dataset-1)
    - [Backbone](#backbone)
    - [Network Structure](#network-structure)
    - [Data Augmentation](#data-augmentation)
- [references](#references)

## PP-ShiTuV2 Introduction

PP-ShiTuV2 is a practical lightweight general image recognition system based on PP-ShiTuV1. Compared with PP-ShiTuV1, it has higher recognition accuracy, stronger generalization ability and similar inference speed<sup>*</sup >. The system is mainly optimized for training data set and feature extraction, with a better backbone, loss function and training strategy. The retrieval performance of PP-ShiTuV2 in multiple practical application scenarios is significantly improved.

<div align="center">
<img src="../../images/structure.jpg" />
</div>

### Dataset

We remove some uncommon datasets add more common datasets in training stage. For more details, please refer to [PP-ShiTuV2 dataset](../image_recognition_pipeline/feature_extraction.md#4-实验部分).

The following takes the dataset of [PP-ShiTuV2](../image_recognition_pipeline/feature_extraction.md#4-实验部分) as an example to introduce the training, evaluation and inference process of the PP-ShiTuV2 model.

### Model Training

Download the 17 datasets in [PP-ShiTuV2 dataset](../image_recognition_pipeline/feature_extraction.md#4-实验部分) and merge them manually, then generate the annotation text file `train_reg_all_data_v2.txt`, and finally place them in `dataset` directory.

The merged 17 datasets structure is as follows:

```python
dataset/
├── Aliproduct/ # Aliproduct dataset folder
├── SOP/ # SOPt dataset folder
├── ...
├── Products-10k/ # Products-10k dataset folder
├── ...
└── train_reg_all_data_v2.txt # Annotation text file
```
The content of the generated `train_reg_all_data_v2.txt` is as follows:

```log
...
Aliproduct/train/50029/1766228.jpg 50029
Aliproduct/train/50029/1764348.jpg 50029
...
Products-10k/train/88823.jpg 186440
Products-10k/train/88824.jpg 186440
...
```

Then run the following command to train:

```shell
# Use GPU 0 for single-card training
export CUDA_VISIBLE_DEVICES=0
python3.7 tools/train.py \
-c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml

# Use 8 GPUs for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch tools/train.py \
-c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml
```
**Note:** `eval_during_train` will be enabled by default during training. After each `eval_interval` epoch, the model will be evaluated on the data set specified by `Eval` in the configuration file (the default is Aliproduct) and calculated for reference. index.

### Model Evaluation

Reference [Model Evaluation](../image_recognition_pipeline/feature_extraction_en.md#43-model-evaluation)

### Model Inference

Refer to [Python Model Reasoning](../quick_start/quick_start_recognition.md#22-Image Recognition Experience) and [C++ Model Reasoning](../../../deploy/cpp_shitu/readme_en.md)

### Model Deployment

Reference [Model Deployment](../inference_deployment/recognition_serving_deploy_en.md#32-service-deployment-and-request)

## Module introduction

### Mainbody Detection

The main body detection model uses `PicoDet-LCNet_x2_5`, for details refer to: [picodet_lcnet_x2_5_640_mainbody](../image_recognition_pipeline/mainbody_detection.md).

### Feature Extraction

#### Dataset

On the basis of the training data set used in PP-ShiTuV1, we removed the iCartoonFace data set, and added more widely used data sets, such as bird400, Cars, Products-10k, fruits- 262.

#### Backbone

We replaced the backbone network from `PPLCNet_x2_5` to [`PPLCNetV2_base`](../models/PP-LCNetV2.md). Compared with `PPLCNet_x2_5`, `PPLCNetV2_base` basically maintains a higher classification accuracy and reduces the 40% of inference time <sup>*</sup>.

**Note:** <sup>*</sup>The inference environment is based on Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz hardware platform, OpenVINO inference platform.

#### Network Structure

We adjust the `PPLCNetV2_base` structure, and added more general and effective optimizations for retrieval tasks such as pedestrian re-detection, landmark retrieval, and face recognition. It mainly includes the following points:

1. `PPLCNetV2_base` structure adjustment: The experiment found that [`ReLU`](../../../ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py#L322) at the end of the network has a great impact on the retrieval performance, [`FC`](../../../ppcls/arch/backbone/legendary_models/pp_lcnet_v2.py#L325) also causes a slight drop in retrieval performance, so we removed `ReLU` and `FC` at the end of BackBone.

2. `last stride=1`: No downsampling is performed at last stage, so as to increase the semantic information of the final output feature map, without having much more computational cost.

3. `BN Neck`: Add a `BatchNorm1D` layer after `BackBone` to normalize each dimension of the feature vector, bringing faster convergence.

    | Model                                                              | training data      | recall@1%(mAP%) |
    | :----------------------------------------------------------------- | :----------------- | :-------------- |
    | GeneralRecognition_PPLCNet_x2_5                                                         | PP-ShiTuV1 dataset | 65.9(54.3)      |
    | GeneralRecognitionV2_PPLCNetV2_base(TripletLoss) | PP-ShiTuV1 dataset | 72.3(60.5)      |

4. `TripletAngularMarginLoss`: We improved on the original `TripletLoss` (difficult triplet loss), changed the optimization objective from L2 Euclidean space to cosine space, and added an additional space between anchor and positive/negtive The hard distance constraint makes the training and testing goals closer and improves the generalization ability of the model.

    | Model | training data | recall@1%(mAP%) |
    | :---- | :------------ |: -------------- |
    | GeneralRecognitionV2_PPLCNetV2_base(TripletLoss) | PP-ShiTuV2 dataset | 71.9(60.2) |
    | GeneralRecognitionV2_PPLCNetV2_base(TripletAngularMarginLoss) | PP-ShiTuV2 dataset | 73.7(61.0) |

#### Data Augmentation

The target object may rotate to a certain extent and may not maintain an upright state when the actual camera is shot, so we add [random rotation augmentation](../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L117) in the data augmentation to make retrieval more robust in real scenes.

Combining the above strategies, the final experimental results on multiple data sets are as follows:

  | Model      | product<sup>*</sup> |
  | :--------- | :------------------ |
  | -          | recall@1%(mAP%)     |
  | GeneralRecognition_PPLCNet_x2_5 | 65.9(54.3)          |
  | GeneralRecognitionV2_PPLCNetV2_base | 73.7(61.0)          |

  | Models     | Aliproduct      | VeRI-Wild       | LogoDet-3k      | iCartoonFace    | SOP             | Inshop           |
  | :--------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :--------------- |
  | -          | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@ 1%(mAP%) |
  | GeneralRecognition_PPLCNet_x2_5 | 83.9(83.2)      | 88.7(60.1)      | 86.1(73.6)      | 84.1(72.3)      | 79.7(58.6)      | 89.1(69.4)       |
  | GeneralRecognitionV2_PPLCNetV2_base | 84.2(83.3)      | 87.8(68.8)      | 88.0(63.2)      | 53.6(27.5)      | 77.6(55.3)      | 90.8(74.3)       |

  | model      | gldv2           | imdb_face       | iNat            | instre          | sketch          | sop<sup>*</sup>  |
  | :--------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :--------------- |
  | -          | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@1%(mAP%) | recall@ 1%(mAP%) |
  | GeneralRecognition_PPLCNet_x2_5 | 98.2(91.6)      | 28.8(8.42)      | 12.6(6.1)       | 72.0(50.4)      | 27.9(9.5)       | 97.6(90.3)       |
  | GeneralRecognitionV2_PPLCNetV2_base | 98.1(90.5)      | 35.9(11.2)      | 38.6(23.9)      | 87.7(71.4)      | 39.3(15.6)      | 98.3(90.9)       |

**Note:** The product dataset is made to verify the generalization performance of PP-ShiTu, and all the data are not present in the training and testing sets. The data contains 7 categories ( cosmetics, landmarks, wine, watches, cars, sports shoes, beverages) and 250 sub-categories. When testing, use the labels of 250 small classes for testing; the sop dataset comes from [GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval](https://arxiv.org/abs/2111.13122), which can be regarded as " SOP" dataset.

## references
1. Schall, Konstantin, et al. "GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval." International Conference on Multimedia Modeling. Springer, Cham, 2022.
2. Luo, Hao, et al. "A strong baseline and batch normalization neck for deep person re-identification." IEEE Transactions on Multimedia 22.10 (2019): 2597-2609.
