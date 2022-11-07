[简体中文](README_ch.md) | English

# PaddleClas

## Introduction

PaddleClas is an image classification and image recognition toolset for industry and academia, helping users train better computer vision models and apply them in real scenarios.

|                       PP-ShiTuV2                       | PULC: **P**ractical **U**ltra **L**ight-weight image **C**lassification solutions |
| :----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="./docs/images/shituv2.gif"  width = "450" /> | <img src="./docs/images/class_simple_en.gif"  width = "600" /> |

## 📣 Recent updates

- 🔥️ Release [PP-ShiTuV2](./docs/en/PPShiTu/PPShiTuV2_introduction.md), recall1 is improved by nearly 8 points, covering 20+ recognition scenarios, with [index management tool](./deploy/shitu_index_manager) and [Android Demo](./docs/en/quick_start/quick_start_recognition_en.md) for better experience.
- 2022.6.15 Release [**P**ractical **U**ltra **L**ight-weight image **C**lassification solutions](./docs/en/PULC/PULC_quickstart_en.md). PULC models inference within 3ms on CPU devices, with accuracy on par with SwinTransformer. We also release 9 practical classification models covering pedestrian, vehicle and OCR scenario.
- 2022.4.21 Added the related [code](https://github.com/PaddlePaddle/PaddleClas/pull/1820/files) of the CVPR2022 oral paper [MixFormer](https://arxiv.org/pdf/2204.02557.pdf).

- 2021.09.17 Add PP-LCNet series model developed by PaddleClas, these models show strong competitiveness on Intel CPUs.
For the introduction of PP-LCNet, please refer to [paper](https://arxiv.org/pdf/2109.15099.pdf) or [PP-LCNet model introduction](docs/en/models/PP-LCNet_en.md). The metrics and pretrained model are available [here](docs/en/algorithm_introduction/ImageNet_models_en.md).

- 2021.06.29 Add [Swin-transformer](docs/en/models/SwinTransformer_en.md)) series model，Highest top1 acc on ImageNet1k dataset reaches 87.2%, training, evaluation and inference are all supported. Pretrained models can be downloaded [here](docs/en/algorithm_introduction/ImageNet_models_en.md#16).
- 2021.06.16 PaddleClas release/2.2. Add metric learning and vector search modules. Add product recognition, animation character recognition, vehicle recognition and logo recognition. Added 30 pretrained models of LeViT, Twins, TNT, DLA, HarDNet, and RedNet, and the accuracy is roughly the same as that of the paper.
- [more](./docs/en/others/update_history_en.md)

## 🌟 Features

PaddleClas release PP-HGNet、PP-LCNetv2、 PP-LCNet and **S**imple **S**emi-supervised **L**abel **D**istillation algorithms, and support plenty of image classification and image recognition algorithms.Based on th algorithms above, PaddleClas release PP-ShiTu image recognition system and [**P**ractical **U**ltra **L**ight-weight image **C**lassification solutions](docs/en/PULC/PULC_quickstart_en.md).


![](https://user-images.githubusercontent.com/11568925/189268878-43d9d35b-90cf-425a-859e-767f8d94c5f7.png)

## Welcome to Join the Technical Exchange Group

* You can also scan the QR code below to join the PaddleClas QQ group and WeChat group (add and replay "C") to get more efficient answers to your questions and to communicate with developers from all walks of life. We look forward to hearing from you.

<div align="center">
<img src="https://user-images.githubusercontent.com/80816848/164383225-e375eb86-716e-41b4-a9e0-4b8a3976c1aa.jpg" width="200"/>
<img src="https://user-images.githubusercontent.com/48054808/160531099-9811bbe6-cfbb-47d5-8bdb-c2b40684d7dd.png" width="200"/>
</div>

## Quick Start
Quick experience of PP-ShiTu image recognition system：[Link](./docs/en/quick_start/quick_start_recognition_en.md)

<div align="center">
<img src="./docs/images/quick_start/android_demo/PPShiTu_qrcode.png"  width = "40%" />
<p>PP-ShiTuV2 Android Demo</p>
</div>

Quick experience of **P**ractical **U**ltra **L**ight-weight image **C**lassification models：[Link](docs/en/PULC/PULC_quickstart_en.md)

## Tutorials

- [Install Paddle](./docs/en/installation/install_paddle_en.md)
- [Install PaddleClas Environment](./docs/en/installation/install_paddleclas_en.md)
- [PP-ShiTuV2 Image Recognition Systems Introduction](./docs/en/PPShiTu/PPShiTuV2_introduction.md)
  - [Image Recognition Quick Start](docs/en/quick_start/quick_start_recognition_en.md)
  - [20+ application scenarios](docs/zh_CN/deployment/PP-ShiTu/application_scenarios.md)
  - Submodule Introduction and Model Training
    - [Mainbody Detection](docs/zh_CN/training/PP-ShiTu/mainbody_detection.md)
    - [Feature Extraction](./docs/en/image_recognition_pipeline/feature_extraction_en.md)
    - [Vector Search](./docs/en/image_recognition_pipeline/vector_search_en.md)
    - [Hash Encoding](docs/zh_CN/training/PP-ShiTu/deep_hashing.md)
  - PipeLine Inference and Deployment
    - [Python Inference](docs/en/inference_deployment/python_deploy_en.md)
    - [C++ Inference](deploy/cpp_shitu/readme_en.md)
    - [Serving Deployment](docs/en/inference_deployment/recognition_serving_deploy_en.md)
    - [Lite Deployment](docs/en/inference_deployment/paddle_lite_deploy_en.md)
    - [Shitu Gallery Manager Tool](docs/zh_CN/deployment/PP-ShiTu/gallery_manager.md)
- [Practical Ultra Light-weight image Classification solutions](./docs/en/PULC/PULC_train_en.md)
  - [PULC Quick Start](docs/en/PULC/PULC_quickstart_en.md)
  - [PULC Model Zoo](docs/en/PULC/PULC_model_list_en.md)
    - [PULC Classification Model of Someone or Nobody](docs/en/PULC/PULC_person_exists_en.md)
    - [PULC Recognition Model of Person Attribute](docs/en/PULC/PULC_person_attribute_en.md)
    - [PULC Classification Model of Wearing or Unwearing Safety Helmet](docs/en/PULC/PULC_safety_helmet_en.md)
    - [PULC Classification Model of Traffic Sign](docs/en/PULC/PULC_traffic_sign_en.md)
    - [PULC Recognition Model of Vehicle Attribute](docs/en/PULC/PULC_vehicle_attribute_en.md)
    - [PULC Classification Model of Containing or Uncontaining Car](docs/en/PULC/PULC_car_exists_en.md)
    - [PULC Classification Model of Text Image Orientation](docs/en/PULC/PULC_text_image_orientation_en.md)
    - [PULC Classification Model of Textline Orientation](docs/en/PULC/PULC_textline_orientation_en.md)
    - [PULC Classification Model of Language](docs/en/PULC/PULC_language_classification_en.md)
- PP Series Backbone
    - [PP-HGNet](docs/en/models/PP-HGNet_en.md)
    - [PP-LCNet](docs/en/models/PP-LCNet_en.md)
    - [PP-LCNetv2](docs/en/models/PP-LCNetv2_en.md)
- [Introduction to Image Recognition Systems](#Introduction_to_Image_Recognition_Systems)
- [Image Recognition Demo images](#Rec_Demo_images)
- [PULC demo images](#Clas_Demo_images)
- Algorithms Introduction
    - [Backbone Network and Pre-trained Model Library](./docs/en/algorithm_introduction/ImageNet_models_en.md)
    - [Mainbody Detection](./docs/en/image_recognition_pipeline/mainbody_detection_en.md)
    - [Feature Learning](./docs/en/image_recognition_pipeline/feature_extraction_en.md)
    - [Vector Search](./deploy/vector_search/README.md)
- Inference Model Prediction
    - [Python Inference](./docs/en/inference_deployment/python_deploy_en.md)
    - [C++ Classfication Inference](./deploy/cpp/readme_en.md)
- Model Deploy (only support classification for now, recognition coming soon)
    - [Hub Serving Deployment](./deploy/hubserving/readme_en.md)
    - [Mobile Deployment](./deploy/lite/readme_en.md)
    - [Inference Using whl](./docs/en/inference_deployment/whl_deploy_en.md)
- Advanced Tutorial
    - [Knowledge Distillation](./docs/en/advanced_tutorials/distillation/distillation_en.md)
    - [Model Quantization](./docs/en/algorithm_introduction/model_prune_quantization_en.md)
    - [Data Augmentation](./docs/en/advanced_tutorials/DataAugmentation_en.md)
- [License](#License)
- [Contribution](#Contribution)

<a name="Introduction_to_PULC"></a>
## Introduction to Practical Ultra Light-weight image Classification solutions
<div align="center">
<img src="https://user-images.githubusercontent.com/19523330/173011854-b10fcd7a-b799-4dfd-a1cf-9504952a3c44.png"  width = "800" />
</div>
PULC solutions consists of PP-LCNet light-weight backbone, SSLD pretrained models, Ensemble of Data Augmentation strategy and SKL-UGI knowledge distillation.
PULC models inference within 3ms on CPU devices, with accuracy comparable with SwinTransformer. We also release 9 practical models covering pedestrian, vehicle and OCR.

<a name="Introduction_to_Image_Recognition_Systems"></a>
## Introduction to Image Recognition Systems

<div align="center">
<img src="./docs/images/structure.jpg"  width = "800" />
</div>

PP-ShiTuV2 is a practical lightweight general image recognition system, which is mainly composed of three modules: mainbody detection model, feature extraction model and vector search tool. The system adopts a variety of strategies including backbone network, loss function, data augmentations, optimal hyperparameters, pre-training model, model pruning and quantization. Compared to V1, PP-ShiTuV2, Recall1 is improved by nearly 8 points. For more details, please refer to [PP-ShiTuV2 introduction](./docs/en/PPShiTu/PPShiTuV2_introduction.md).
For a new unknown category, there is no need to retrain the model, just prepare images of new category, extract features and update retrieval database and the category can be recognised.

<a name="Rec_Demo_images"></a>
## PP-ShiTuV2 Demo images

- Drinks recognition

<div align="center">
<img src="docs/images/drink_demo.gif">
</div>


- Product recognition

<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769644-51604f80-d2d7-11eb-8290-c53b12a5c1f6.gif"  width = "400" />
</div>


- Cartoon character recognition

<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769746-6b019700-d2d7-11eb-86df-f1d710999ba6.gif"  width = "400" />
</div>


- Logo recognition

<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769837-7fde2a80-d2d7-11eb-9b69-04140e9d785f.gif"  width = "400" />
</div>



- Car recognition

<div align="center">
<img src="https://user-images.githubusercontent.com/18028216/122769916-8ec4dd00-d2d7-11eb-8c60-42d89e25030c.gif"  width = "400" />
</div>


<a name="Clas_Demo_images"></a>
## PULC demo images
<div align="center">
<img src="docs/images/classification_en.gif">
</div>


<a name="License"></a>
## License
PaddleClas is released under the Apache 2.0 license <a href="https://github.com/PaddlePaddle/PaddleCLS/blob/master/LICENSE">Apache 2.0 license</a>


<a name="Contribution"></a>
## Contribution
Contributions are highly welcomed and we would really appreciate your feedback!!


- Thank [nblib](https://github.com/nblib) to fix bug of RandErasing.
- Thank [chenpy228](https://github.com/chenpy228) to fix some typos PaddleClas.
- Thank [jm12138](https://github.com/jm12138) to add ViT, DeiT models and RepVGG models into PaddleClas.
- Thank [FutureSI](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/76563) to parse and summarize the PaddleClas code.
