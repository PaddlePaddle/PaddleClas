[简体中文](README.md) | English

# PaddleClas

## Introduction

PaddleClas is a toolset for image recognition tasks prepared for the industry and academia helping users train better computer vision models and apply them in real scenarios.

**Recent updates**

- 2021.06.16 PaddleClas release/2.2.
   - Add metric learning and vector search module.
   - Add product recognition, cartoon character recognition, car recognition and logo recognition.
   - Added 30 pretrained models of LeViT, Twins, TNT, DLA, HarDNet, and RedNet, and the accuracy is roughly the same as that of the paper.

- 2021.05.14
   - Add `SwinTransformer` series pretrained models, whose Top-1 Acc on ImageNet-1k dataset reaches 87.19%.

- 2021.04.15
   - Add `MixNet` and `ReXNet` pretrained models, `MixNet_L`'s Top-1 Acc on ImageNet-1k reaches 78.6% and `ReXNet_3_0` reaches 82.09%.

- [more](./docs/en/update_history_en.md)

## Features

- A practical image recognition system: Integrate modules for detection, feature learning and retrieval, widely applicable to all types of image recognition tasks.
Four sample solutions are provided, including product recognition, vehicle recognition, logo recognition and animation character recognition.

- Rich library of pre-trained models: Provide a total of 134 ImageNet pre-trained models in 29 series, with 6 selected series of models supporting fast structural modification.

- Comprehensive and easy-to-use feature learning components: A large number of metric learning methods are integrated and can be combined and switched at will through configuration files.

- SSLD knowledge distillation: the recognition accuracy of distilled models based on this solution is generally improved by more than 3%.

- Data augmentation: Support for 8 data augmentation algorithms such as AutoAugment, Cutout, Cutmix, etc.  with detailed introduction, code replication and evaluation of effectiveness in a unified experimental environment.

 

## Image Recognition System Effect Demonstration
<div align="center">
<img src="./docs/images/recognition.gif"  width = "400" />
</div>

## Welcome to Join the Technical Exchange Group

* You can also scan the QR code below to join the PaddleClas WeChat group to get more efficient answers to your questions and to communicate with developers from all walks of life. We look forward to hearing from you.

<div align="center">
<img src="./docs/images/wx_group.png"  width = "200" />
</div>

## Quick Start 
Quick experience of image recognition：[Link](./docs/zh_CN/tutorials/quick_start_recognition.md)

## Tutorials

- [Quick Installatiopn](./docs/zh_CN/tutorials/install.md)
- [Quick Start of Recognition](./docs/zh_CN/tutorials/quick_start_recognition.md)
- Algorithms Introduction（Updating）
    - [Backbone Network and Pre-trained Model Library](./docs/zh_CN/models/models_intro.md)
    - [Mainbody Detection](./docs/zh_CN/application/object_detection.md)
    - Image Classification
        - [ImageNet Classification](./docs/zh_CN/tutorials/quick_start_professional.md)
    - 特征学习
        - [Product Recognition](./docs/zh_CN/application/product_recognition.md)
        - [Vehicle Recognition](./docs/zh_CN/application/vehicle_reid.md)
        - [Logo Recognition](./docs/zh_CN/application/logo_recognition.md)
        - [Animation Character Recognition](./docs/zh_CN/application/cartoon_character_recognition.md)
    - [Vector Retrieval](./deploy/vector_search/README.md)
- Models Training/Evaluation
    - [Image Classification](./docs/zh_CN/tutorials/getting_started.md)
    - [Feature Learning](./docs/zh_CN/application/feature_learning.md)
- MOdel Prediction（Updating）
    - [Python Inference](./docs/zh_CN/tutorials/getting_started.md)
    - [C++ Inference](./deploy/cpp_infer/readme.md)
    - [Hub Serving Deployment](./deploy/hubserving/readme.md)
    - [Mobile Deployment](./deploy/lite/readme.md)
    - [Inference Using whl](./docs/zh_CN/whl.md)
- Advanced Tutorial
    - [Knowledge Distillation](./docs/zh_CN/advanced_tutorials/distillation/distillation.md)
    - [Model Quantization](./docs/zh_CN/extension/paddle_quantization.md)
    - [Data Augmentation](./docs/zh_CN/advanced_tutorials/image_augmentation/ImageAugment.md)
- FAQ(Suspended Updates)
    - [Image Classification FAQ](docs/zh_CN/faq.md)
- [License](#许可证书)
- [Contribution](#贡献代码)


## Introduction to Image Recognition Systems

<a name="Introduction to Image Recognition Systems"></a>
<div align="center">
<img src="./docs/images/structure.png"  width = "400" />
</div>

The steps of image recognition can be divided into three steps:
- （1）Identify region proposal for image objects through a detection model；
 - （2）Extract the image features for each region proposal;
- （3）Match the extracted features with the images in the retrieval database and output results;

For a new unknown category, there is no need to retrain the model, just fill in the retrieval database with images of that category, recreate the retrieval database, and the category can be recognised.

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
