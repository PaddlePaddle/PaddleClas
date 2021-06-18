# Logo Recognition

Logo recognition is a field that is widely used in real life, such as whether the Adidas or Nike logo appears in a photo, or whether the Starbucks or Coca-Cola logo appears on a cup. Usually, when the number of logo categories is large, the two-stage method of detection and recognition is often used. The detection module is responsible for detecting the potential logo area, and then feed the logo area to the recognition module to identify the category. The recognition module mostly adopts retrieval-based method, and sorts the similarity of the query and the gallery to obtain the predicted category. This document mainly introduces the feature learning part.

## 1 Pipeline

See the pipline of [feature learning](./feature_learning_en.md) for details.

The config file of logo recognition: [ResNet50_ReID.yaml](../../../ppcls/configs/Logo/ResNet50_ReID.yaml).

The details are as follows.

### 1.1 Data Augmentation

Different from classification, this part mainly uses the following methods:

- `Resize` to 224.  The input image is already croped using bbox by a logo detector.
- [AugMix](https://arxiv.org/abs/1912.02781v1)：Simulate lighting changes, camera position changes and other real scenes.
- [RandomErasing](https://arxiv.org/pdf/1708.04896v2.pdf)：Simulate occlusion.

### 1.2 Backbone

Using `ResNet50` as backbone, and make the following modifications:

- Last stage stride = 1, keep the size of the final output feature map to 14x14. At the cost of increasing a small amount of calculation, the ability of feature representation is greatly improved.
- Use pretrained weights of ImageNet

code：[ResNet50_last_stage_stride1](../../../ppcls/arch/backbone/variant_models/resnet_variant.py)

### 1.3 Neck

In order to reduce the complexity of calculating feature distance in inference, an embedding convolution layer is added, and the feature dimension is set to 512.

### 1.4 Metric Learning Losses

[PairwiseCosface](../../../ppcls/loss/pairwisecosface.py) , [CircleMargin](../../../ppcls/arch/gears/circlemargin.py) [1] are used. The weight ratio of two losses is 1:1.

## 2 Experiment

<img src="../../images/logo/logodet3k.jpg" style="zoom:50%;" />

LogoDet-3K[2] dataset is used for experiments. The dataset is fully labeled, with 3000 logo categories, about 200,000 high-quality manually labeled logo objects and 158,652 images.

Since the dataset is original desigined for detection task, only the cropped logo area is used in the logo recognition stage. Therefore, the labeled bbox annotations are used to crop the logo area to form the training set, eliminating the influence of the background in the recognition stage. After cropping preprocessing, the dataset was splited to 155,427 images as training sets, covering 3000 logo categories (also used as the gallery during testing), and 3225 as test sets, which were used as query sets. The cropped dataset is available [download here](https://arxiv.org/abs/2008.05359)

On this data, the single model Recall@1 Acc: 89.8%.

## 3 References

[1] Circle loss: A unified perspective of pair similarity optimization. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.

[2] LogoDet-3K: A Large-Scale Image Dataset for Logo Detection[J]. arXiv preprint arXiv:2008.05359, 2020.
