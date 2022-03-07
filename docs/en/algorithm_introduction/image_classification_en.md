# Image Classification Task Introduction
## Catalogue

- [1. Dataset Introduction](#1)
  - [1.1 ImageNet-1k](#1.1)
  - [1.2 CIFAR-10/CIFAR-100](#1.2)
- [2. Image Classification Process](#2)
  - [2.1 Data and its Preprocessing](#2.1)
  - [2.2 Prepare the model](#2.2)
  - [2.3 Train the model](#2.3)
  - [2.4 Evaluate the model](#2.4)
- [3. Main Algorithms Introduction](#3)

Image Classification is a fundamental task that classifies the image by semantic information and assigns it to a specific label. Image Classification is the foundation of Computer Vision tasks, such as object detection, image segmentation, object tracking, and behavior analysis. Image Classification enjoys comprehensive applications, including face recognition and smart video analysis in the security and protection field, traffic scenario recognition in the traffic field, image retrieval and electronic photo album classification in the internet industry, and image recognition in the medical industry.

Generally speaking, Image Classification attempts to fully describe the whole image by feature engineering and assigns labels by a classifier. Hence, how to extract the features of images is the essential part. Before we have deep learning, the most adopted classification method is the Bag of Words model. However, Image Classification based on deep learning can learn the hierarchical feature description by supervised and unsupervised learning, replacing the manual image feature selection. Recently, Convolution Neural Network (CNN) in deep learning gives an awesome performance in the image field. It uses pixel information as the input to get all the information to the maximum extent. Additionally, since the model uses convolution to extract features, the classification result is the output. Thus, this end-to-end method performs well and is widespread.

Image Classification is a basic but important field in computer vision, whose research results have a lasting impact on the development of computer vision and even deep learning. Image classification has many sub-fields, such as multi-label image classification and fine-grained image classification. Here we only brief on the single-label image classification.


<a name="1"></a>
## 1. Dataset Introduction


<a name="1.1"></a>
### 1.1 ImageNet-1k

The ImageNet project is a large-scale visual database for the research of visual object recognition software. More than 14 million images have been annotated manually to point out objects in the picture, and at least 1 million images are provided with borders. ImageNet-1k is a subset of the ImageNet dataset, which contains 1000 categories. The training set contains 1281167 image data, and the validation set contains 50,000 image data. Since 2010, ImageNet began to hold an annual image classification competition, namely, the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) with ImageNet-1k as its specified dataset. To date, ImageNet-1k has become one of the most significant contributors to the development of computer vision, based on which numerous initial models of downstream computer vision tasks are trained.


<a name="1.2"></a>
### 1.2 CIFAR-10/CIFAR-100

The CIFAR-10 data set consists of 60,000 color images of 10 categories with an image resolution of 32x32, and each category has 6000 images, including 5000 in the training set and 1000 in the validation set. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The CIFAR-100 dataset is an extension of CIFAR-10 and consists of 60,000 color images of 100 classes with an image resolution of 32x32, and each class has 600 images, including 500 in the training set and 100 in the validation set. Researchers can try different algorithms quickly due to their small scale. These two data sets are also commonly used for testing the quality of models in image classification.


<a name="2"></a>
## 2.  Image Classification Process

The prepared training data is correspondingly preprocessed and then passed through the image classification model. The output of the model and the real label are used in a cross-entropy loss function which describes the convergence direction of the model. An image classification model can be obtained by repeatedly traversing all the image data input models, conducting the corresponding gradient descent for the final loss function through some optimizers, returning the gradient information to the model, and updating the weight of the model.


<a name="2.1"></a>
### 2.1 Data and its Preprocessing

The quality and quantity of data often determine the performance of a model. In the field of image classification, data includes images and labels. In most cases, labeled data is scarce to an extent that hard to saturate the model. In order to enable the model to learn more image features, a lot of image transformation or data augmentation is required before the image enters the model, so as to ensure the diversity of input data, hence better generalization capabilities of the model. PaddleClas provides standard image transformation for training ImageNet-1k and 8 data augmentation methods. For related codes, please refer to [Data Preprocess](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/data/preprocess)，and the configuration file to [Data Augmentation Configuration File](https://github.com/PaddlePaddle/PaddleClas/blob/develop/ppcls/configs/ImageNet/DataAugment).


<a name="2.2"></a>
### 2.2 Prepare the Model

After the data is settled, the model often determines the upper limit of the final accuracy. In the field of image classification, classic models emerge in endlessly. PaddleClas provides 36 series, or a total of 164 ImageNet pre-trained models. For specific accuracy, speed and other indicators, please refer to [Backbone Network Introduction](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models).


<a name="2.3"></a>
### 2.3 Train the Model

After preparing the data and model, you can start training the model and updating the parameters of the model. After many iterations, a trained model can finally be obtained for image classification tasks. The training process of image classification requires a lot of experience and involves the setting of many hyperparameters. PaddleClas provides a series of [training tuning methods](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models_training/train_strategy_en.md), which can help you quickly obtain a high-precision model.


<a name="2.4"></a>
### 2.4 Evaluate the Model

After a model is trained, the evaluation results of the model on the validation set can determine the performance of the model. The evaluation index is generally Top1-Acc or Top5-Acc, and the higher the index, the better the model performance.


<a name="3"></a>
## 3. Main Algorithms Introduction

- LeNet: Yan LeCun et al. first applied convolutional neural networks to image classification tasks in the 1990s, and creatively proposed LeNet, which achieved great success in handwritten digit recognition tasks.
- AlexNet: Alex Krizhevsky et al. proposed AlexNet in 2012 and applied it to ImageNet, and won the 2012 ImageNet classification competition. Since then, a deep learning boom is created.
- VGG: Simonyan and Zisserman put forward the VGG network structure in 2014. This network structure uses a smaller convolution kernel to stack the entire network, achieving better performance in ImageNet classification and providing new ideas for the subsequent network structure design.
- GoogLeNet: Christian Szegedy et al. presented GoogLeNet in 2014. This network uses a multi-branch structure and a global average pooling layer (GAP). While maintaining the accuracy of the model, the amount of model storage and calculation witnesses a drastic decrease. The network won the 2014 ImageNet classification competition.
- ResNet: Kaiming He et al. delivered ResNet in 2015, which deepened the depth of the network by introducing a residual module, reducing the recognition error rate of ImageNet classification to 3.6%, which exceeded the recognition accuracy of normal human eyes for the first time.
- DenseNet: Huang Gao et al. proposed DenseNet in 2017. The network designed a denser connected block and achieved higher performance with a smaller amount of parameters.
- EfficientNet: Mingxing Tan et al. introduced EfficientNet in 2019. This network balances the width of the network, the depth of the network, and the resolution of the input image. With the same FLOPS and parameters, it reaches the state-of-the-art results.

For more algorithm introduction, please refer to [Algorithm Introduction](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models).
