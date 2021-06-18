# Image Classification
---

Image Classification is a fundamental task that classifies the image by semantic information and assigns it to a specific label. Image Classification is the foundation of Computer Vision tasks, such as object detection, image segmentation, object tracking and behavior analysis. Image Classification has comprehensive applications, including face recognition and smart video analysis in the security and protection field, traffic scenario recognition in the traffic field, image retrieval and electronic photo album classification in the internet industry, and image recognition in the medical industry.  

Generally speaking, Image Classification attempts to comprehend an entire image as a whole by feature engineering and assigns labels by a classifier. Hence, how to extract the features of image is the essential part. Before we have deep learning, the most used classification method is the Bag of Words model. However, Image Classification based on deep learning can learn the hierarchical feature description by supervised and unsupervised learning, replacing the manually image feature selection. Recently, Convolution Neural Network in deep learning has an awesome performance in the image field. CNN uses the pixel information as the input to get the all information to the maximum extent. Additionally, since the model uses convolution to extract features, the classification result is the output. Thus, this kind of end-to-end method achieves ideal performance and is applied widely. 

Image Classification is a very basic but important field in the subject of computer vision. Its research results have always influenced the development of computer vision and even deep learning. Image classification has many sub-fields, such as multi-label image classification and fine-grained image classification. Here is only a brief description of single-label image classification.


## 1 Dataset Introduction

### 1.1 ImageNet-1k

The ImageNet project is a large-scale visual database for the research of visual object recognition software. More than 14 million images have been annotated manually to point out objects in the picture in this project, and at least more than 1 million images provide borders. ImageNet-1k is a subset of the ImageNet dataset, which contains 1000 categories. The training set contains 1281167 image data, and the validation set contains 50,000 image data. Since 2010, the ImageNet project has held an image classification competition every year, which is the ImageNet Large-scale Visual Recognition Challenge (ILSVRC). The dataset used in the challenge is ImageNet-1k. So far, ImageNet-1k has become one of the most important data sets for the development of computer vision, and it promotes the development of the entire computer vision. The initialization models of many computer vision downstream tasks are based on the weights trained on this dataset.

### 1.2 CIFAR-10/CIFAR-100

The CIFAR-10 data set consists of 60,000 color images in 10 categories, with an image resolution of 32x32, and each category has 6000 images, including 5000 in the training set and 1000 in the validation set. 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships and trucks. The CIFAR-100 data set is an extension of CIFAR-10. It consists of 60,000 color images in 100 classes, with an image resolution of 32x32, and each class has 600 images, including 500 in the training set and 100 in the validation set. Researchers can try different algorithms quickly because these two data sets are small in scale. These two data sets are also commonly used data sets for testing the quality of models in the image classification field.

## 2 Image Classification Process

The prepared training data is preprocessed by the corresponding data and then passed through the image classification model. The output of the model and the real label are used in a cross-entropy loss function. This loss function describes the convergence direction of the model.  Traverse all the image data input model, do the corresponding gradient descent for the final loss function through some optimizers, return the gradient information to the model, update the weight of the model, and traverse the data repeatedly. Finally, an image classification model can be obtained.

### 2.1 Data and its preprocessing

The quality and quantity of data often determine the performance of a model. In the field of image classification, data includes images and labels. In most cases, labeled data is scarce, so the amount of data is difficult to reach the level of saturation of the model. In order to enable the model to learn more image features, a lot of image transformation or data augmentation is required before the image enters the model, so as to ensure the diversity of input image data. Ultimately ensure that the model has better generalization capabilities. PaddleClas provides standard image transformation for training ImageNet-1k, and also provides 8 data augmentation methods. For related codes, please refer to[data preprocess](../../../ppcls/data/preprocess)，The configuration file refer to [Data Augmentation Configuration File](../../../ppcls/configs/ImageNet/DataAugment).

### 2.2 Prepare the model

After the data is determined, the model often determines the upper limit of the final accuracy. In the field of image classification, classic models emerge in an endless stream. PaddleClas provides 35 series and a total of 164 ImageNet pre-trained models. For specific accuracy, speed and other indicators, please refer to[Backbone network introduction](../models)。

### 2.3 Train the model

After preparing the data and model, you can start training the model and update the parameters of the model. After many iterations, a trained model can finally be obtained for image classification tasks. The training process of image classification requires a lot of experience and involves the setting of many hyperparameters. PaddleClas provides a series of [training tuning methods](../models/Tricks_en.md), which can quickly help you obtain a high-precision model.

### 2.4 Evaluate the model

After a model is trained, the evaluation results of the model on the validation set can determine the performance of the model. The evaluation index is generally Top1-Acc or Top5-Acc. The higher the index, the better the model performance.


## 3 Main Algorithms Introduction 

- LeNet: Yan LeCun et al. first applied convolutional neural networks to image classification tasks in the 1990s, and creatively proposed LeNet, which achieved great success in handwritten digit recognition tasks.

- AlexNet: Alex Krizhevsky et al. proposed AlexNet in 2012 and applied it to ImageNet, and won the 2012 ImageNet classification competition. Since then, deep learning has become popular

- VGG: Simonyan and Zisserman proposed the VGG network structure in 2014. This network structure uses a smaller convolution kernel to stack the entire network, achieving better performance in ImageNet classification, it provides new ideas for the subsequent network structure design.

- GoogLeNet: Christian Szegedy et al. proposed GoogLeNet in 2014. This network uses a multi-branch structure and a global average pooling layer (GAP). While maintaining the accuracy of the model, the amount of model storage and calculation is greatly reduced. The network won the 2014 ImageNet classification competition.

- ResNet: Kaiming He et al. proposed ResNet in 2015, which deepened the depth of the network by introducing a residual module. In the end, the network reduced the recognition error rate of ImageNet classification to 3.6%, which exceeded the recognition accuracy of normal human eyes for the first time.

- DenseNet: Huang Gao et al. proposed DenseNet in 2017. The network designed a denser connected block and achieved higher performance with a smaller amount of parameters.

- EfficientNet: Mingxing Tan et al. proposed EfficientNet in 2019. This network balances the width of the network, the depth of the network, and the resolution of the input image. With the same FLOPS and parameters, the accuracy reaches the state-of-the-art.

For more algorithm introduction, please refer to [Algorithm Introduction](../models).
