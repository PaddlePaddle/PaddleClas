# FAQ

## Before You Read

- We collect some frequently asked questions in issues and user groups since PaddleClas is open-sourced and provide brief answers, aiming to give some reference for the majority to save you from twists and turns.
- There are many talents in the field of image classification, recognition and retrieval with quickly updated models and papers, and the answers here mainly rely on our limited project practice, so it is not possible to cover all facets. We sincerely hope that the man of insight will help to supplement and correct the content, thanks a lot.

## Catalogue

- [1. 30 Questions About Image Classification](#1)
  - [1.1 Basic Knowledge](#1.1)
  - [1.2 Model Training](#1.2)
  - [1.3 Data](#1.3)
  - [1.4 Model Inference and Prediction](#1.4)
- [2. Application of PaddleClas](#2)

<a name="1"></a>
## 1. 30 Questions About Image Classification

<a name="1.1"></a>
### 1.1 Basic Knowledge

- Q: How many classification metrics are commonly used in the field of image classification?
- A:
  - For a single-label image classification (containing only 1 category and background), the evaluation metrics are Accuracy, Precision, Recall, F-score, etc. If TP(True Positive) means predicting positive class as positive, FP(False Positive) means predicting negative class as positive, TN( True Negative) means the negative class is predicted to be negative, and FN(False Negative) means the positive class is predicted to be negative. Then Accuracy=(TP + TN) / NUM, Precision=TP /(TP + FP), Recall=TP /(TP + FN).
  - For the image classification problem with the number of classes greater than 1, the evaluation metrics are Accuary and Class-wise Accuracy. Accuary indicates the percentage of the number of images correctly predicted by all classes to the total number of images; Class-wise Accuracy is obtained by calculating the Accuracy for each class of images and then averaging the Accuracy of all classes.

> >

- Q: How to choose the right training model?
- A: If you want to deploy on the server with a high requirement for accuracy but not model storage size or prediction speed, then it is recommended to use ResNet_vd, Res2Net_vd, DenseNet, Xception, etc., which are suitable for server-side models. If you want to deploy on the mobile side, then it is recommended to use MobileNetV3 and GhostNet. Meanwhile, we suggest you refer to the speed-accuracy metrics chart in [Model Library](../algorithm_introduction/ImageNet_models_en.md) when choosing models.

> >

- Q: How to initialize the parameters and what kind of initialization can speed up the convergence of the model?
- A: It is well known that the initialization of parameters can affect the final performance of the model. In general, if the target dataset is not very large, it is recommended to use the pre-trained model obtained by training ImageNet-1k for initialization. If the network is designed manually or there are no pre-trained weights based on ImageNet-1k training, you can use Xavier initialization or MSRA initialization, where the former is proposed for Sigmoid function, which is less friendly to RELU function. The deeper the network is, the smaller the variance of each layer input, the harder the network is to train. So when more RELU activation functions are used in the neural network, MSRA initialization is a better choice.

> >

- Q: What are the better solutions to the problem of parameter redundancy in deep neural networks?
- A: There are several major approaches to compressing models and reducing the model parameter redundancy, such as pruning, quantization, and knowledge distillation. Model pruning refers to removing relatively unimportant weights from the weight matrix and then fine-tuning the network again. Model quantization refers to a technique that converts floating-point computation into low-ratio specific-point computation, such as 8-bit, 4-bit, etc., which can effectively reduce the computational intensity, parameter size, and memory consumption of the model. Knowledge distillation refers to the use of a teacher model to guide a student model to learn a specific task, ensuring that the small model has a great performance improvement or even obtains similar accuracy metrics as the large model with the same number of parameters.

> >

- Q: How to choose the right classification model as a backbone network in other tasks, such as target detection, image segmentation, key point detection, etc.?

- A:

  Without considering the speed, it is most recommended to use pre-training models and backbone networks with higher accuracy. A series of SSLD knowledge distillation pre-training models are open-sourced in PaddleClas, such as ResNet50_vd_ssld, Res2Net200_vd_26w_4s_ssld, etc., which excel in both model accuracy and speed. For specific tasks, such as image segmentation or key point detection, which require higher image resolution, it is recommended to use neural network models such as HRNet that can take into account both network depth and resolution. And PaddleClas also provides  HRNet SSLD distillation series pre-training models including HRNet_W18_C_ssld, HRNet_W48_C_ssld, etc., which have very high accuracy. You can use these models and the backbone network to improve your own model accuracy on other tasks.

> >

- Q: What is the attention mechanism? What are the common methods of it?
- A: The Attention Mechanism (AM) originated from the study of human vision. Using the mechanism on computer vision tasks can effectively capture the useful regions in the images and thus improve the overall network performance. Currently, the most commonly used ones are [SE block](https://arxiv.org/abs/1709.01507), [SK-block](https://arxiv.org/abs/1903.06586), [Non-local block](https://arxiv. org/abs/1711.07971), [GC block](https://arxiv.org/abs/1904.11492), [CBAM](https://arxiv.org/abs/1807.06521), etc. The core idea is to learn the importance of feature maps in different regions or different channels, so that the network can pay more attention to the regions of salience.


<a name="1.2"></a>
### 1.2 Model Training

> >

- Q: What will happen if a model with 10 million classes is trained during the image classification with deep convolutional networks?
- A: Because of the large number of parameters in the FC layer, the memory/video memory/model storage usage will increase significantly; the model convergence speed will also be slower. In this case, it is recommended to add a layer of FC with a  smaller dimension before the last FC layer, which can drastically reduce the storage size of the model.

> >

- Q: What are the possible reasons if the model converges poorly during the training process?
- A: There are several points that can be investigated: (1) The data annotation should be checked to ensure that there are no problems with the labeling of the training and validation sets. (2) Try to adjust the learning rate (initially by a factor of 10). A learning rate that is too large (training oscillation) or too small (slow convergence) may lead to poor convergence. (3) Huge amount of data and an overly small model may prevent it from learning all the features of the data. (4) See if normalization is used in the data preprocessing process. It may be slower without normalization operation. (5) If the amount of data is relatively small, you can try to load the pre-trained model based on ImageNet-1k dataset provided in PaddleClas, which can greatly improve the training convergence speed. (6) There is a long tail problem in the dataset, you can refer to the [solution to the long tail problem of data](#long_tail).

> >

- Q: How to choose the right optimizer when training image classification tasks?
- A: Since the emergence of deep learning, there has been a lot of research on optimizers, which aim to minimize the loss function to find the right weights for a given task. Currently, the main optimizers used in the industry are SGD, RMSProp, Adam, AdaDelt, etc. Among them, since the SGD optimizer with momentum is widely used in academia and industry (only for classification tasks), most of the models we published also adopt this optimizer to achieve gradient descent of the loss function. It has two disadvantages, one is the slow convergence speed, and the other is the reliance on experiences of the initial learning rate setting. However, if the initial learning rate is set properly with a sufficient number of iterations, the optimizer will also stand out among many other optimizers, obtaining higher accuracy on the validation set. Some optimizers with adaptive learning rates, such as Adam and RMSProp, tend to converge fast, but the final convergence accuracy will be slightly worse. If you pursue faster convergence speed, we recommend using these adaptive learning rate optimizers, and SGD optimizers with momentum for higher convergence accuracy.

- Q: What are the current mainstream learning rate decay strategies? How to choose?
- A: The learning rate is the speed at which the hyperparameters of the network weights are adjusted by the gradient of the loss function. The lower the learning rate, the slower the loss function will change. While using a low learning rate ensures that no local minimal values are missed, it also means that it takes longer to converge, especially if trapped in a plateau region. Throughout the whole training process, we cannot adopt the same learning rate to update the weights, otherwise, the optimal point cannot be reached, So we need to adjust the learning rate during the training. In the initial stage of training, since the weights are in a random initialization state and the loss function decreases fast, a larger learning rate can be set. And in the later stage of training, since the weights are close to the optimal value, a larger learning rate cannot further find the optimal value, so a smaller learning rate needs is a better choice. As for the learning rate decay strategy, many researchers or practitioners use piecewise_decay (step_decay), which is a stepwise decay learning rate. In addition, there are also other methods proposed by researchers, such as polynomial_decay, exponential_ decay, cosine_decay, etc. Among them, cosine_decay requires no adjustment of hyperparameters and has higher robustness, thus emerging as the preferred learning rate decay method to improve model accuracy. The learning rates of cosine_decay and piecewise_decay are shown in the following figure. It is easy to observe that cosine_decay keeps a large learning rate throughout the training, so it is slow in convergence, but its final effect is better than peicewise_decay.

![](../../images/models/lr_decay.jpeg)

> >

- Q: What is the Warmup strategy? Where is it applied?
- A:  The warmup strategy, which, as the name implies, is a warm-up for the learning rate with no direct adoption of maximum learning rate at the beginning of training, but to train the network with a gradually increasing rate, and then decay the learning rate when it peaks. When training a neural network with a large batch_size, it is recommended to use the warmup strategy. Experiments show that warmup can steadily improve the accuracy of the model when the batch_size is large. For example, when training MobileNetV3, we set the epoch in warmup to 5 by default, i.e., first increase the learning rate from 0 to the maximum value with 5 epochs, and then conduct the corresponding decay of the learning rate.

> >

- Q: What is `batch size`？How to choose the appropriate `batch size` during training?
- A: `batch size` is an important hyperparameter in neural networks training, whose value determines how much data is fed into the neural network for training at a time. According to the paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677), when the value of `batch size` is linearly related to the value of learning rate, the convergence accuracy is almost unaffected. When training ImageNet-1k data, most of the neural networks choose an initial learning rate of 0.1 and a `batch size` of 256. Therefore, depending on the actual model size and video memory, the learning rate can be set to 0.1*k and the batch_size to 256*k. This setting can also be used as the initial parameter to further adjust the learning rate parameter and obtain better performance in real tasks.

> >

- Q: What is weight_decay？How to choose？
- A: Overfitting is a common term in machine learning, which is simply understood as a model that performs well on training data but less satisfactory on test data. In image classification, there is also the problem of overfitting, and many regularization methods are proposed to avoid it, among which weight_decay is one of the widely used ways. When using SGD optimizer, weight_decay is equivalent to adding L2 regularization after the final loss function, which makes the weights of the network tend to choose smaller values, so eventually, the parameter values in the whole network tend to be more towards 0, and the generalization performance of the model is improved accordingly. In the implementation of major deep learning frameworks, this value means the coefficient before the L2 regularization, which is called L2Decay in the PaddlePaddle framework. The larger the coefficient is, the stronger the added regularization is, and the more the model tends to be underfitted. When training ImageNet, most networks set the value of this parameter to 1e-4, and in some smaller networks such as the MobileNet series network, the value is set between 1e-5 and 4e-5 to avoid the underfitting. Of course, the setting of this value is also related to specific datasets. When the dataset of the task is large, the network itself tends to be under-fitted and the value should be reduced appropriately, and when it is small, the network itself tends to be over-fitted and the value should be increased. The following table shows the accuracy of MobileNetV1_x0_25 on ImageNet-1k using different l2_decay. Since MobileNetV1_x0_25 is a relatively small network, too large a l2_decay will tend to underfit the network, so 3e-5 is a better choice in this network compared to 1e-4.

| Model             | L2_decay | Train acc1/acc5 | Test acc1/acc5 |
| ----------------- | -------- | --------------- | -------------- |
| MobileNetV1_x0_25 | 1e-4     | 43.79%/67.61%   | 50.41%/74.70%  |
| MobileNetV1_x0_25 | 3e-5     | 47.38%/70.83%   | 51.45%/75.45%  |

> >

- Q: What does label smoothing (label_smoothing) refer to? What is the effect? What kind of scenarios does it usually apply to?
- A: Label_smoothing is a regularization method in deep learning, whose full name is Label Smoothing Regularization (LSR). In the traditional classification task, the loss function is calculated by the cross-entropy of the real one hot label and the output of the neural network, while label_smoothing is a label smoothing of the real one hot label, so that the label learned by the network is no longer a hard label, but a soft label with a probability value, where the probability at the position corresponding to the category is the largest and others small. See the paper[2] for detailed calculation methods.  In label_smoothing, the epsilon parameter describes the degree of label softening, the larger the value, the smaller the label probability value of the label vector after label smoothing, the smoother the label, and vice versa.  The value is usually set to 0.1 in experiments training ImageNet-1k, and there is a steady increase in accuracy for models of the ResNet50 size and above after using label_smooting. The following table shows the accuracy metrics of ResNet50_vd before and after using label_smoothing. At the same time, since label_smoohing can be regarded as a regularization method, the accuracy improvement is not obvious or even decreases on a relatively small model. The following table shows the accuracy metrics of ResNet18 before and after using label_smoothing on ImageNet-1k. It is clear that the accuracy drops after using label_smoothing.

| Model       | Use_label_smoothing | Test acc1 |
| ----------- | ------------------- | --------- |
| ResNet50_vd | 0                   | 77.9%     |
| ResNet50_vd | 1                   | 78.4%     |
| ResNet18    | 0                   | 71.0%     |
| ResNet18    | 1                   | 70.8%     |

> >

- Q: How to determine the tuning strategy by the accuracy or loss of the training and validation sets during training?
- A: In the process of training a network, the accuracy of the training set and validation set are usually printed for each epoch, which portrays the performance of the model on both datasets. Generally speaking, it is good to have a comparable accuracy or a slightly higher accuracy in the training set than in the validation set. If we find that the accuracy of the training set is much higher than the validation set, it means that the training set is overfitted and we need to add more regularity, such as increasing the value of L2Decay, adding more data augmentation strategies, introducing label_smoothing strategies, etc. If we find that the accuracy of the training set is lower than the validation set, it means that the training set is probably underfitted, and the regularization effect should be weakened during the training, such as reducing the value of L2Decay, decreasing the data augmentation methods, increasing the area of the crop area, weakening the image stretching, removing label_smoothing, etc.

> >

- Q: How to improve the accuracy of my own dataset by pre-training the model?
- A: At this stage, it has become a common practice in the image recognition field to load pre-trained models to train their own tasks, which can often improve the accuracy of a particular task compared to training from random initialization. In general, the pre-training model widely used in the industry is obtained by training the ImageNet-1k dataset of 1.28 million images of 1000 classes. The fc layer weights of this pre-training model are a matrix of k*1000, where k is the number of neurons before the fc layer, and it is not necessary to load the fc layer weights when loading the pre-training weights. In terms of the learning rate, if your dataset is particularly small (e.g., less than 1,000), we recommend you to adopt a small initial learning rate, e.g., 0.001 (batch_size:256, the same below), so as not to corrupt the pre-training weights with a larger learning rate. If your training dataset is relatively large (>100,000), we suggest you try a larger initial learning rate, such as 0.01 or above.

<a name="1.3"></a>
### 1.3 Data

> >

- Q: What are the general steps involved in the data pre-processing for image classification?
- A: When training ResNet50 on ImageNet-1k dataset, an image is fed into the network, and there are several steps: image decoding, random cropping, random horizontal flipping, normalization, data rearrangement, group batching and feeding into the network. Image decoding refers to reading the image file into memory; random cropping refers to randomly stretching and cropping the read image to an image with the length and width of 224 ; random horizontal flipping refers to flipping the cropped image horizontally with a probability of 0.5; normalization refers to centering the data of each channel of the image by de-meaning, so that the data conforms to the `N(0,1)` normal distribution as much as possible; data rearrangement refers to changing the data from `[224,224,3]` format to `[3,224,224]`; and group batching refers to forming a batch of multiple images and feeding them into the network for training.

> >

- Q: How does random-crop affect the performance of small model training?
- A: In the standard preprocessing of ImageNet-1k data, the random_crop function defines two values, scale and ratio, which respectively determine the size of the image crop and the degree of image stretching, where the default value of the former is 0.08-1 (lower_scale-upper_scale), and the latter is 3/4-4/3 (lower_ratio-upper_ratio). In very small networks, this kind of data augmentation can lead to network underfitting and decreased accuracy. To the end, the data augmentation can be made weaker by increasing the crop area of the image or decreasing the stretching of the image. Weaker image transformation can be achieved by increasing the value of lower_scale or reducing the difference between lower_ratio and upper_scale, respectively. The following table shows the accuracy of training MobileNetV2_x0_25 with different lower_scale, and we can see that the training accuracy and verification accuracy are improved by increasing the crop area of the images.

| Model             | Range of Scale | Train_acc1/acc5 | Test_acc1/acc5 |
| ----------------- | -------------- | --------------- | -------------- |
| MobileNetV2_x0_25 | [0.08,1]       | 50.36%/72.98%   | 52.35%/75.65%  |
| MobileNetV2_x0_25 | [0.2,1]        | 54.39%/77.08%   | 53.18%/76.14%  |

> >

- Q: What are the common data augmentation methods currently available to increase the richness of training samples when the amount of data is insufficient?
- A: PaddleClas classifies data augmentation methods into three categories, which are image transformation, image cropping and image aliasing. Image transformation mainly includes AutoAugment and RandAugment, image cropping contains CutOut, RandErasing, HideAndSeek and GridMask, and image aliasing comprises Mixup and Cutmix. More detailed introduction to data augmentation can be found in the chapter of [Data Augmentation ](../algorithm_introduction/DataAugmentation_en.md).

> >

- Q: For image classification scenarios where occlusion is common, what data augmentation methods should be used to improve the accuracy of the model?
- A: During the training, you can try to adopt cropping data augmentations including CutOut, RandErasing, HideAndSeek and GridMask on the training set, so that the model can learn not only the significant regions but also the non-significant regions, thus better performing the recognition task.

> >

- Q: What data augmentation methods should be used to improve model accuracy in the case of complex color transformations?
- A:  Consider using the data augmentation strategies of AutoAugment or RandAugment, both of which include rich color transformations such as sharpening and histogram equalization, allowing the model to be more robust to these transformations during the training process.

> >

- Q: How do Mixup and Cutmix work? Why are they effective methods of data augmentation?
- A: Mixup generates a new image by linearly overlaying two images, and the corresponding labels also undertake the same process for training, while Cutmix crops a random region of interest (ROI) from an image and overlays the corresponding region in the current image, and the labels are linearly overlaid in proportion to the image area. They actually generate different samples and labels from the training set and for the learning of the network, thus enriching the samples.

> >

- Q: What is the size of the training dataset for an image classification task that does not require high accuracy?
- A: The amount of the training data is related to the complexity of the problem to be solved. The greater the difficulty and the higher the accuracy requirement, the larger the dataset needs to be. And it is a universal rule that the more training data the better the result in practice. Of course, in general, 10-20 images per category with pre-trained models can guarantee the basic classification effect; or at least 100-200 images per category without pre-training models.



> >
<a name="long_tail"></a>
- Q: What are the common methods currently used for datasets with long-tailed distributions?
- A:(1) the categories with fewer data can be resampled to increase the probability of their occurrence; (2) the loss can be modified to increase the loss weight of images in categories corresponding to fewer images; (3) the method of transfer learning can be borrowed to learn generic knowledge from common categories and then migrate to the categories with fewer samples.


<a name="1.4"></a>
### 1.4 Model Inference and Prediction

> >

- Q: How to deal with the poor recognition performance when the original image is taken for classification with only a small part of the image being the foreground object of interest?
- A: A mainbody detection model can be added before classification to detect the foreground objects, which can greatly improve the final recognition results. If time cost is not a concern, multi-crop can also be used to fuse all the predictions to determine the final category.

> >

- Q: What are the currently recommended model inference methods?
- A: After the model is trained, it is recommended to use the exported inference model to make inferences based on the Paddle inference engine, which currently supports python inference and cpp inference. If you want to deploy the inference model based on service, it is recommended to use the PaddleServing.

> >

- Q: What are the appropriate inference methods to further improve the model accuracy after training?
- A:(1) A larger inference scale can be used, e.g., 224 for training, then 288 or 320 for inference, which will directly bring about a 0.5% improvement in accuracy. (2) Test Time Augmentation (TTA) can be used to create multiple copies of the test set by rotating, flipping, color transforming, and so on, and then fuse all the prediction results, which can greatly improve its accuracy and robustness. (3) Of course, a multi-model fusion strategy can also be adopted to fuse the prediction results of multiple models for the same images.

> >

- Q: How to choose the best model for the fusion of multiple models?
- A: Without considering the inference speed, models with the highest possible are recommended; it is also suggested to choose models with different structures or series for fusion. For example, the model fusion results of ResNet50_vd and Xception65 tend to be better than those of ResNet50_vd and ResNet101_vd with similar accuracy.

> >

- Q: What are the common acceleration methods when using a fixed model for inference?
- A: (1) Using a GPU with better performance; (2) increasing the batch size; (3) using TenorRT and FP16 half-precision floating-point methods.


<a name="2"></a>
## 2. Application of PaddleClas

> >

- Q: Why can't I import parameters even though I have specified the address of the folder where the pre-trained model is located during evaluation and inference?
- A: When loading a pretrained model, you need to specify the prefix of it. For example, if the folder where the pretrained model parameters are located is `output/ResNet50_vd/19` and the name of the pretrained model parameters is `output/ResNet50_vd/19/ppcls.pdparams`, then the `pretrained_model ` parameter needs to be specified as `output/ResNet50_vd/19/ppcls`, and PaddleClas will automatically complete the `.pdparams` suffix.

> >

- Q: Why the final accuracy is always about 0.3% lower than the official one when evaluating the `EfficientNetB0_small` model?
- A: The `EfficientNet` series network uses `cubic interpolation` for resize (the interpolation value of the resize parameter is set to 2), while other models are None by default, so the interpolation value of resize needs to be explicitly specified during training and evaluation. Specifically, you can refer to the ResizeImage parameter in the preprocessing process in the following configuration.

```
  Eval:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/val_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
            interpolation: 2
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
```

> >

- Q: Why `TypeError: __init__() missing 1 required positional argument: 'sync_cycle'` is reported when using visualdl under python2？
- A: Currently visualdl only supports running under python3 with a required version of 2.0 or higher. If visualdl is not the right version, you can install it as follows: `pip3 install visualdl -i https://mirror.baidu.com/pypi/simple`

> >

- Q: Why is it that the inference speed of a single image by ResNet50_vd is much lower than the benchmark provided by the official website while the CPU is much faster than GPU?
- A: The model inference needs to be initialized, and it is time-consuming. Therefore, when counting the inference speed, we need to run a batch of images, remove the inference time of the first few images, and then count the average time.GPU is slower than CPU to test a single image because the initialization of GPU is much slower than CPU.

> >

- Q: Can grayscale maps be used for model training?
- A: The grayscale image can also be used for model training, but the input shape of the model needs to be modified to `[1, 224, 224]`, and the data augmentation also needs to be adapted. However, for better use of the PaddleClas code, it is recommended to adapt the grayscale image to a 3-channel image for training (RGB channels have equal pixel values).

> >

- Q: How to train the model on windows or cpu?
- A: You can refer to [Getting Started Tutorial](../models_training/classification_en.md) for detailed tutorials on model training, evaluation and inference in Linux , Windows, CPU, and other environments.

> >

- Q: How to use label smoothing in model training?
- A:  This can be set in the `Loss` field in the configuration file as follows. `epsilon=0.1` means set the value to 0.1, if the `epsilon` field is not set, then `label smoothing` will not be used.

```
Loss:
  Train:
    - CELoss:
        weight: 1.0
        epsilon: 0.1
```

> >

- Q: Is the 10W class image classification pre-training model provided by PaddleClas available for model inference?
- A: This 10W class image classification pre-training model does not provide parameters for the fc fully connected layer, which cannot be used for model inference but is available for model fine-tuning at present.

> >

- Q: Why is `Error: Pass tensorrt_subgraph_pass has not been registered` reported  When using `deploy/python/predict_cls.py` for model prediction?
- A: If you want to use TensorRT for model prediction and inference, you need to install or compile PaddlePaddle with TensorRT by yourself.  For Linux, Windows, macOS users, you can refer to [download inference library](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html). If there is no required version, you need to compile and install it locally, which is detailed in [source code compilation](https://paddleinference.paddlepaddle.org.cn/user_guides/source_compile.html).

> >

- Q: How to train with Automatic Mixed Precision (AMP) during training?
- A: You can refer to [ResNet50_amp_O1.yaml](../../../ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1.yaml). Specifically, if you want your configuration file to support automatic mixed precision during model training, you can add the following information to the file.

```
# mixed precision training
AMP:
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  use_pure_fp16: &use_pure_fp16 True
```
