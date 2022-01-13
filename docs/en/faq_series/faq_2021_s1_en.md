# Image Classification FAQ Summary - 2021 Season 1

## Catalogue

- [1. Issue 1](#1)(2021.01.05)
- [2. Issue 2](#2)(2021.01.14)
- [3. Issue 3](#3)(2020.01.21)
- [4. Issue 4](#4)(2021.01.28)
- [5. Issue 5](#5)(2021.02.03)

<a name="1"></a>

## Issue 1

### Q1.1: Why is the prediction accuracy of the exported inference model very low ?

**A**：You can check the following aspects:

- Check whether the path of the pre-training model is correct or not.
- The default class number is 1000 when exporting the model. If the pre-training model has a custom class number, you need to specify the parameter `--class_num=k` when exporting, k is the custom class number.
- Compare the output class id and score of `tools/infer/infer.py` and `tools/infer/predict.py` for the same input. If they are exactly the same, the pre-trained model may have poor accuracy itself.

### Q1.2: How to deal with the unbalanced categories of training samples?

**A**：There are several commonly used methods.

- From the perspective of sampling
  - The samples can be sampled dynamically according to the categories, with different sampling probabilities for each category and ensure that the number of training samples in different categories is basically the same or in the desired proportion in the same minibatch or epoch.
  - You can use the oversampling method to oversample the categories with a small number of images.
- From the perspective of loss function
  - The OHEM (online hard example miniing) method can be used to filter the hard example based on the loss of the samples for gradient backpropagation and parameter update of the model.
  - The Focal loss method can be used to assign a smaller weight to the loss of some easy samples and a larger weight to the loss of hard samples, so that the loss of easy samples contributes to the overall loss of the network without dominating the loss.

### Q1.3 When training in docker, the data path and configuration are fine, but it keeps reporting `SystemError: (Fatal) Blocking queue is killed because the data reader raises an exception`, why is this?

**A**：

This may be caused by the small shared memory in docker. When creating docker, the default size of `/dev/shm` is 64M, if you use multiple processes to read data, the shared memory may fall short, so you need to allocate more space to `/dev/shm`. When creating a docker, input `--shm-size=8g` to allocate 8G to `/dev/shm`, which is usually enough.

### Q1.4 Where can I download the 10W class image classification pre-training model provided by PaddleClas and how to use it?

**A**：

Based on ResNet50_vd, Baidu open-sourced its own large-scale classification pre-training model with 100,000 categories and 43 million images. The former is available for download at [download address](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_10w_pretrained.tar), where it should be noted that the pre-training model does not provide the final FC layer parameters and thus cannot be used directly for inference; however, it can be used as a pre-training model to fine-tune it on your own dataset. It is verified that this pre-training model has a more significant accuracy gain of up to 30% on different datasets than the ResNet50_vd pre-training model based on the ImageNet1k dataset.

### Q1.5 How to accelerate when using C++ for inference deployment?

**A**：You can speed up the inference in the following ways.

1. For CPU inference, you can turn on mkldnn and increase the number of threads (cpu_math_library_num_threads, in `tools/config.txt`), which is usually 6~10.
2. For GPU inference, you can enable TensorRT inference and FP16 inference if the hardware condition allows, which can further speed up the speed.
3. If the memory or video memory is sufficient, the batch size of inference can be increased.
4. The image preprocessing logic (mainly designed for resize, crop, normalize, etc.) can be run on the GPU, which can further speed up the process.

You are welcomed to add more tips on inference deployment acceleration.

<a name="2"></a>

## Issue 2

### Q2.1: Does PaddleClas have to start from 0 when setting labels, and does class_num have to equal the number of classes in the dataset?

**A**：

In PaddleClas, the label start from 0 by default, so try to set the label from 0. Of course, it is possible to set them from other values, which will result in a larger class_num and will in turn lead to a larger number of FC layer parameters for the classification, so the weight file will take up more storage space. In the case of a continuous set of classes, set class_num equal to the number of classes in the dataset (of course, it is acceptable to set it greater than the number of classes in the dataset, and even higher accuracy can be obtained in many datasets, but it will also result in a larger number of FC layers), and in the case of a discontinuous set of classes, the class_num should be equal to the largest class_id+1 in the dataset.

### Q2.2: How to address the the problem of large space occupation of weight file resulted from great FC due to numerous class number?

**A**：

The final FC weight is a large matrix of size C*class_num, where C is the number of neural units in the previous layer of FC, e.g., C is 2048 in ResNet50, and the size of FC weight can be further reduced by decreasing the value of C. For example, a layer of FC with smaller dimension can be added after GAP, which can greatly reduce the weight size of the final classification layer.

### Q2.3: Why did the training of ssld distillation on a custom dataset using PaddleClas fail to meet the expectation?

First, it is necessary to ensure that the accuracy of the Teacher model. Second, it is necessary to ensure that the Student model is successfully loaded with the pre-training weights of ImageNet-1k and the Teacher model is successfully loaded with the weights of the training custom dataset. Finally, it is necessary to ensure that the initial learning rate is not too large, or at least smaller than the value of the ImageNet-1k training.

### Q2.4: Which networks have advantages on mobile or embedded side?

It is recommended to use the Mobile Series network, and the details can be found in [Introduction to Mobile Series Network Structure](../models/Mobile_en.md). If the speed of the task is the priority, MobileNetV3 series can be considered, and if the model size is more important, the specific structure can be determined based on the StorageSize - Accuracy in the Introduction to the Mobile Series Network Structure.

### Q2.5: Why use a network with large number of parameters and computation such as ResNet when the mobile network is so fast?

Different network structures have various speed advantages running on disparate devices. On the mobile side, mobile series networks run faster than server-side networks, but on the server side, networks with specific optimizations such as ResNet have greater advantages for the same accuracy. So the specific network structure needs to be decided on a case-by-case basis.

<a name="3"></a>

## Issue 3

### Q3.1: What are the characteristics of a double (multi)-branch structure and a Plain structure, respectively?

**A**：

Plain networks, represented by VGG, have evolved into multi-branch network structures, represented by the ResNet series (with residual modules) and the Inception series (with multiple convolutional kernels in parallel). It is found that the multi-branch structure is more friendly in the model training, and the larger network width can bring stronger feature fitting ability, while the residual structure can avoid the problem of disappearing gradient of the deep network. However, in the inference phase, the model with a multi-branch structure has no speed advantage. Even though the FLOPs of the model are lower, the computational density of the multi-branch structure model is also dissatisfactory. For example, the FLOPs of VGG16 model are much larger than those of EfficientNetB3, but the inference speed of the latter is significantly faster than the former. Therefore, the multi-branch structure is more friendly in the model training, while the Plain structure model is more suitable for the inference. Starting from this, we can use a multi-branch network structure in the training phase to exchange a larger training time cost for a model with better feature fitting ability, and convert the multi-branch structure to Plain structure in the inference phase to exchange a shorter inference time. The conversion from multi-branch to a Plain structure can be achieved by the structural re-parameterization technique.

In addition, the Plain structure is more friendly for pruning operations.

Note: The term "Plain structure" and "structural re-parameterization" are from the paper "RepVGG: Making VGG-style ConvNets Great Again". Plain structure network model means that there is no branching structure in the whole network, i.e., the input of layer `i` is the output of layer `i-1` and the output of layer `i` is the input of layer `i+1`.

### Q3.2: What are the main innovations of ACNet?

**A**： ACNet means "Asymmetric Convolution Block", and the idea is from the paper "ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks", which proposes a set of three CNN convolutional kernels with "ACB" structure to replace the traditional square convolutional kernels in existing convolutional neural networks in the training phase.

The size of the square convolution kernel is assumed to be `d*d`, i.e., the width and height are all `d`. The ACB structure used to replace this convolution kernel is three convolution kernels of size `d*d`, `1*d`, and `d*1`, and then the outputs of the three convolution kernels are added directly to obtain the same size as the original square convolution kernel. After the training, the ACB structure is replaced by the original square convolution kernel, and the parameters of which are directly added to the parameters of the three convolution kernels of the ACB structure (see `Q3.4`), so the same model structure as before is used for inference, and the ACB structure is only used in the training phase.

During the training, the network width of the model is improved by the ACB structure, and more features are extracted using the two asymmetric convolution kernels of `1*d` and `d*1` to enrich the information of the feature maps extracted by the `d*d` convolution kernels. In the inference stage, this design idea does not bring additional parameters and computational overhead. The following figure shows the form of convolutional kernels for the training phase and the inference deployment phase, respectively.

![](../../images/faq/TrainingtimeACNet.png)

![](../../images/faq/DeployedACNet.png)

Experiments by the authors of the article show that the model capability can be significantly improved by using ACNet structures in the training of the original network model, as explained by the original authors as follows.

1. Experiments show that for a `d*d` convolution kernel, the parameters of the skeleton position (e.g., the `skeleton` position of the convolution kernel in the above figure) have a greater impact on the model accuracy than the parameters of the corner position (e.g., the `corners` position of the convolution kernel in the above figure) of the eliminated convolution kernel, so the parameters of the skeleton position of the convolution kernel are essential. And the two asymmetric convolution kernels in the ACB structure enhance the weight of the square convolution kernel skeleton position parameter, making it more significant. About whether this summation will weaken the role of the skeleton position parameter due to the offsetting effect of positive and negative numbers, the authors found through experiments that the training of the network always goes in the direction of increasing the role of the skeleton position parameter, and there is no weakening due to the offsetting effect.
2. The asymmetric convolution kernel is more robust for flipped images, as shown in the following figure, the horizontal asymmetric convolution kernel is more robust for flipped images up and down. The feature maps extracted by the asymmetric convolution kernel are the same for semantically the same position in the image before and after the flip, which is better than the square convolution kernel.

![](../../images/faq/HorizontalKernel.png)

### Q3.3: What are the main innovations of RepVGG?

**A**：

Through Q3.1 and Q3.2, it may occur to us to decouple the training phase and inference phase by ACNet, and use multi-branch structure in the training phase and Plain structure in inference phase, which is the innovation of RepVGG. The following figure shows the comparison of the network structures of ResNet and RepVGG in the training and inference phases.

![](../../images/faq/RepVGG.png)

First, the RepVGG in the training phase adopts a multi-branch structure, which can be regarded as a residual structure with `1*1` convolution and constant mapping on top of the traditional VGG network, while the RepVGG in the inference phase degenerates to a VGG structure. The transformation of the network structure from RepVGG in the training phase to RepVGG in the inference phase is implemented using the "structural reparameterization" technique.

The constant mapping can be regarded as the output of the `1*1` convolution kernel with `1` parameters acting on the input feature map, so the convolution module of RepVGG in the training phase can be regarded as two `1*1` convolutions and one `3*3` convolution, and the parameters of the `1*1` convolution can be directly added to the parameters at the center of the `3*3` convolution kernel (this operation is similar to the operation of adding the parameters of the asymmetric convolution kernel to the parameters of the skeleton position of the square convolution kernel in ACNet).  By doing this, the constant mapping, `1*1` convolution, and `3*3` convolution branches of the network structure can be combined into one `3*3` convolution in the inference stage, as described in `Q3.4`.

### Q3.4: What are the similarities and differences between the struct re-parameters in ACNet and RepVGG?

**A**：

From the above, it can be simply understood that RepVGG is the extreme version of ACNet. Re-parameters operation in ACNet is shown in the following figure.

![](../../images/faq/ACNetReParams.png)

Take `conv2` as an example, the asymmetric convolution can be regarded as a square convolution kernel of `3*3`, except that the upper and lower six parameters of the square convolution kernel are `0`, and it is the same for `conv3`. On top of that, the sum of the results of `conv1`, `conv2`, and `conv3` is equivalent to the sum of three convolution kernels followed by convolution. With `Conv` denoting the convolution operation and `+` denoting the addition operation of the matrix, then: `Conv1(A)+Conv2(A)+Conv3(A) == Convk(A)`, where `Conv1`, ` Conv2`, `Conv3` have convolution kernels `Kernel1`, `kernel2`, `kernel3`, and `Convk` has convolution kernels `Kernel1 + kernel2 + kernel3`, respectively.

The RepVGG network is the same as ACNet, except that the `1*d` asymmetric convolution of ACNet becomes the `1*1` convolution, and the position where the `1*1` convolutions are summed becomes the center of the `3*3` convolution.

### Q3.5: What are the factors that affect the computation speed of a model? Does a model with a larger number of parameters necessarily have a slower computation speed?

**A**：

There are many factors that affect the computation speed of the model, and the number of parameters is only one of them. Specifically, without considering the hardware differences, the computation speed of the model can be referred to the following aspects.

1. Number of parameters: the number of parameters used to measure the model, the larger the number of parameters of the model, the higher the memory (video memory) requirements of the model during computation. However, the size of the memory (video memory) footprint does not depend entirely on the number of parameters. In the figure below, assuming that the input feature map memory footprint size is `1` unit, for the residual structure on the left, the peak memory footprint during computation is twice as large as that of the Plain structure on the right, because the results of the two branches need to be recorded and then added together.

![](../../images/faq/MemoryOccupation.png)

2. FLOPs: Note that FLOPs are distinguished from floating point operations per second (FLOPS), which can be simply understood as the amount of computation and is usually adopted to measure the computational complexity of a model. Taking the common convolution operation as an example, considering no batch size, activation function, stride operation, and bias, assuming that the input future map size is `Min*Min` and the number of channels is `Cin`, the output future map size is `Mout*Mout` and the number of channels is `Cout`, and the conv kernel size is `K*K`, the FLOPs for a single convolution can be calculated as follows.
   1. The number of feature points contained in the output feature map is: `Cout * Mout * Mout`.
   1. For the convolution operation for each feature point in the output feature map: the number of multiplication calculations is: `Cin * K * K`; the number of addition calculations is: `Cin * K * K - 1`.
   1. So the total number of computations is: `Cout * Mout * Mout * (Cin * K * K + Cin * K * K - 1)`, i.e. `Cout * Mout * Mout * (2Cin * K * K - 1)`.

3. Memory Access Cost (MAC): The computer needs to read the data from memory (general memory, including video memory) to the operator's Cache before performing operations on data (such as multiplication and addition), and the memory access is very time-consuming. Take grouped convolution as an example, suppose it is divided into `g` groups, although the number of parameters and FLOPs of the model remain unchanged after grouping, the number of memory accesses for grouped convolution becomes `g` times of the previous one (this is a simple calculation without considering multi-level Cache), so the MAC increases significantly and the computation speed of the model slows down accordingly.

4. Parallelism: The term parallelism often includes data parallelism and model parallelism, in this case, model parallelism. Take convolutional operation as an example, the number of parameters in a convolutional layer is usually very large, so if the matrix in the convolutional layer is chunked and then handed over to multiple GPUs separately, the purpose of acceleration can be achieved. Even some network layers with too many parameters for a single GPU memory may be divided into multiple GPUs, but whether they can be divided into multiple GPUs in parallel depends not only on hardware conditions, but also on the specific form of operation. Of course, the higher the degree of parallelism, the faster the model can run.



<a name="4"></a>

## Issue 4

### Q4.1: There is certain synthetic data in image classification task, is it necessary to use sample equalization?

**A**:

1. If the number of samples of different categories varies greatly, and the samples of one category are expanded to more than several times of other categories due to the synthetic data set, it is necessary to reduce the weights of that category appropriately.
2. If some categories are synthetic and some are semi-synthetic and semi-real, the equalization is not needed as long as the number is in an order of magnitude. Or you can try to train it, testing whether the synthetic category samples can be accurately identified.
3. If the performance of the category of different sources of data is degraded due to the increase of synthetic data, it is necessary to consider whether the synthetic dataset has noise or difficult samples. You can also properly increase the weight of the category to obtain better recognition performance of the category.

### Q4.2: What new opportunities and challenges will be brought by the introduction of Vision Transformer (ViT) into the field of image classification by academia? What are the advantages over CNN?

Paper address: [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://openreview.net/pdf?id=YicbFdNTTy)

**A**:

1. The dependence of images on CNNs is unnecessary, and the computational efficiency and scalability of the Transformer allow for training very large models without saturation as the model and dataset grow. Inspired by the Transformer for NLP, when being used in image classification tasks, images are divided into sequential patches that are fed into a linear unit embedding as input to the transformer.
2. In medium-sized datasets such as ImageNet1k, ImageNet21k, the visual Transformer model is several percentage points lower than ResNet of the same size. It is speculated that this is because the transformer lacks the Locality and Spatial Invariance that CNNs have, and it is difficult to outperform convolutional networks when the amount of data is not large enough. But for this problem, the data augmentation adopted by [DeiT](https://arxiv.org/abs/2012.12877) to some extent addresses the reliance of Vision Transformer on very large datasets for training.
3. This approach can go beyond local information and model more long-range dependencies when training on super large-scale datasets 14M-300M, while CNN can better focus on local information but is weak in capturing global information.
4. Transformer once reigned in the field of NLP, but was also questioned as not applicable to the CV field. The current several pieces of visual field articles also deliver competitive performance as the SOTA of CNN. We believe that a joint Vision-Language or multimodal model will be proposed that can solve both visual and linguistic problems.

### Q4.3: For the Vision Transformer model, how is the image converted into sequence information for the Encoder?

**A**:

1. Using the Transformer model, mainly the attention approach. We want to conceive a scenario where semantic embedding information is applicable. But image classification is not very relevant to the semantic information of sequences, so Vision Transformer has its own unique design, and it is precisely the goal of ViT to use attention mechanisms instead of CNNs.
2. Consider the input form of the Encoder in Transformer, as shown below:
   - (1) variable-length sequential input, because it is RNN structure with various amounts of words in one sentence. If it is an NLP scene, the change of word order affects little of the semantics, but the position of the image means a lot since great misunderstanding can be caused when different regions are connected in a different order.
   - (2) Single patch position is transformed into a vector with fixed dimension. Encoder input is patch pixel information embedding, combined with some fixed position vector concate to synthesize a vector with fixed dimension and position information in it.

![](../../images/faq/Transformer_input.png)

3. Consider the following question: How to pass an image to an encoder?

- As the following figure shows. Suppose the input image is [224,224,3], which is cut into many patches in order from left to right and top to bottom, and the patch size can be [p,p,3] (p can be 16, 32). Convert it into a feature vector using the Linear Projection of Flattened Patches module and concat a position vector into the Encoder.

![](../../images/faq/ViT_structure.png)

4. As shown above, given an image of `H×W×C` and a block size P, the image can be divided into `N` blocks of `P×P×C`, `N=H×W/(P×P)`. After getting the blocks, we have to use linear transformation to convert them into D-dimensional feature vectors, and then add the position encoding vectors. Similar to BERT, ViT also adds a classification flag bit before the sequence, denoted as `[CLS]`. The ViT input sequence `z` is shown in the following equation, where `x` represents an image block.

![](../../images/faq/ViT.png)

5. ViT model is basically the same as Transformer, where the input sequence is passed into ViT and then the final output features are classified using the `[CLS]` flags. viT consists mainly of MSA (multiheaded self-attentive) and MLP (two-layer fully connected network using GELU activation function), with LayerNorm and residual connections before MSA and MLP

### Q4.4: How to understand Inductive Bias?

**A**:

1. In machine learning, some assumptions are made about the problem to be applied, and this assumption is called inductive preference. Certain a priori rules are inducted from the phenomena observed in real life, and then certain constraints are made on the model, thus playing the role of model selection. In CNN, it is assumed that the features have the characteristics of Locality and Spatial Invariance, that is, the adjacent features are connected but not those that are far away, and the adjacent features are fused together to produce "solutions" more easily; there is also the attention mechanism, which is the rule inducted from human intuition and life experience.
2. Vision Transformer utilizes an inductive bias that is linked to Sequentiality and Time Invariance, i.e., the time interval in sequence order, and therefore also yields better performance than CNN-like models on larger datasets. In the Conclusion of the article, "Unlike prior works using self-attention in computer vision, we do not introduce any image-specific inductive biases into the architecture" and "We find that large scale training trumps inductive bias" in the Introduction, we can conclude that intuitively the generation of inductive bias in the case of large amounts of data is performance-degrading and should be discarded whenever possible.

### Q4.5:  Why does ViT add a [CLS] flag bit? Why is the vector corresponding to the [CLS] used as the semantic representation of the whole sequence?

**A**:

1. Similar to BERT, ViT adds a `[CLS]` flag bit before the first patch, and the vector corresponding to the last end flag bit can be used as a semantic representation of the whole image, and thus for downstream classification tasks, etc. Therefore, the whole embedding group can characterize the features of the image at different locations.
2. The vector corresponding to the `[CLS]` flag bit is used as the semantic representation of the whole image because this symbol with no obvious semantic information will "fairly" integrate the semantic information of each patch in the image compared with other patches, and thus better represent the semantic of the whole image.

<a name="5"></a>

## Issue 5

### Q5.1: What is included in the PaddleClas training profile? How to modify during the model training?

**A**: PaddleClas configures 6 modules, namely: global configuration, network architecture, learning rate, optimizer, training and validation.

The global configuration contains information about the configuration of the task, such as the number of categories, the amount of data in the training set, the number of epochs to train, the size of the network input, etc. If you want to train a custom task or use your own training set, please pay attention to this section.

The configuration of the network structure defines the network to be used. In practice, the first step is to select the appropriate configuration file, so this part of the configuration is usually not modified. Modifications are only made when customizing the network structure, or when there are special requirements for the task.

It is recommended to use the default configuration for the learning rate and optimizer. These parameters are the ones that are already tuned. Fine-tuning can also be done if the changes to the task are significant.

The training and inference configurations include batch_size, dataset, data transforms (transforms), number of workers (num_workers) and other important configurations, which should be modified according to the actual environment. Note that the batch_size in paddleClas is a single card configuration, if it is a multi-card training, the total batch_size is a multiple of the one set in the configuration file. For example, if the configuration file has the batch_size of 64 and 4 cards trained, the total batch_size is 4*64=256. num_workers defines the number of processes on a single card, i.e., if num_workers is 8 with 4 cards for training, there are actually 32 workers.

### Q5.2: How to quickly modify the configuration in the command line?

**A**:

During training, we often need to constantly fine-tune individual configurations without frequent modification of the configuration file. This can be adjusted using -o. The modification is done by first writing the name of the configuration to be changed by level, splitting the levels with dots, and then writing the value to be modified. For example, if we want to modify batch_size, we can add -o DataLoader.TRAIN.sampler.batch_size=512 after the training command.

### Q5.3: How to choose the right model according to the accuracy curve of PaddleClas?

**A**:

PaddleClas provides benchmarks for several models and plots performance curves. There are mainly three kinds: accuracy-inference time curve, accuracy-parameter count curve and accuracy-FLOPS curve, with accuracy on the vertical axis and the other three on the horizontal axis. in general, different models perform consistently on the three plots. Models of the same series are represented on the plots using the same symbols and connected by curves.

Taking the accuracy-inference time curve as an example, a higher point indicates higher accuracy, and a left one indicates faster speed. For example, the model in the upper-left region is a fast and accurate model, while the leftmost point close to the vertical axis is a lightweight model. When using this, you can choose the right model by considering both accuracy and time. As an example, if we want the most accurate model that runs under 10ms, first draw a vertical line 10ms out from the horizontal axis, and then find the highest point on the left side, which is the model that meets the requirements.

In practice, the number of parameters and FLOPS of the model are constant, while the operation time varies under different hardware and software conditions. If you want to choose the model more accurately, then you can run the test in your own environment and get the corresponding performance graph.

### Q5.4: If I want to add two classes in imagenet, can I fix the parameters of the existing fully connected layer and only train the new two?

**A**:

This idea works in theory, but I am afraid it will not work too well. If only the fully-connected layers are fixed and the parameters of the preceding convolutional layers are changed, there is no guarantee that these fully-connected layers will work the same as they did at the beginning. If the parameters of the entire network are kept constant and only the two new categories of fully connected layers are trained, it is also difficult to train the desired results.

If you really need the original 1000 categories to be accurate, you can add the data of the new categories to the original training set, and then finetune with the pre-training model. if you only need a few of the 1000 categories, you can pick out this part of the data, and finetune it after mixing it with the new data.

### Q5.5: When using classification models as pre-training models for other tasks, which layers should be selected as features?

**A**:

There are many strategies to use classification models as the backbone for other tasks, and a more basic approach is presented here. First, remove the final fully-connected layer, which mainly contains the classification information of the original task. If the task is relatively simple, just use the output of the previous layer as featuremap and add the structure corresponding to the task on top of it. If the task involves multiple scales, and the anchor of different scales needs to be selected, such as some detection models, then the output of the layer before each downsampling can be selected as the featuremap.
