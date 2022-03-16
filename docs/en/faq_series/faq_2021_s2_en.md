# PaddleClas FAQ Summary - 2021 Season 2

## Before You Read

- We collect some frequently asked questions in issues and user groups since PaddleClas is open-sourced and provide brief answers, aiming to give some reference for the majority to save you from twists and turns.
- There are many talents in the field of image classification, recognition and retrieval with quickly updated models and papers, and the answers here mainly rely on our limited project practice, so it is not possible to cover all facets. We sincerely hope that the man of insight will help to supplement and correct the content, thanks a lot.

## Catalogue

- [1. Theory](#1)
  - [1.1 Basic Knowledge of PaddleClas](#1.1)
  - [1.2 Backbone Network and Pre-trained Model Library](#1.2)
  - [1.3 Image Classification](#1.3)
  - [1.4 General Detection](#1.4)
  - [1.5 Image Recognition](#1.5)
  - [1.6 Vector Search](#1.6)
- [2. Practice](#2)
  - [2.1 Common Problems in Training and Evaluation](#2.1)
  - [2.2 Image Classification](#2.2)
  - [2.3 General Detection](#2.3)
  - [2.4 Image Recognition](#2.4)
  - [2.5 Vector Search](#2.5)
  - [2.6 Model Inference Deployment](#2.6)

<a name="1"></a>

## 1. Theory

<a name="1.1"></a>

### 1.1 Basic Knowledge of PaddleClas

#### Q1.1.1 Differences between PaddleClas and PaddleDetection

**A**：PaddleClas is an image recognition repo that integrates mainbody detection, image classification, and image retrieval to solve most image recognition problems. It can be easily adopted by users to solve small sample and multi-category issues in the field. PaddleDetection provides the ability of target detection, keypoint detection, multi-target tracking, etc., which is convenient for users to locate the points and regions of interest in images, and is widely used in industrial quality inspection, remote sensing image detection, unmanned inspection and other projects.

#### Q1.1.3: What does the parameter momentum mean in the Momentum optimizer?

**A**:

Momentum optimizer is based on SGD optimizer and introduces the concept of "momentum". In the SGD optimizer, the update of the parameter `w` at the time `t+1` can be expressed as

```
w_t+1 = w_t - lr * grad
```

`lr` is the learning rate and `grad` is the gradient of the parameter `w` at this point. With the introduction of momentum, the update of the parameter `w` can be expressed as

```
v_t+1 = m * v_t + lr * grad
w_t+1 = w_t - v_t+1
```

Here `m` is the `momentum`, which is the weighted value of the cumulative momentum, generally taken as `0.9`. And when the value is less than `1`, the earlier the gradient is, the smaller the impact on the current. For example, when the momentum parameter `m` takes `0.9`, the weighted value of the gradient of `t-5` is `0.9 ^ 5 = 0.59049` at time `t`, while the value at time `t-2` is `0.9 ^ 2 = 0.81`. Therefore, it is intuitive that gradient information that is too "far away" is of little significance for the current reference, while "recent" historical gradient information matters more.

[](../../images/faq/momentum.jpeg)

By introducing the concept of momentum, the effect of historical updates is taken into account in parameter updates, thus speeding up the convergence and improving the loss (cost, loss) oscillation caused by the `SGD` optimizer.

#### Q1.1.4: Does PaddleClas have an implementation of the paper `Fixing the train-test resolution discrepancy`?

**A**: Currently, it is not implemented. If needed, you can try to modify the code yourself. In brief, the idea proposed in this paper is to fine-tune the final FC layer of the trained model using a larger resolution as input. Specifically, train the model network on a lower resolution dataset first, then set the parameter `stop_gradient=True ` for the weights of all layers of the network except the final FC layer, and at last fine-tune the network with a larger resolution input.

<a name="1.2"></a>

### 1.2 Backbone Network and Pre-trained Model Library

<a name="1.3"></a>

### 1.3 Image Classification

#### Q1.3.1: Does PaddleClas provide data enhancement for adjusting image brightness, contrast, saturation, hue, etc.?

**A**：

PaddleClas provides a variety of data augmentation methods, which can be divided into 3 categories.

1. Image transformation ： AutoAugment, RandAugment;
2. Image cropping： CutOut、RandErasing、HideAndSeek、GridMask；
3. Image aliasing ：Mixup, Cutmix.

Among them, RandAngment provides a variety of random combinations of data augmentation methods, which can meet the needs of brightness, contrast, saturation, hue and other aspects.

<a name="1.4"></a>

### 1.4 General Detection

#### Q1.4.1 Does the mainbody detection only export one subject detection box at a time?

**A**：The number of outputs for the main body detection is configurable through the configuration file. In the configuration file, Global.threshold controls the threshold value for detection, so boxes smaller than this threshold are discarded; and Global.max_det_results controls the maximum number of results returned. The two together determine the number of output detection boxes.

#### Q1.4.2 How is the data selected for training the mainbody detection model? Will it harm the accuracy to switch to a smaller model?

**A**：

The training data is a randomly selected subset of publicly available datasets such as COCO, Object365, RPC, and LogoDet. We are currently introducing an ultra-lightweight mainbody detection model in version 2.3, which can be found in [Mainbody Detection](../../en/image_recognition_pipeline/mainbody_detection_en.md#2-model-selection).

#### Q1.4.3: Is there any false detections in some scenarios with the current mainbody detection model?

**A**：The current mainbody detection model is trained using publicly available datasets such as COCO, Object365, RPC, LogoDet, etc. If the data to be detected is similar to industrial quality inspection and other data with large differences from common categories, it is necessary to fine-tune the training based on the current detection model again.

<a name="1.5"></a>

### 1.5 Image Recognition

#### Q1.5.1 Is `triplet loss` needed for  `circle loss` ?

**A**：

`circle loss` is a unified form of sample pair learning and classification learning, and `triplet loss` can be added if it is a classification learning.

#### Q1.5.2 Which recognition model is better if not to recognize open source images in all four directions?

**A**：

The product recognition model is recommended. For one, the range of products is wider and the probability that the recognized image is a product is higher. For two, the training data of the product recognition model uses 50,000 categories of data, which has better generalization ability and more robust features.

#### Q1.5.3 Why is 512-dimensional vector is finally adopted instead of 1024 or others?

**A**：

Vectors with small dimensions should be adopted. 128 or even smaller are practically used to speed up the computation. In general, a dimension of 512 is large enough to adequately represent the features.

<a name="1.6"></a>

### 1.6 Vector Search

#### Q1.6.1 Does the Möbius vector search algorithm currently used by PaddleClas support index.add() similar to the one used by faiss? Also, do I have to train every time I build a new graph? Is the train here to speed up the search or to build similar graphs?

**A**：The faiss retrieval module is now supported in the release/2.3 branch and is no longer supported by Möbius, which provides a graph-based algorithm that is similar to the nearest neighbor search and currently supports two types of distance calculation: inner product and L2 distance. However, Möbius does not support the index.add function provided in faiss. So if you need to add content to the search library, you need to rebuild a new index from scratch. The search algorithm internally performs a train-like process each time the index is built, which is different from the train interface provided by faiss. Therefore, if you need the faiss module, you can use the release/2.3 branch, and if you need Möbius, you need to fall back to the release/2.2 branch for now.

#### Q1.6.2: What exactly are the `Query` and `Gallery` configurations used for in the PaddleClas image recognition for Eval configuration file?

**A**:

 Both `Query` and `Gallery` are data set configurations, where `Gallery` is used to configure the base library data and `Query` is used to configure the validation set. When performing Eval, the model is first used to forward compute feature vectors on the `Gallery` base library data, which are used to construct the base library, and then the model forward computes feature vectors on the data in the `Query` validation set, and then computes metrics such as recall rate in the base library.

<a name="2"></a>

## 2. Practice

<a name="2.1"></a>

### 2.1 Common Problems in Training and Evaluation

#### Q2.1.1 Where is the `train_log` file in PaddleClas?

**A**：`train.log` is stored in the path where the weights are stored.

#### Q2.1.2 Why nan is the output of the model training?

**A**： 1. Make sure the pre-trained model is loaded correctly, the easiest way is to add the parameter `-o Arch.pretrained=True`; 2. When fine-tuning the model, the learning rate should not be too large, e.g. 0.001.

#### Q2.1.3 Is it possible to perform frame-by-frame prediction in a video?

**A**：Yes, but currently PaddleClas does not support video input. You can try to modify the code of PaddleClas or store the video frame by frame before using PaddleClas.

#### Q2.1.4: In data preprocessing, what setting can be adopted without cropping the input data? Or how to set the size of the crop?

**A**: The data preprocessing operators supported by PaddleClas can be viewed in `ppcls/data/preprocess/__init__.py`, and all supported operators can be configured in the configuration file. The name of the operator needs to be the same as the operator class name, and the parameters need to be the same as the constructor parameters of the corresponding operator class. If you do not need to crop the image, you can remove `CropImage` and `RandCropImage` and replace them with `ResizeImage`, and you can set different resize methods with its parameters by using the `size` parameter to directly scale the image to a fixed size, and using the `resize_short` parameter to maintain the image aspect ratio for scaling. To set the crop size, use the `size` parameter of the `CropImage` operator, or the `size` parameter of the `RandCropImage` operator.

#### Q2.1.5: Why do I get a usage error after PaddlePaddle installation and cannot import any modules under paddle (import paddle.xxx)?

**A**:

You can first test if Paddle is installed correctly by using the following code.

```
import paddle
paddle.utils.install_check.run_check(）
```

When installed correctly, the following prompts will be displayed.

```
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

Otherwise, the relevant question will prompt out. Also, after installing both the CPU and the GPU version of Paddle, you will need to uninstall both versions and reinstall the required version due to conflicts between the two versions.

#### Q2.1.6: How to save the optimal model during training?

**A**:

PaddleClas saves/updates the following three types of models during training.

1. the latest model (`latest.pdopt`, `latest.pdparams`, `latest.pdstates`), which can be used to resume training when it is unexpectedly interrupted.
2. the best model (`best_model.pdopt`, `best_model.pdparams`, `best_model.pdstates`).
3. breakpoints at the end of an epoch during training (`epoch_xxx.pdopt`, `epoch_xxx.pdparams`, `epoch_xxx.pdstates`). The `Global.save_interval` field in the training profile indicates the save interval for this model. If you make it larger than the total number of epochs, intermediate breakpoint models will no longer be saved.

#### Q2.1.7: How to address the `ERROR: Unexpected segmentation fault encountered in DataLoader workers.` during training?

**A**：Try setting the field `num_workers` in the training profile to `0`; try making the field `batch_size` in the training profile smaller; ensure that the dataset format and the dataset path in the profile are correct.

#### Q2.1.8: How to use `Mixup` and `Cutmix` during training?

**A**：

- For `Mixup`, please refer to [Mixup](../../../ppcls/configs/ImageNet/DataAugment/ResNet50_ Mixup.yaml#L63-L65); and`Cuxmix`, please refer to [Cuxmix](../../../ppcls/configs/ImageNet/DataAugment/ResNet50_Cutmix.yaml#L63-L65).
- The training accuracy (Acc) metric cannot be calculated when using `Mixup` or `Cutmix` for training, so you need to remove the `Metric.Train.TopkAcc` field in the configuration file, please refer to [Metric.Train.TopkAcc](../../../ppcls/configs/ImageNet/DataAugment/ResNet50_Cutmix.yaml#L125-L128).

#### Q2.1.9: What are the fields `Global.pretrain_model` and `Global.checkpoints` used for in the training configuration file yaml?

**A**：

- When `fine-tune` is required, the path of the file of pre-training model weights can be configured via the field `Global.pretrain_model`, which usually has the suffix `.pdparams`.
- During training, the training program automatically saves the breakpoint information at the end of each epoch, including the optimizer information `.pdopt` and model weights information `.pdparams`. In the event that the training process is unexpectedly interrupted and needs to be resumed, the breakpoint information file saved during training can be configured via the field `Global.checkpoints`, for example by configuring `checkpoints: . /output/ResNet18/epoch_18` to restore the breakpoint information at the end of 18 epoch training. PaddleClas will automatically load `epoch_18.pdopt` and `epoch_18.pdparams` to continue training from 19 epoch.

<a name="2.2"></a>

### 2.2 Image Classification

#### Q2.2.1 How to distill the small model after pre-training the large model on 500M data and then distill the fintune model on 1M data in SSLD?

**A**：The steps are as follows:

1. Obtain the `ResNet50-vd` model based on the distillation of the open-source `ResNeXt101-32x16d-wsl` model from Facebook.
2. Use this `ResNet50-vd` to distill `MobilNetV3` on a 500W dataset.
3. Considering that the distribution of the 500W dataset is not exactly the same as that of the 100W data, this piece of data is finetuned on the 100W data to slightly improve the accuracy.

#### Q2.2.2 nan appears in loss when training SwinTransformer

**A**：When training SwinTransformer, please use `Paddle` `2.1.1` or above, and load the pre-trained model we provide. Also, the learning rate should be kept at an appropriate level.

<a name="2.3"></a>

### 2.3 General Detection

#### Q2.3.1 Why are there some images that are detected as the original image?

**A**：The mainbody detection model returns the detection frame, but in fact, in order to make the subsequent recognition model more accurate, the original image is also returned along with the detection frame. Subsequently, the original image or the detection frame will be sorted according to its similarity with the images in the library, and the label of the image in the library with the highest similarity will be the label of the recognized image.

#### Q2.3.2：

**A**：A real-time detection presents high requirements for the detection speed; PP-YOLO is a lightweight target detection model provided by Paddle team, which strikes a good balance of detection speed and accuracy, you can try PP-YOLO for detection. For the use of PP-YOLO, you can refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/README_cn.md).

#### Q2.3.3: For unknown labels, adding gallery dataset can be used for subsequent classification recognition (without training), but if the previous detection model cannot locate and detect the unknown labels, is it still necessary to train the previous detection model?

**A**：If the detection model does not perform well on your own dataset, you need to finetune it again on your own detection dataset.

<a name="2.4"></a>

### 2.4 Image Recognition

#### Q2.4.1: Why is `Illegal instruction` reported during the recognition inference?

**A**：If you are using the release/2.2 branch, it is recommended to update it to the release/2.3 branch, where we replaced the Möbius search model with the faiss search module, as described in [Vector Search Tutorial](../image_recognition_pipeline/vector_search_en.md). If you still have problems, you can contact us in the WeChat group or raise an issue on GitHub.

#### Q2.4.2: How can recognition models be fine-tuned to train on the basis of pre-trained models?

**A**：The fine-tuning training of the recognition model is similar to that of the classification model. The recognition model can be loaded with a pre-trained model of the product, and the training process can be found in [recognition model training](../models_training/recognition_en.md), and we will continue to refine the documentation.

#### Q2.4.3: Why does it fail to run all mini-batches in each epoch when training metric learning?

**A**：When training metric learning, the Sampler used is DistributedRandomIdentitySampler, which does not sample all the images, resulting in each epoch sampling only part of the data, so it is normal that the mini-batch cannot run through the display. This issue has been optimized in the release/2.3 branch, please update to release/2.3 to use it.

#### Q2.4.4: Why do some images have no recognition results?

**A**：In the configuration file (e.g. inference_product.yaml), `IndexProcess.score_thres` controls the minimum value of cosine similarity of the recognized image to the image in the library. When the cosine similarity is less than this value, the result will not be printed. You can adjust this value according to your actual data.

<a name="2.5"></a>

### 2.5 Vector Search

#### Q2.5.1: Why is the error `assert text_num >= 2` reported after adding an image to the index?

**A**：Make sure that the image path and the image name in data_file.txt is separated by a single table instead of a space.

#### Q2.5.2: Do I need to rebuild the index to add new base data?

**A**：Starting from release/2.3 branch, we have replaced the Möbius search model with the faiss search module, which already supports the addition of base data without building the base library, as described in [Vector Search Tutorial](../image_recognition_pipeline/vector_search_en.md).

#### Q2.5.3: How to deal with the reported error clang: error: unsupported option '-fopenmp' when recompiling index.so in Mac?

**A**：

If you are using the release/2.2 branch, it is recommended to update it to the release/2.3 branch, where we replaced the Möbius search model with the faiss search module, as described in [Vector Search Tutorial](../image_recognition_pipeline/vector_search_en.md). If you still have problems, you can contact us in the user WeChat group or raise an issue on GitHub.

#### Q2.5.4: How to set the parameter `pq_size` when build searches the base library?

**A**：

`pq_size` is a parameter of the PQ search algorithm, which can be simply understood as a "tiered" search algorithm. And `pq_size` is the "capacity" of each tier, so the setting of this parameter will affect the performance. However, in the case that the total data volume of the base library is not too large (less than 10,000), this parameter has little impact on the performance. So for most application scenarios, there is no need to modify this parameter when building the base library. For more details on the PQ search algorithm, see the related [paper](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf).

<a name="2.6"></a>

### 2.6 Model Inference Deployment

#### Q2.6.1: How to add the parameter of a module that is enabled by hub serving?

**A**：See [hub serving parameters](../../../deploy/hubserving/clas/params.py) for more details.

#### Q2.6.2: Why is the result not accurate enough when exporting the inference model for inference deployment?

**A**:

This problem is usually caused by the incorrect loading of the model parameters when exporting. First check the export log for something like the following.

```
UserWarning: Skip loading for ***. *** is not found in the provided dict.
```

If it exists, the model weights were not loaded successfully. Please further check the `Global.pretrained_model` field in the configuration file to see if the path of the model weights file is correctly configured. The suffix of the model weights file is usually `pdparams`, note that the file suffix is not required when configuring this path.

#### Q2.6.3: How to convert the model to `ONNX` format?

**A**：

Paddle supports two ways and relies on the `paddle2onnx` tool, which first requires the installation of `paddle2onnx`.

```
pip install paddle2onnx
```

- From inference model to ONNX format model:

  Take the `combined` format inference model (containing `.pdmodel` and `.pdiparams` files) exported from the dynamic graph as an example, run the following command to convert the model format:

  ```
  paddle2onnx --model_dir ${model_path}  --model_filename  ${model_path}/inference.pdmodel --params_filename ${model_path}/inference.pdiparams --save_file ${save_path}/model.onnx --enable_onnx_checker True
  ```

  In the above commands：

  - `model_dir`: this parameter needs to contain `.pdmodel` and `.pdiparams` files.
  - `model_filename`: this parameter is used to specify the path of the `.pdmodel` file under the parameter `model_dir`.
  - `params_filename`: this parameter is used to specify the path of the `.pdiparams` file under the parameter `model_dir`.
  - `save_file`: this parameter is used to specify the path to the directory where the converted model is saved.

  For the conversion of a non-`combined` format inference model exported from a static diagram (usually containing the file `__model__` and multiple parameter files), and more parameter descriptions, please refer to the official documentation of [paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README.md#parameters).

- Exporting ONNX format models directly from the model networking code.

  Take the model networking code of dynamic graphs as an example, the model class is a subclass that inherits from `paddle.nn.Layer` and the code is shown below:

  ```python
  import paddle
  from paddle.static import InputSpec

  class SimpleNet(paddle.nn.Layer):
      def __init__(self):
          pass
      def forward(self, x):
          pass

  net = SimpleNet()
  x_spec = InputSpec(shape=[None, 3, 224, 224], dtype='float32', name='x')
  paddle.onnx.export(layer=net, path="./SimpleNet", input_spec=[x_spec])
  ```

  Among them：

  - `InputSpec()` function is used to describe the signature information of the model input, including the `shape`, `type` and `name` of the input data (can be omitted).
  - The `paddle.onnx.export()` function needs to specify the model grouping object `net`, the save path of the exported model `save_path`, and the description of the model's input data `input_spec`.

  Note that the `paddlepaddle`  `2.0.0` or above should be adopted.See [paddle.onnx.export](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/onnx/export_en.html) for more details on the parameters of the  `paddle.onnx.export()` function.
