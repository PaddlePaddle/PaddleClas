**简体中文 | English(TODO)**

# FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence

**论文出处：**[https://arxiv.org/abs/2001.07685](https://arxiv.org/abs/2001.07685)

## 目录

* [1. 原理介绍](#1-%E5%8E%9F%E7%90%86%E4%BB%8B%E7%BB%8D)
* [2. 精度指标](#2-%E7%B2%BE%E5%BA%A6%E6%8C%87%E6%A0%87)
* [3. 数据准备](#3-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)
* [4. 模型训练](#4-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
* [5. 模型评估与推理部署](#5-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E4%B8%8E%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2)
* [5.1 模型评估](#51-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0)
* [5.2 模型推理](#52-%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)
  * [5.2.1 推理模型准备](#521-%E6%8E%A8%E7%90%86%E6%A8%A1%E5%9E%8B%E5%87%86%E5%A4%87)
  * [5.2.2 基于 Python 预测引擎推理](#522-%E5%9F%BA%E4%BA%8E-python-%E9%A2%84%E6%B5%8B%E5%BC%95%E6%93%8E%E6%8E%A8%E7%90%86)
  * [5.2.3 基于 C++ 预测引擎推理](#523-%E5%9F%BA%E4%BA%8E-c-%E9%A2%84%E6%B5%8B%E5%BC%95%E6%93%8E%E6%8E%A8%E7%90%86)
* [5.4 服务化部署](#54-%E6%9C%8D%E5%8A%A1%E5%8C%96%E9%83%A8%E7%BD%B2)
* [5.5 端侧部署](#55-%E7%AB%AF%E4%BE%A7%E9%83%A8%E7%BD%B2)
* [5.6 Paddle2ONNX 模型转换与预测](#56-paddle2onnx-%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E4%B8%8E%E9%A2%84%E6%B5%8B)
* [6. 参考资料](#6-%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99)

## 1. 原理介绍

**作者提出一种简单而有效的半监督学习方法。主要是在有标签的数据训练的同时，对无标签的数据进行强弱两种不同的数据增强。如果无标签的数据弱数据增强的分类结果，大于阈值，则弱数据增强的输出标签作为软标签，对强数据增强的输出进行loss计算及模型训练。如示例图所示。**

![](https://raw.githubusercontent.com/google-research/fixmatch/master/media/FixMatch%20diagram.png)

## 2. 精度指标

**以下表格总结了复现的 FixMatch在 Cifar10 数据集上的精度指标。**

| **Labels**             | **40**            | **250**           | **4000**          |
| ---------------------------- | ----------------------- | ----------------------- | ----------------------- |
| **Paper (tensorflow)** | **86.19 ± 3.37** | **94.93 ± 0.65** | **95.74 ± 0.05** |
| **pytorch版本**        | **93.60**         | **95.31**         | **95.77**         |
| **paddle版本**         | **93.14**         | **95.37**         | **95.89**         |

**cifar10上，paddle版本配置文件及训练好的模型如下表所示**

| **label** | **配置文件地址**                                                     | **模型下载链接**                                                                                                                     |
| --------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **40**    | [配置文件](../../../../ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40.yaml)   | [模型地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/semi_superwised_learning/FixMatch_WideResNet_cifar10_label40.pdparams)   |
| **250**   | [配置文件](../../../../ppcls/configs/ssl/FixMatch/FixMatch_cifar10_250.yaml)  | [模型地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/semi_superwised_learning/FixMatch_WideResNet_cifar10_label250.pdparams)  |
| **4000**  | [配置文件](../../../../ppcls/configs/ssl/FixMatch/FixMatch_cifar10_4000.yaml) | [模型地址](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/semi_superwised_learning/FixMatch_WideResNet_cifar10_label4000.pdparams) |

上表中的配置是基于单GPU训练的，使用4GPU训练40 label可参考[配置文件](../../../../ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40_4gpu.yaml)，使用这两个配置文件训练得到的模型精度接近。

**接下来主要以** `FixMatch/FixMatch_cifar10_40.yaml`配置和训练好的模型文件为例，展示在cifar10数据集上进行训练、测试、推理的过程。

## 3. 数据准备

在训练及测试的过程中，cifar10数据集会自动下载，请保持联网。如网络问题，则提前下载好[相关数据](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz)，并在以下命令中，添加如下参数

```
${cmd} -o DataLoader.Train.dataset.data_file=${data_file} -o DataLoader.UnLabelTrain.dataset.data_file=${data_file} -o DataLoader.Eval.dataset.data_file=${data_file}
```

**其中：**`${cmd}`为以下的命令，`${data_file}`是下载数据的路径。如4.1中单卡命令就改为：

```shell
python tools/train.py -c ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40.yaml -o DataLoader.Train.dataset.data_file=cifar-10-python.tar.gz -o DataLoader.UnLabelTrain.dataset.data_file=cifar-10-python.tar.gz -o DataLoader.Eval.dataset.data_file=cifar-10-python.tar.gz
```

## 4. 模型训练

1. **执行以下命令开始训练**
   **单卡训练：**

   ```
   python tools/train.py -c ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40.yaml
   ```

   **注：单卡训练大约需要2-4个天。**
2. **查看训练日志和保存的模型参数文件**
   训练过程中会在屏幕上实时打印loss等指标信息，同时会保存日志文件`train.log`、模型参数文件 `*.pdparams`、优化器参数文件 `*.pdopt`等内容到 `Global.output_dir`指定的文件夹下，默认在 `PaddleClas/output/WideResNet/`文件夹下。

## 5. 模型评估与推理部署

### 5.1 模型评估

准备用于评估的 `*.pdparams`模型参数文件，可以使用训练好的模型，也可以使用[4. 模型训练](#4-%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)中保存的模型。

* 以训练过程中保存的`best_model_ema.ema.pdparams`为例，执行如下命令即可进行评估。

  ```
  python3.7 tools/eval.py \
  -c ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40.yaml \
  -o Global.pretrained_model="./output/WideResNet/best_model_ema.ema"
  ```
* 以训练好的模型为例，下载提供的已经训练好的模型，到`PaddleClas/pretrained_models` 文件夹中，执行如下命令即可进行评估。

  ```
  # 下载模型
  cd PaddleClas
  mkdir pretrained_models
  cd pretrained_models
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/semi_superwised_learning/FixMatch_WideResNet_cifar10_label40.pdparams
  cd ..
  # 评估
  python3.7 tools/eval.py \
  -c ppcls/configs/ssl/FixMatch/FixMatch_cifar10_40.yaml \
  -o Global.pretrained_model="pretrained_models/FixMatch_WideResNet_cifar10_label40"
  ```

  **注：**`pretrained_model` 后填入的地址不需要加 `.pdparams` 后缀，在程序运行时会自动补上。
* 查看输出结果

  ```
  ...
  ...
  CELoss: 0.58960, loss: 0.58960, top1: 0.95312, top5: 0.98438, batch_cost: 3.00355s, reader_cost: 1.09548, ips: 21.30810 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 20/157]CELoss: 0.14618, loss: 0.14618, top1: 0.93601, top5: 0.99628, batch_cost: 0.02379s, reader_cost: 0.00016, ips: 2690.05243 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 40/157]CELoss: 0.01801, loss: 0.01801, top1: 0.93216, top5: 0.99505, batch_cost: 0.02716s, reader_cost: 0.00015, ips: 2356.48846 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 60/157]CELoss: 0.63351, loss: 0.63351, top1: 0.92982, top5: 0.99539, batch_cost: 0.02585s, reader_cost: 0.00015, ips: 2475.86506 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 80/157]CELoss: 0.85084, loss: 0.85084, top1: 0.93191, top5: 0.99576, batch_cost: 0.02578s, reader_cost: 0.00015, ips: 2482.59021 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 100/157]CELoss: 0.04171, loss: 0.04171, top1: 0.93147, top5: 0.99567, batch_cost: 0.02676s, reader_cost: 0.00015, ips: 2391.99053 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 120/157]CELoss: 0.89842, loss: 0.89842, top1: 0.93027, top5: 0.99561, batch_cost: 0.02647s, reader_cost: 0.00015, ips: 2418.24635 images/sec
  ppcls INFO: [Eval][Epoch 0][Iter: 140/157]CELoss: 0.57866, loss: 0.57866, top1: 0.93107, top5: 0.99568, batch_cost: 0.02678s, reader_cost: 0.00015, ips: 2389.46068 images/sec
  ppcls INFO: [Eval][Epoch 0][Avg]CELoss: 0.59721, loss: 0.59721, top1: 0.93140, top5: 0.99570
  ```

  默认评估日志保存在`PaddleClas/output/WideResNet/eval.log`中，可以看到我们提供的模型在 cifar10 数据集上的评估指标为top1: 0.93140, top5: 0.99570

### 5.2 模型推理

#### 5.2.1 推理模型准备

将训练过程中保存的模型文件转换成 inference 模型，同样以`best_model_ema.ema.pdparams` 为例，执行以下命令进行转换

```
python3.7 tools/export_model.py \
-c ppcls/configs/ssl/FixMatch_cifar10_40.yaml \
-o -o Global.pretrained_model=output/WideResNet/best_model_ema.ema \
-o Global.save_inference_dir="./deploy/inference"
```

#### 5.2.2 基于 Python 预测引擎推理

1. 修改`PaddleClas/deploy/configs/inference_cls.yaml`

   - 将`infer_imgs:` 后的路径段改为 query 文件夹下的任意一张图片路径（下方配置使用的是 `demo.jpg`图片的路径）
   - 将`rec_inference_model_dir:` 后的字段改为解压出来的 inference模型文件夹路径
   - 将`transform_ops:` 字段下的预处理配置改为 `FixMatch_cifar10_40.yaml` 中 `Eval.dataset` 下的预处理配置

   ```
   Global:
     infer_imgs: "demo"
     rec_inference_model_dir: "./inferece"
     batch_size: 1
     use_gpu: False
     enable_mkldnn: True
     cpu_num_threads: 10
     enable_benchmark: False
     use_fp16: False
     ir_optim: True
     use_tensorrt: False
     gpu_mem: 8000
     enable_profile: False

   RecPreProcess:
     transform_ops:
      -  NormalizeImage:
           scale: 1.0/255.0
           mean: [0.4914, 0.4822, 0.4465]
           std: [0.2471, 0.2435, 0.2616]
           order: hwc
   PostProcess: null
   ```
2. 执行推理命令

   ```
   cd ./deploy/
   python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
   ```
3. 查看输出结果，实际结果为一个长度10的向量，表示图像分类的结果，如

   ```
   demo.JPG:        [ 0.02560742  0.05221584  ...  0.11635944 -0.18817757
   0.07170864]
   ```

#### 5.2.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../../deployment/image_classification/cpp/linux.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考基于 Visual Studio 2019 Community CMake 编译指南完成相应的预测库编译和模型预测工作。

### 5.4 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考Paddle Serving 代码仓库。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../../deployment/PP-ShiTu/paddle_serving.md)来完成相应的部署工作。

### 5.5 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考Paddle Lite 代码仓库。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../../deployment/image_classification/paddle_lite.md)来完成相应的部署工作。

### 5.6 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考Paddle2ONNX 代码仓库。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考**[Paddle2ONNX 模型转换与预测](../../deployment/image_classification/paddle2onnx.md)来完成相应的部署工作。

### 6. 参考资料

1. [FixMatch](https://arxiv.org/abs/2001.07685)
