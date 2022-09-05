## PP-ShiTuV2图像识别系统

## 目录

- [1. PP-ShiTuV2模型和应用场景介绍](#1-pp-shituv2模型和应用场景介绍)
- [2. 模型快速体验](#2-模型快速体验)
  - [2.1 安装 paddlepaddle](#21-安装-paddlepaddle)
  - [2.2 安装 paddleclas](#22-安装-paddleclas)
  - [2.3 Python推理预测](#23-python推理预测)
- [3. 模型训练、评估和预测](#3-模型训练评估和预测)
  - [3.1 环境配置](#31-环境配置)
  - [3.2 数据准备](#32-数据准备)
    - [3.2.1 主体检测模型数据准备](#321-主体检测模型数据准备)
    - [3.2.2 特征提取模型数据准备](#322-特征提取模型数据准备)
  - [3.3 模型训练](#33-模型训练)
    - [3.3.1 主体检测模型训练](#331-主体检测模型训练)
    - [3.3.2 特征提取模型训练](#332-特征提取模型训练)
  - [3.4 模型评估](#34-模型评估)
- [4. 模型推理部署](#4-模型推理部署)
  - [4.1 推理模型准备](#41-推理模型准备)
    - [4.1.1 基于训练得到的权重导出 inference 模型](#411-基于训练得到的权重导出-inference-模型)
    - [4.1.2 直接下载 inference 模型](#412-直接下载-inference-模型)
  - [4.2 测试数据准备](#42-测试数据准备)
  - [4.3 基于 Python 预测引擎推理](#43-基于-python-预测引擎推理)
    - [4.3.1 预测单张图像](#431-预测单张图像)
    - [4.3.2 基于文件夹的批量预测](#432-基于文件夹的批量预测)
  - [4.4. 基于 C++ 预测引擎推理](#44-基于-c-预测引擎推理)
- [5 模块介绍](#5-模块介绍)
  - [5.1 主体检测模型](#51-主体检测模型)
  - [5.2 特征提取模型](#52-特征提取模型)
    - [5.2.1 训练数据集优化与扩充](#521-训练数据集优化与扩充)
    - [5.2.2 骨干网络优化](#522-骨干网络优化)
    - [5.2.3 网络结构优化](#523-网络结构优化)
    - [5.2.4 数据增强优化](#524-数据增强优化)
  - [6.5 服务化部署](#65-服务化部署)
  - [6.6 端侧部署](#66-端侧部署)
  - [6.7 Paddle2ONNX 模型转换与预测](#67-paddle2onnx-模型转换与预测)
- [参考文献](#参考文献)

## 1. PP-ShiTuV2模型和应用场景介绍

PP-ShiTuV2 是基于 PP-ShiTuV1 改进的一个实用轻量级通用图像识别系统，相比 PP-ShiTuV1 具有更高的识别精度、更强的泛化能力以及相近的推理速度<sup>*</sup>。该系统主要针对**训练数据集**、特征提取两个部分进行优化，使用了更优的骨干网络、损失函数与训练策略。使得 PP-ShiTuV2 在多个实际应用场景上的检索性能有显著提升。

**本文档提供了用户使用 PaddleClas 的 PP-ShiTuV2 图像识别方案进行快速构建轻量级、高精度、可落地的图像识别pipeline。该pipeline可以广泛应用于商场商品识别场景、安防人脸或行人识别场景、海量图像检索过滤等场景中。**

<div align="center">
<img src="../../images/structure.jpg" />
</div>


下表列出了 PP-ShiTuV2 中的识别模型用不同的模型结构与训练策略所得到的相关指标，

| 模型                   | 延时(ms) | 存储(MB) | product<sup>*</sup> |      | Aliproduct |      | VeRI-Wild |      | LogoDet-3k |      | iCartoonFace |      | SOP      |      | Inshop   |      | gldv2    |      | imdb_face |      | iNat     |      | instre   |      | sketch   |      | sop      |      |
| :--------------------- | :----------- | :------ | :------------------ | :--- | ---------- | ---- | --------- | ---- | ---------- | ---- | ------------ | ---- | -------- | ---- | -------- | ---- | -------- | ---- | --------- | ---- | -------- | ---- | -------- | ---- | -------- | ---- | -------- | ---- |
|                        |              |         | recall@1            | mAP  | recall@1   | mAP  | recall@1  | mAP  | recall@1   | mAP  | recall@1     | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1  | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  | recall@1 | mAP  |
| PP-ShiTuV1_general_rec | 5.0          | 34       | 63.0                | 51.5 | 83.9       | 83.2 | 88.7      | 60.1 | 86.1       | 73.6 | 84.1         | 72.3 | 79.7     | 58.6 | 89.1     | 69.4 | 98.2     | 91.6 | 28.8      | 8.42 | 12.6     | 6.1  | 72.0     | 50.4 | 27.9     | 9.5  | 97.6     | 90.3 |
| [PP-ShiTuV2_general_rec](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/PPShiTuV2/general_PPLCNetV2_base_pretrained_v1.0.pdparams) | 6.1          | 19       | 73.7                | 61.0 | 84.2       | 83.3 | 87.8      | 68.8 | 88.0       | 63.2 | 53.6         | 27.5 | 77.6     | 55.3 | 90.8     | 74.3 | 98.1     | 90.5 | 35.9      | 11.2 | 38.6     | 23.9 | 87.7     | 71.4 | 39.3     | 15.6 | 98.3     | 90.9 |

**注：**
- product数据集是为了验证PP-ShiTu的泛化性能而制作的数据集，所有的数据都没有在训练和测试集中出现。该数据包含8个大类（人脸、化妆品、地标、红酒、手表、车、运动鞋、饮料），299个小类。测试时，使用299个小类的标签进行测试；sop数据集来自[GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval](https://arxiv.org/abs/2111.13122)，可视为“SOP”数据集的子集。
- recall及mAP指标的介绍可以参考 [常用指标](../algorithm_introduction/reid.md#22-常用指标)。
- 延时是基于 Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz 测试得到，开启 MKLDNN 加速策略，线程数为10。

## 2. 模型快速体验

### 2.1 安装 paddlepaddle

- 您的机器安装的是 CUDA9 或 CUDA10，请运行以下命令安装
  ```shell
  python3.7 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```
- 您的机器是CPU，请运行以下命令安装
  ```shell
  python3.7 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```
### 2.2 安装 paddleclas

使用如下命令快速安装 paddleclas
```shell
python3.7 setup.py install
```

### 2.3 Python推理预测

```shell
# 安装faiss库
python3.7 -m pip install faiss-cpu==1.7.1post2

# 准备检测、特征提取模型
cd deploy
mkdir models
cd models
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar && tar -xf general_PPLCNetV2_base_pretrained_v1.0_infer.tar

# 执行识别命令
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.use_gpu=False -o Global.infer_imgs="../docs/images/recognition/drink_data_demo/test_images/100.jpeg"
```

最终输出结果如下。

```log
[{'bbox': [437, 71, 660, 728], 'rec_docs': '元气森林', 'rec_scores': 0.7740249}, {'bbox': [221, 72, 449, 701], 'rec_docs': '元气森林', 'rec_scores': 0.6950992}, {'bbox': [794, 104, 979, 652], 'rec_docs': '元气森林', 'rec_scores': 0.6305153}]
```
其中 `bbox` 表示检测出的主体所在位置，`rec_docs` 表示索引库中与检测框最为相似的类别，`rec_scores` 表示对应的置信度。

检测的可视化结果默认保存在 `output` 文件夹下，对于本张图像，识别结果可视化如下所示。

如需更换其他预测的数据，只需要改变 `-o Global.infer_imgs=` 后的路径即可，路径可以是文件或者文件夹。

## 3. 模型训练、评估和预测

### 3.1 环境配置
请参考文档 [PaddleClas环境准备](../installation/install_paddleclas.md) 以及 [PaddleDetection环境准备](../image_recognition_pipeline/mainbody_detection.md#31-环境准备)，配置 PaddleClas 与 PaddleDetection 运行环境。

### 3.2 数据准备

#### 3.2.1 主体检测模型数据准备
请参考文档 [主体检测数据准备](../image_recognition_pipeline/mainbody_detection.md#32-数据准备)


#### 3.2.2 特征提取模型数据准备
请参考文档 [特征提取数据准备](../image_recognition_pipeline/feature_extraction.md#51-数据准备)

### 3.3 模型训练

#### 3.3.1 主体检测模型训练
请参考文档 [主体检测模型训练](../image_recognition_pipeline/mainbody_detection.md#34-启动训练)

#### 3.3.2 特征提取模型训练
请参考文档 [特征提取模型训练](../image_recognition_pipeline/feature_extraction.md#52-模型训练)

### 3.4 模型评估
检测模型评估代码可参考 [PaddleDetection模型评估](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/GETTING_STARTED_cn.md#5-%E8%AF%84%E4%BC%B0)，特征提取模型评估请参考文档 [特征提取模型评估](../image_recognition_pipeline/feature_extraction.md#53-模型评估)

## 4. 模型推理部署

### 4.1 推理模型准备
Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。相比于直接基于预训练模型进行预测，Paddle Inference可使用 MKLDNN、CUDNN、TensorRT 进行预测加速，从而实现更优的推理性能。更多关于 Paddle Inference 推理引擎的介绍，可以参考 [Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)。

当使用 Paddle Inference 推理时，加载的模型类型为 inference 模型。本案例提供了两种获得 inference 模型的方法，如果希望得到和文档相同的结果，请选择 [直接下载 inference 模型](#412-直接下载-inference-模型) 的方式。

#### 4.1.1 基于训练得到的权重导出 inference 模型
主体检测模型权重导出请参考文档 [主体检测推理模型准备](../image_recognition_pipeline/mainbody_detection.md#41-推理模型准备)

此处，我们提供了特征提取模型的权重转换脚本，执行该脚本可以得到对应的 inference 模型：
```shell
python3.7 tools/export_model.py \
-c ./ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml \
-o Global.pretrained_model="./output/GeneralRecognitionV2_PPLCNetV2_base/RecModel/best_model" \
-o Global.save_inference_dir=deploy/models/GeneralRecognitionV2_PPLCNetV2_base`
```
执行完该脚本后会在 `deploy/models/` 下生成 `GeneralRecognitionV2_PPLCNetV2_base` 文件夹，具有如下文件结构：

```log
deploy/models/
├── GeneralRecognitionV2_PPLCNetV2_base
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

#### 4.1.2 直接下载 inference 模型
[4.1.1 小节](#411-基于训练得到的权重导出-inference-模型) 提供了导出 inference 模型的方法，此处也提供了我们导出好的 inference 模型，可以直接下载体验。
```shell
cd deploy/models

# 下载主体检测inference模型并解压
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar && tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar

# 下载特征提取inference模型并解压
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0_infer.tar && tar -xf general_PPLCNetV2_base_pretrained_v1.
```

### 4.2 测试数据准备

执行以下命令下载并解压测试数据：

```shell
cd deploy
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar && tar -xf drink_dataset_v2.0.tar
```

### 4.3 基于 Python 预测引擎推理

#### 4.3.1 预测单张图像

首先返回至 `deploy` 目录
```shell
cd ../
```

然后执行以下命令对单张图像 `./drink_dataset_v2.0/test_images/100.jpeg` 进行识别。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs="./drink_dataset_v2.0/test_images/100.jpeg"
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.use_gpu=False -o Global.infer_imgs="./drink_dataset_v2.0/test_images/100.jpeg"
```

最终输出结果如下。

```log
[{'bbox': [437, 71, 660, 728], 'rec_docs': '元气森林', 'rec_scores': 0.7740249}, {'bbox': [221, 72, 449, 701], 'rec_docs': '元气森林', 'rec_scores': 0.6950992}, {'bbox': [794, 104, 979, 652], 'rec_docs': '元气森林', 'rec_scores': 0.6305153}]
```

#### 4.3.2 基于文件夹的批量预测

如果希望预测文件夹内的图像，可以直接修改配置文件中的 Global.infer_imgs 字段，也可以通过下面的 -o 参数修改对应的配置。

```shell
# 使用下面的命令使用 GPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.infer_imgs="./drink_dataset_v2.0/test_images"
# 使用下面的命令使用 CPU 进行预测
python3.7 python/predict_system.py -c configs/inference_general.yaml -o Global.use_gpu=False -o Global.infer_imgs="./drink_dataset_v2.0/test_images"
```

终端中会输出该文件夹内所有图像的分类结果，如下所示。

```log
...
[{'bbox': [0, 0, 600, 600], 'rec_docs': '红牛-强化型', 'rec_scores': 0.74081033}]
Inference: 120.39852142333984 ms per batch image
[{'bbox': [0, 0, 514, 436], 'rec_docs': '康师傅矿物质水', 'rec_scores': 0.6918598}]
Inference: 32.045602798461914 ms per batch image
[{'bbox': [138, 40, 573, 1198], 'rec_docs': '乐虎功能饮料', 'rec_scores': 0.68214047}]
Inference: 113.41428756713867 ms per batch image
[{'bbox': [328, 7, 467, 272], 'rec_docs': '脉动', 'rec_scores': 0.60406065}]
Inference: 122.04337120056152 ms per batch image
[{'bbox': [242, 82, 498, 726], 'rec_docs': '味全_每日C', 'rec_scores': 0.5428652}]
Inference: 37.95266151428223 ms per batch image
[{'bbox': [437, 71, 660, 728], 'rec_docs': '元气森林', 'rec_scores': 0.7740249}, {'bbox': [221, 72, 449, 701], 'rec_docs': '元气森林', 'rec_scores': 0.6950992}, {'bbox': [794, 104, 979, 652], 'rec_docs': '元气森林', 'rec_scores': 0.6305153}]
...
```

其中 `bbox` 表示检测出的主体所在位置，`rec_docs` 表示索引库中与检测框最为相似的类别，`rec_scores` 表示对应的置信度。

### 4.4. 基于 C++ 预测引擎推理
PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考 [服务器端 C++ 预测](../../../deploy/cpp_shitu/readme.md) 来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考 [基于 Visual Studio 2019 Community CMake 编译指南](../inference_deployment/cpp_deploy_on_windows.md) 完成相应的预测库编译和模型预测工作。

## 5 模块介绍

### 5.1 主体检测模型

主体检测模型使用 `PicoDet-LCNet_x2_5`，详细信息参考：[picodet_lcnet_x2_5_640_mainbody](../image_recognition_pipeline/mainbody_detection.md)。

### 5.2 特征提取模型

#### 5.2.2 骨干网络

我们将骨干网络从 `PPLCNet_x2_5` 替换成了 [`PPLCNetV2_base`](../models/PP-LCNetV2.md)，相比 `PPLCNet_x2_5`， `PPLCNetV2_base` 基本保持了较高的分类精度，并减少了40%的推理时间<sup>*</sup>。

**注：** <sup>*</sup>推理环境基于 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 硬件平台，OpenVINO 推理平台。

#### 5.2.3 网络结构

我们对 `PPLCNetV2_base` 结构做了微调，并加入了在行人重检测、地标检索、人脸识别等任务上较为通用有效的优化调整。主要包括以下几点：

1. `PPLCNetV2_base` 结构微调：去掉了 BackBone 末尾的 `ReLU` 和 `FC`。

2. `last stride=1`：只将最后一个 stage 的 stride 改为1，增加最后输出的特征图的语义信息。

3. `BN Neck`：特征向量的每个维度进行标准化，让模型更快收敛。

    | 模型                                                               | training data     | recall@1%(mAP%) |
    | :----------------------------------------------------------------- | :---------------- | :-------------- |
    | PP-ShiTuV1                                                         | PP-ShiTuV1 数据集 | 63.0(51.5)      |
    | PP-ShiTuV1+`PPLCNetV2_base`+`last_stride=1`+`BNNeck`+`TripletLoss` | PP-ShiTuV1 数据集 | 72.3(60.5)      |

4. `TripletAngularMarginLoss`：我们基于原始的 `TripletLoss` (困难三元组损失)进行了改进，将优化目标从 L2 欧几里得空间更换成余弦空间，并加入了 anchor 与 positive/negtive 之间的硬性距离约束，让训练与测试的目标更加接近，提升模型的泛化能力。

    | 模型                                                                            | training data     | recall@1%(mAP%) |
    | :------------------------------------------------------------------------------ | :---------------- | :-------------- |
    | PP-ShiTuV1+`PPLCNetV2_base`+`last_stride=1`+`BNNeck`+`TripletLoss`              | PP-ShiTuV2 数据集 | 71.9(60.2)      |
    | PP-ShiTuV1+`PPLCNetV2_base`+`last_stride=1`+`BNNeck`+`TripletAngularMarginLoss` | PP-ShiTuV2 数据集 | 73.7(61.0)      |

#### 5.2.4 数据增强

我们考虑到实际相机拍摄时目标主体可能出现一定的旋转而不一定能保持正立状态，因此我们在数据增强中加入了适当的 [随机旋转增强](../../../ppcls/configs/GeneralRecognitionV2/GeneralRecognitionV2_PPLCNetV2_base.yaml#L117)，以提升模型在真实场景中的检索能力。

### 6.5 服务化部署
Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考 [Paddle Serving 代码仓库](https://github.com/PaddlePaddle/Serving)。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考 [模型服务化部署](../inference_deployment/recognition_serving_deploy.md) 来完成相应的部署工作。

### 6.6 端侧部署
Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考 [Paddle Lite 代码仓库](https://github.com/PaddlePaddle/Paddle-Lite)。

### 6.7 Paddle2ONNX 模型转换与预测
Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考 [Paddle2ONNX 代码仓库](https://github.com/PaddlePaddle/Paddle2ONNX)。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考 [Paddle2ONNX 模型转换与预测](../../../deploy/paddle2onnx/readme.md) 来完成相应的部署工作。

## 参考文献
1. Schall, Konstantin, et al. "GPR1200: A Benchmark for General-Purpose Content-Based Image Retrieval." International Conference on Multimedia Modeling. Springer, Cham, 2022.
2. Luo, Hao, et al. "A strong baseline and batch normalization neck for deep person re-identification." IEEE Transactions on Multimedia 22.10 (2019): 2597-2609.
