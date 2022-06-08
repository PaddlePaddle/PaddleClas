简体中文|[English](../../en/algorithm_introduction/reid.md)
# ReID行人重识别

## 目录

- [1. 算法/应用场景简介](#1-算法应用场景简介)
- [2. ReID算法](#2-reid算法)
  - [2.1 ReID strong-baseline](#21-reid-strong-baseline)
    - [2.1.1 原理介绍](#211-原理介绍)
    - [2.1.2 精度指标](#212-精度指标)
    - [2.1.3 数据准备](#213-数据准备)
    - [2.1.4 模型训练](#214-模型训练)
    - [2.1.5 模型评估](#215-模型评估)
    - [2.1.6 模型推理部署](#216-模型推理部署)
      - [2.1.6.1 推理模型准备](#2161-推理模型准备)
      - [2.1.6.2 基于 Python 预测引擎推理](#2162-基于-python-预测引擎推理)
      - [2.1.6.3 基于 C++ 预测引擎推理](#2163-基于-c-预测引擎推理)
    - [2.1.7 服务化部署](#217-服务化部署)
    - [2.1.8 端侧部署](#218-端侧部署)
    - [2.1.9 Paddle2ONNX 模型转换与预测](#219-paddle2onnx-模型转换与预测)
- [3. 总结](#3-总结)
  - [3.1 方法总结与对比](#31-方法总结与对比)
  - [3.2 使用建议/FAQ](#32-使用建议faq)
- [4. 参考资料](#4-参考资料)

### 1. 算法/应用场景简介

行人重识别（Person re-identification）也称行人再识别，是利用[计算机视觉](https://baike.baidu.com/item/计算机视觉/2803351)技术判断[图像](https://baike.baidu.com/item/图像/773234)或者视频序列中是否存在特定行人的[技术](https://baike.baidu.com/item/技术/13014499)。广泛被认为是一个[图像检索](https://baike.baidu.com/item/图像检索/1150910)的子问题。给定一个监控行人图像，检索跨设备下的该行人图像。旨在弥补固定的摄像头的视觉局限，并可与[行人检测](https://baike.baidu.com/item/行人检测/20590256)/行人跟踪技术相结合，可广泛应用于[智能视频监控](https://baike.baidu.com/item/智能视频监控/10717227)、智能安保等领域。

常见的行人重识别方法通过特征提取模块来提取输入图片的局部/全局、单粒度/多粒度特征，再经过融合模块得到一个高维的特征向量。在训练时使用分类头将该特征向量转换成属于每个类别的概率从而以分类任务的方式来优化特征提取模型；在测试或推理时直接将高维的特征向量作为图片描述向量在检索向量库中进行检索，以得到检索结果。而ReID strong-baseline算法则是提出了多个有效优化训练与检索的方法来提升整体模型性能。
<img src="../../images/reid/reid_overview.jpg" align="middle">

### 2. ReID算法

#### 2.1 ReID strong-baseline

论文出处：[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

<img src="../../images/reid/strong-baseline.jpg" width="50%">

##### 2.1.1 原理介绍

作者以普遍使用的基于 ResNet50 的行人重识别模型为基础，探索并总结了以下几种有效且适用性较强的优化方法，大幅度提高了在多个行人重识别数据集上的指标。

1. Warmup：在训练一开始让学习率从一个较小值逐渐升高后再开始下降，有利于梯度下降优化时的稳定性，从而找到更优的参数模型。
2. Random erasing augmentation：随机区域擦除，通过数据增强来提升模型的泛化能力。
3. Label smoothing：标签平滑，提升模型的泛化能力。
4. Last stride=1：设定特征提取模块的最后一个stage的下采样为1，增大输出特征图的分辨率来保留更多细节，提升模型的分类能力。
5. BNNeck：特征向量输入分类头之前先经过BNNeck，让特征在超球体表面服从正态分布，减少了同时优化IDLoss和TripLetLoss的难度。
6. Center loss：给每个类别一个可学习的聚类中心，训练时让类内特征靠近聚类中心，减少类内差异，增大类间差异。
7. Reranking：在检索时考虑查询图像的近邻候选对象，根据候选对象的近邻图像的是否也含有查询图像的情况来优化距离矩阵，最终提升检索精度。

##### 2.1.2 精度指标

以下表格总结了复现的ReID strong-baseline的3种配置在 Market1501 数据集上的精度指标，

| 配置文件                 | recall@1 | mAP   | 参考recall@1 | 参考mAP |
| ------------------------ | -------- | ----- | ------------ | ------- |
| baseline.yaml            | 88.21    | 74.12 | 87.7         | 74.0    |
| softmax.yaml             | 94.18    | 85.76 | 94.1         | 85.7    |
| softmax_with_center.yaml | 94.19    | 85.80 | 94.1         | 85.7    |

注：上述参考指标由使用作者开源的代码在我们的设备上训练多次得到，由于系统环境、torch版本、CUDA版本不同等原因，与作者提供的指标可能存在略微差异。

接下来主要以`softmax_triplet_with_center.yaml`配置和训练好的模型文件为例，展示在 Market1501 数据集上进行训练、测试、推理的过程。

##### 2.1.3 数据准备

下载 [Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770) 数据集，解压到`PaddleClas/dataset/`下，并组织成以下文件结构：

  ```shell
  PaddleClas/dataset/market1501
  └── Market-1501-v15.09.15/
      ├── bounding_box_test/
      ├── bounding_box_train/
      ├── gt_bbox/
      ├── gt_query/
      ├── query/
      ├── generate_anno.py
      ├── bounding_box_test.txt
      ├── bounding_box_train.txt
      ├── query.txt
      └── readme.txt
  ```

##### 2.1.4 模型训练

1. 执行以下命令开始训练

    ```shell
    python3.7 tools/train.py -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
    ```

    注：单卡训练大约需要1个小时。

2. 查看训练日志和保存的模型参数文件

    训练过程中会在屏幕上实时打印loss等指标信息，同时会保存日志文件`train.log`、模型参数文件`*.pdparams`、优化器参数文件`*.pdopt`等内容到`Global.output_dir`指定的文件夹下，默认在`PaddleClas/output/RecModel/`文件夹下

##### 2.1.5 模型评估

准备用于评估的`*.pdparams`模型参数文件，可以使用训练好的模型，也可以使用[2.2 模型训练](#22-模型训练)中保存的模型。

- 以训练过程中保存的`latest.pdparams`为例，执行如下命令即可进行评估。

  ```shell
  python3.7 tools/eval.py \
  -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="./output/RecModel/latest"
  ```

- 以训练好的模型为例，下载 [reid_strong_baseline_softmax_with_center.epoch_120.pdparams](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/reid_strong_baseline_softmax_with_center.epoch_120.pdparams) 到 `PaddleClas/pretrained_models` 文件夹中，执行如下命令即可进行评估。

  ```shell
  # 下载模型
  cd PaddleClas
  mkdir pretrained_models
  cd pretrained_models
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/reid_strong_baseline_softmax_with_center.epoch_120.pdparams
  cd ..
  # 评估
  python3.7 tools/eval.py \
  -c ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="pretrained_models/reid_strong_baseline_softmax_with_center.epoch_120"
  ```
  注：`pretrained_model` 后填入的地址不需要加 `.pdparams` 后缀，在程序运行时会自动补上。

- 查看输出结果
  ```log
  ...
  ...
  ppcls INFO: gallery feature calculation process: [0/125]
  ppcls INFO: gallery feature calculation process: [20/125]
  ppcls INFO: gallery feature calculation process: [40/125]
  ppcls INFO: gallery feature calculation process: [60/125]
  ppcls INFO: gallery feature calculation process: [80/125]
  ppcls INFO: gallery feature calculation process: [100/125]
  ppcls INFO: gallery feature calculation process: [120/125]
  ppcls INFO: Build gallery done, all feat shape: [15913, 2048], begin to eval..
  ppcls INFO: query feature calculation process: [0/27]
  ppcls INFO: query feature calculation process: [20/27]
  ppcls INFO: Build query done, all feat shape: [3368, 2048], begin to eval..
  ppcls INFO: re_ranking=False
  ppcls INFO: [Eval][Epoch 0][Avg]recall1: 0.94270, recall5: 0.98189, mAP: 0.85799
  ```
  默认评估日志保存在`PaddleClas/output/RecModel/eval.log`中，可以看到我们提供的 `reid_strong_baseline_softmax_with_center.epoch_120.pdparams` 模型在 Market1501 数据集上的评估指标为recall@1=0.94270，recall@5=0.98189，mAP=0.85799

##### 2.1.6 模型推理部署

###### 2.1.6.1 推理模型准备
可以选择使用训练过程中保存的模型文件转换成 inference 模型并推理，或者使用我们提供的转换好的 inference 模型直接进行推理
  - 将训练过程中保存的模型文件转换成 inference 模型，同样以`latest.pdparams`为例，执行以下命令进行转换
    ```shell
    python3.7 tools/export_model.py \
    -c ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
    -o Global.pretrained_model="output/RecModel/latest" \
    -o Global.save_inference_dir="./deploy/reid_srong_baseline_softmax_with_center"
    ```

  - 或者下载并解压我们提供的 inference 模型
    ```shell
    cd PaddleClas/deploy
    wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/reid_srong_baseline_softmax_with_center.tar
    tar xf reid_srong_baseline_softmax_with_center.tar
    cd ../
    ```

###### 2.1.6.2 基于 Python 预测引擎推理

  1. 修改 `PaddleClas/deploy/configs/inference_rec.yaml`。将 `infer_imgs:` 后的字段改为 Market1501 中 query 文件夹下的任意一张图片路径（下方代码使用的是`0294_c1s1_066631_00.jpg`图片路径）；将 `rec_inference_model_dir:` 后的字段改为解压出来的 reid_srong_baseline_softmax_with_center 文件夹路径；将 `transform_ops` 字段下的预处理配置改为 `softmax_triplet_with_center.yaml` 中`Eval.Query.dataset` 下的预处理配置。如下所示

      ```yaml
      Global:
        infer_imgs: "../dataset/market1501/Market-1501-v15.09.15/query/0294_c1s1_066631_00.jpg"
        rec_inference_model_dir: "./reid_srong_baseline_softmax_with_center"
        batch_size: 1
        use_gpu: False
        enable_mkldnn: True
        cpu_num_threads: 10
        enable_benchmark: True
        use_fp16: False
        ir_optim: True
        use_tensorrt: False
        gpu_mem: 8000
        enable_profile: False

      RecPreProcess:
        transform_ops:
          - ResizeImage:
              size: [128, 256]
              return_numpy: False
              interpolation: "bilinear"
              backend: "pil"
          - ToTensor:
          - Normalize:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]

      RecPostProcess: null
      ```

  2. 执行推理命令

       ```shell
       cd PaddleClas/deploy/
       python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
       ```

  3. 查看输出结果，实际结果为一个长度2048的向量，表示输入图片经过模型转换后得到的特征向量

       ```shell
       0294_c1s1_066631_00.jpg:        [ 0.01806974  0.00476423 -0.00508293 ...  0.03925538  0.00377574
        -0.00849029]
       ```
        推理时的输出向量储存在[predict_rec.py](../../../deploy/python/predict_rec.py#L134-L135)的`result_dict`变量中。

  4. 批量预测
    将配置文件中`infer_imgs:`后的路径改为为文件夹即可，如`../dataset/market1501/Market-1501-v15.09.15/query`，则会预测并输出出query下所有图片的特征向量。

###### 2.1.6.3 基于 C++ 预测引擎推理

PaddleClas 提供了基于 C++ 预测引擎推理的示例，您可以参考[服务器端 C++ 预测](../inference_deployment/cpp_deploy.md)来完成相应的推理部署。如果您使用的是 Windows 平台，可以参考基于 Visual Studio 2019 Community CMake 编译指南完成相应的预测库编译和模型预测工作。

##### 2.1.7 服务化部署

Paddle Serving 提供高性能、灵活易用的工业级在线推理服务。Paddle Serving 支持 RESTful、gRPC、bRPC 等多种协议，提供多种异构硬件和多种操作系统环境下推理解决方案。更多关于Paddle Serving 的介绍，可以参考Paddle Serving 代码仓库。

PaddleClas 提供了基于 Paddle Serving 来完成模型服务化部署的示例，您可以参考[模型服务化部署](../inference_deployment/paddle_serving_deploy.md)来完成相应的部署工作。

##### 2.1.8 端侧部署

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。更多关于 Paddle Lite 的介绍，可以参考Paddle Lite 代码仓库。

PaddleClas 提供了基于 Paddle Lite 来完成模型端侧部署的示例，您可以参考[端侧部署](../inference_deployment/paddle_lite_deploy.md)来完成相应的部署工作。

##### 2.1.9 Paddle2ONNX 模型转换与预测

Paddle2ONNX 支持将 PaddlePaddle 模型格式转化到 ONNX 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。更多关于 Paddle2ONNX 的介绍，可以参考Paddle2ONNX 代码仓库。

PaddleClas 提供了基于 Paddle2ONNX 来完成 inference 模型转换 ONNX 模型并作推理预测的示例，您可以参考[Paddle2ONNX 模型转换与预测](../../../deploy/paddle2onnx/readme.md)来完成相应的部署工作。

### 3. 总结

#### 3.1 方法总结与对比

上述算法能快速地迁移至多数的ReID模型中，能进一步提升ReID模型的性能。

#### 3.2 使用建议/FAQ

Market1501 数据集比较小，可以尝试训练多次取最高精度。

### 4. 参考资料

1. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
2. [michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)
3. [行人重识别数据集之 Market1501 数据集_star_function的博客-CSDN博客_market1501数据集](https://blog.csdn.net/qq_39220334/article/details/121470106)
4. [Deep Learning for Person Re-identification:A Survey and Outlook](https://arxiv.org/abs/2001.04193)
