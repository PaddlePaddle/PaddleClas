简体中文|[English](../../en/algorithm_introduction/reid.md)
# ReID行人重识别

### 1. 算法/应用场景简介

行人重识别（Person re-identification）也称行人再识别，是利用[计算机视觉](https://baike.baidu.com/item/计算机视觉/2803351)技术判断[图像](https://baike.baidu.com/item/图像/773234)或者视频序列中是否存在特定行人的[技术](https://baike.baidu.com/item/技术/13014499)。广泛被认为是一个[图像检索](https://baike.baidu.com/item/图像检索/1150910)的子问题。给定一个监控行人图像，检索跨设备下的该行人图像。旨在弥补固定的摄像头的视觉局限，并可与[行人检测](https://baike.baidu.com/item/行人检测/20590256)/行人跟踪技术相结合，可广泛应用于[智能视频监控](https://baike.baidu.com/item/智能视频监控/10717227)、智能安保等领域。

### 2. ReID strong-baseline算法介绍

以往的行人重识别方法通过特征提取模块来提取图片的全局或多粒度特征，再经过融合模块得到一个高维的特征向量。在训练时使用分类头将该特征向量映射成属于每个类别的概率从而以分类任务的方式来优化整个模型；在测试或推理时直接将高维的特征向量作为图片描述子在检索库中进行检索，以得到检索结果。而ReID strong-baseline算法则是提出了多个有效优化训练与检索的方法来提升整体模型性能。

#### 2.1 ReID strong-baseline算法原理

论文出处：[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

<img src="../../images/reid/strong-baseline.jpg" width="50%">

原理介绍：作者主要使用了以下优化方法

1. Warmup，在训练一开始让学习率逐渐升高后再开始下降，有利于梯度下降时找到更优的参数。
2. Random erasing augmentation，随机区域擦除，增强模型的泛化性。
3. Label smoothing，标签平滑，增强模型的泛化性。
4. Last stride=1，取消特征提取模块的最后一个stage的下采样，增大输出特征图的分辨率来保留更多细节，增强模型的分类能力。
5. BNNeck，特征向量输入分类头之前先经过BNNeck，让特征向量变成正态分布形式，减少了同时优化ID Loss和TripLetLoss的难度。
6. Center loss，给每个类别一个可学习的聚类中心特征，训练时让类内特征靠近聚类中心，减少类内差异，增大类间差异。
7. Reranking，在检索时考虑检索图像的近邻候选对象是否同时含有检索目标，以此来优化距离矩阵，最终提升检索精度。

#### 2.2a 快速体验

快速体验章节主要以`softmax_triplet_with_center.yaml`配置和训练好的模型文件为例，在 Market1501 数据集上进行测试。

1. 下载[Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770)数据集，解压到`PaddleClas/dataset/`下，并组织成以下文件结构：

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

2. 下载 [reid_strong_baseline_softmax_with_center.epoch_120.pdparams](reid_strong_baseline_softmax_with_center.epoch_120.pdparams) 到 `PaddleClas/pretrained_models` 文件夹中

   ```shell
   cd PaddleClas
   mkdir pretrained_models
   cd pretrained_models
   wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/pretrain/reid_strong_baseline_softmax_with_center.epoch_120.pdparams
   cd ..
   ```

3. 使用下载好的 `softmax_triplet_with_center.pdparams` 在 Market1501 数据集上进行测试

   ```shell
   python3.7 tools/eval.py \
   -c ppcls/configs/reid/strong_baseline/baseline.yaml \
   -o Global.pretrained_model="pretrained_models/reid_strong_baseline_softmax_with_center.epoch_120"
   ```

4. 查看输出结果

   ```log
   ...
   [2022/06/02 03:08:07] ppcls INFO: gallery feature calculation process: [0/125]
   [2022/06/02 03:08:11] ppcls INFO: gallery feature calculation process: [20/125]
   [2022/06/02 03:08:15] ppcls INFO: gallery feature calculation process: [40/125]
   [2022/06/02 03:08:19] ppcls INFO: gallery feature calculation process: [60/125]
   [2022/06/02 03:08:23] ppcls INFO: gallery feature calculation process: [80/125]
   [2022/06/02 03:08:27] ppcls INFO: gallery feature calculation process: [100/125]
   [2022/06/02 03:08:31] ppcls INFO: gallery feature calculation process: [120/125]
   [2022/06/02 03:08:32] ppcls INFO: Build gallery done, all feat shape: [15913, 2048], begin to eval..
   [2022/06/02 03:08:33] ppcls INFO: query feature calculation process: [0/27]
   [2022/06/02 03:08:36] ppcls INFO: query feature calculation process: [20/27]
   [2022/06/02 03:08:38] ppcls INFO: Build query done, all feat shape: [3368, 2048], begin to eval..
   [2022/06/02 03:08:38] ppcls INFO: re_ranking=False
   [2022/06/02 03:08:39] ppcls INFO: [Eval][Epoch 0][Avg]recall1: 0.94270, recall5: 0.98189, mAP: 0.85799
   ```

   可以看到我们提供的 `reid_strong_baseline_softmax_with_center.epoch_120.pdparams` 模型在 Market1501 数据集上的指标为recall@1=0.94270，recall@5=0.98189，mAP=0.85799

#### 2.2b 模型训练/推理等

- 模型训练

  1. 下载[Market-1501-v15.09.15.zip](https://pan.baidu.com/s/1ntIi2Op?_at_=1654142245770)数据集，解压到`PaddleClas/dataset/`下，并组织成以下文件结构：

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

  2. 执行以下命令开始训练

     ```shell
     python3.7 tools/train.py -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml
     ```

     注：单卡训练大约需要1个小时。

- 模型测试

  假设需要测试的模型文件路径为 `./output/RecModel/latest.pdparams` ，执行下述命令即可进行测试

  ```shell
  python3.7 tools/eval.py \
  -c ./ppcls/configs/reid/strong_baseline/softmax_triplet_with_center.yaml \
  -o Global.pretrained_model="./output/RecModel/latest"
  ```

  注：`pretrained_model` 后填入的地址不需要加 `.pdparams` 后缀，在程序运行时会自动补上。

- 模型推理

  1. 下载 inference 模型并解压：[reid_srong_baseline_softmax_with_center.tar](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/reid_srong_baseline_softmax_with_center.tar)

     ```shell
     cd PaddleClas/deploy
     wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/reid/inference/reid_srong_baseline_softmax_with_center.tar
     tar xf reid_srong_baseline_softmax_with_center.tar
     ```

  2. 修改 `PaddleClas/deploy/configs/inference_rec.yaml`。将 `infer_imgs:` 后的字段改为 Market1501 中 query 文件夹下的任意一张图片；将 `rec_inference_model_dir:` 后的字段改为解压出来的 reid_srong_baseline_softmax_with_center 文件夹路径；将 `transform_ops` 字段下的预处理配置改为 `softmax_triplet_with_center.yaml` 中`Eval.Query.dataset` 下的预处理配置。如下所示

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
           interpolation: 'bilinear'
           backend: "pil"
       - ToTensor:
       - Normalize:
           mean: [0.485, 0.456, 0.406]
           std: [0.229, 0.224, 0.225]

     RecPostProcess: null
     ```

  3. 执行推理命令

     ```shell
     python3.7 python/predict_rec.py -c ./configs/inference_rec.yaml
     ```

  4. 查看输出结果，实际结果为一个长度2048的向量，表示输入图片经过模型转换后得到的特征向量

     ```shell
     0294_c1s1_066631_00.jpg:        [ 0.01806974  0.00476423 -0.00508293 ...  0.03925538  0.00377574
      -0.00849029]
     ```

### 3. 总结

#### 3.1 方法总结、对比等

以下表格总结了我们提供的ReID strong-baseline的3种配置在 Market1501 数据集上的精度指标，

| 配置文件                 | recall@1 | mAP   | 参考recall@1 | 参考mAP |
| ------------------------ | -------- | ----- | ------------ | ------- |
| baseline.yaml            | 88.21    | 74.12 | 87.7         | 74.0    |
| softmax.yaml             | 94.18    | 85.76 | 94.1         | 85.7    |
| softmax_with_center.yaml | 94.19    | 85.80 | 94.1         | 85.7    |

注：上述参考指标由使用作者开源的代码在我们的设备上训练多次得到，由于系统环境、torch版本、CUDA版本不同等原因，与作者提供的指标可能存在略微差异。

#### 3.2 使用建议/FAQ

Market1501 数据集比较小，可以尝试训练多次取最高精度。

#### 4 参考资料

1. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
2. [michuanhaohao/reid-strong-baseline: Bag of Tricks and A Strong Baseline for Deep Person Re-identification (github.com)](https://github.com/michuanhaohao/reid-strong-baseline)
3. [行人重识别数据集之 Market1501 数据集_star_function的博客-CSDN博客_market1501数据集](https://blog.csdn.net/qq_39220334/article/details/121470106)
