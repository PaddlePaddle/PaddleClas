# 主体检测

主体检测技术是目前应用非常广泛的一种检测技术，它指的是检测出图片中最突出的主体坐标位置，然后将图像中的对应区域裁剪下来，进行识别，从而完成整个识别过程。主体检测是识别任务的前序步骤，可以有效提升识别精度。

本部分主要从数据集、模型训练2个方面对该部分内容进行介绍。

## 1. 数据集

在PaddleClas的识别任务中，训练主体检测模型时主要用到了以下几个数据集。

| 数据集       | 数据量   | 主体检测任务中使用的数据量   | 场景  | 数据集地址 |
| ------------  | ------------- | -------| ------- | -------- |
| Objects365 | 170W | 6k | 通用场景 | [地址](https://www.objects365.org/overview.html) |
| COCO2017 | 12W | 5k  | 通用场景 | [地址](https://cocodataset.org/) |
| iCartoonFace | 2k | 2k | 动漫人脸检测 | [地址](https://github.com/luxiangju-PersonAI/iCartoonFace) |
| LogoDet-3k | 3k | 2k | Logo检测 | [地址](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
| RPC | 3k | 3k  | 商品检测 | [地址](https://rpc-dataset.github.io/) |

在实际训练的过程中，将所有数据集混合在一起。由于是主体检测，这里将所有标注出的检测框对应的类别都修改为"前景"的类别，最终融合的数据集中只包含1个类别，即前景。


## 2. 模型训练

目标检测方法种类繁多，比较常用的有两阶段检测器（如FasterRCNN系列等）；单阶段检测器（如YOLO、SSD等）；anchor-free检测器（如FCOS等）。

PP-YOLO由[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)提出，从骨干网络、数据增广、正则化策略、损失函数、后处理等多个角度对yolov3模型进行深度优化，最终在"速度-精度"方面达到了业界领先的水平。具体地，优化的策略如下。

- 更优的骨干网络: ResNet50vd-DCN
- 更大的训练batch size: 8 GPUs，每GPU batch_size=24，对应调整学习率和迭代轮数
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- 更优的预训练模型

更多关于PP-YOLO的详细介绍可以参考：[PP-YOLO 模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.1/configs/ppyolo/README_cn.md)

在主体检测任务中，为了保证检测效果，我们使用ResNet50vd-DCN的骨干网络，使用配置文件[ppyolov2_r50vd_dcn_365e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml)，更换为自定义的主体检测数据集，进行训练，最终得到检测模型。
主体检测模型的inference模型下载地址为：[链接](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar)。
