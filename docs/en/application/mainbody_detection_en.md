# Mainbody Detection

The mainbody detection technology is currently a very widely used detection technology, which refers to the detect one or more mainbody objects in the picture, crop the corresponding area in the image and carry out recognition, thereby completing the entire recognition process. Mainbody detection is the first step of the recognition task, which can effectively improve the recognition accuracy.


This tutorial will introduce the dataset and model training for mainbody detection in PaddleClas.


## 1. Dataset

The datasets we used for mainbody detection task are shown in the following table.


| Dataset       | Image number   | Image number used in mainbody detection   | Scenarios  | Dataset link |
| ------------  | ------------- | -------| ------- | -------- |
| Objects365 | 170W | 6k | General Scenarios | [link](https://www.objects365.org/overview.html) |
| COCO2017 | 12W | 5k  | General Scenarios | [link](https://cocodataset.org/) |
| iCartoonFace | 2k | 2k | Cartoon Face | [link](https://github.com/luxiangju-PersonAI/iCartoonFace) |
| LogoDet-3k | 3k | 2k | Logo | [link](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
| RPC | 3k | 3k  | Product | [link](https://rpc-dataset.github.io/) |


In the actual training process, all datasets are mixed together. Categories of all the labeled boxes are modified to the category `foreground`, and the detection model we trained just contains one category (`foreground`).

## 2. Model Training


There are many types of object detection methods such as the commonly used two-stage detectors (FasterRCNN series, etc.), single-stage detectors (YOLO, SSD, etc.), anchor-free detectors (FCOS, etc.) and so on.

PP-YOLO由[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)提出，从骨干网络、数据增广、正则化策略、损失函数、后处理等多个角度对yolov3模型进行深度优化，最终在"速度-精度"方面达到了业界领先的水平。具体地，优化的策略如下。



PP-YOLO is proposed by [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). It deeply optimizes the yolov3 model from multiple perspectives such as backbone, data augmentation, regularization strategy, loss function, and post-processing. Finally, it reached the state of the art in terms of "speed-precision". Specifically, the optimization strategy is as follows.

- Better backbone: ResNet50vd-DCN
- Larger training batch size: 8 GPUs and mini-batch size as 24 on each GPU
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- Better ImageNet pretrain weights

For more information about PP-YOLO, you can refer to [PP-YOLO tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.1/configs/ppyolo/README.md)


In the mainbody detection task, we use `ResNet50vd-DCN` as our backbone for better performance. The config file is [ppyolov2_r50vd_dcn_365e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml) used for the model training, in which the dagtaset path is modified to the mainbody detection dataset.
The final inference model can be downloaded here: [link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar).
