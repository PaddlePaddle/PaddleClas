# 主体检测


主体检测技术是目前应用非常广泛的一种检测技术，它指的是检测出图片中一个或者多个主体的坐标位置，然后将图像中的对应区域裁剪下来，进行识别，从而完成整个识别过程。主体检测是识别任务的前序步骤，可以有效提升识别精度。

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


## 2. 模型选择

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


## 3. 模型训练

本节主要介绍怎样基于PaddleDetection，基于自己的数据集，训练主体检测模型。

### 3.1 环境准备

下载PaddleDetection代码，安装requirements。

```shell
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
# 安装其他依赖
pip install -r requirements.txt
```

更多安装教程，请参考: [安装文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL_cn.md)

### 3.2 数据准备

对于自定义数据集，首先需要将自己的数据集修改为COCO格式，可以参考该[自定义检测数据集教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/static/docs/tutorials/Custom_DataSet.md)制作COCO格式的数据集。

主体检测任务中，所有的检测框均属于前景，在这里需要将标注文件中，检测框的`category_id`修改为1，同时将整个系统中的`categories`映射表修改为下面的格式，即整个类别映射表中只包含`前景`类别。

```json
[{u'id': 1, u'name': u'foreground', u'supercategory': u'foreground'}]
```

<a name="配置文件改动和说明"></a>

### 3.3 配置文件改动和说明

我们使用 `configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml`配置进行训练，配置文件摘要如下：

<div align='center'>
  <img src='../../images/det/PaddleDetection_config.png' width='400'/>
</div>

从上图看到 `ppyolov2_r50vd_dcn_365e_coco.yml` 配置需要依赖其他的配置文件，这些配置文件的含义如下:

```
coco_detection.yml：主要说明了训练数据和验证数据的路径

runtime.yml：主要说明了公共的运行参数，比如是否使用GPU、每多少个epoch存储checkpoint等

optimizer_365e.yml：主要说明了学习率和优化器的配置

ppyolov2_r50vd_dcn.yml：主要说明模型和主干网络的情况

ppyolov2_reader.yml：主要说明数据读取器配置，如batch size，并发加载子进程数等，同时包含读取后预处理操作，如resize、数据增强等等
```
在主体检测任务中，需要将`datasets/coco_detection.yml`中的`num_classes`参数修改为1（只有1个前景类别），同时将训练集和测试集的路径修改为自定义数据集的路径。

此外，也可以根据实际情况，修改上述文件，比如，如果显存溢出，可以将batch size和学习率等比缩小等。


### 3.4 启动训练

PaddleDetection提供了单卡/多卡训练模式，满足用户多种训练需求。

* GPU 单卡训练

```bash
# windows和Mac下不需要执行该命令
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml
```

* GPU多卡训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval
```

--eval：表示边训练边验证

* 模型恢复训练

在日常训练过程中，有的用户由于一些原因导致训练中断，可以使用-r的命令恢复训练:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval -r output/ppyolov2_r50vd_dcn_365e_coco/10000
```

注意：如果遇到 "`Out of memory error`" 问题, 尝试在 `ppyolov2_reader.yml` 文件中调小`batch_size`


### 3.5 模型预测与调试

使用下面的命令完成PaddleDetection的预测过程。

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --infer_img=your_image_path.jpg --output_dir=infer_output/ --draw_threshold=0.5 -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final
```

`--draw_threshold` 是个可选参数. 根据 [NMS](https://ieeexplore.ieee.org/document/1699659) 的计算，不同阈值会产生不同的结果 `keep_top_k`表示设置输出目标的最大数量，默认值为100，用户可以根据自己的实际情况进行设定。

### 3.6 模型导出与预测部署。

执行导出模型脚本：

```bash
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --output_dir=./inference -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams
```

预测模型会导出到`inference/ppyolov2_r50vd_dcn_365e_coco`目录下，分别为`infer_cfg.yml`(预测不需要), `model.pdiparams`, `model.pdiparams.info`,`model.pdmodel` 。

注意：`PaddleDetection`导出的inference模型的文件格式为`model.xxx`，这里如果希望与PaddleClas的inference模型文件格式保持一致，需要将其`model.xxx`文件修改为`inference.xxx`文件，用于后续主体检测的预测部署。

更多模型导出教程，请参考：[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md)

导出模型之后，在主体检测与识别任务中，就可以将检测模型的路径更改为该inference模型路径，完成预测。图像识别快速体验可以参考：[图像识别快速开始教程](../tutorials/quick_start_recognition.md)。
