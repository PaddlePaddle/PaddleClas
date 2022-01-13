# Mainbody Detection

The mainbody detection technology is currently a widely used detection technology, which refers to a whole image recognition process of identifying the coordinate position of one or more objects and then cropping down the corresponding area for recognition. Mainbody detection is the first step of the recognition task, which can effectively improve the recognition accuracy.

This tutorial will introduce the technology from three aspects, namely, the datasets, model selection and model training.

## Catalogue

- [1. Dataset](#1)
- [2. Model Selection](#2)
  - [2.1 Lightweight Mainbody Detection Model](#2.1)
  - [2.2 Server-side Mainbody Detection Model](#2.2)
- [3. Model Training](#3)
  - [3.1 Prepare For the Environment](#3.1)
  - [3.2 Prepare For the Dataset](#3.2)
  - [3.3 Configuration Files](#3.3)
  - [3.4 Begin the Training Process](#3.4)
  - [3.5 Model Prediction](#3.5)
  - [3.6 Model Export and Inference Deployment](#3.6)

<a name="1"></a>

## 1. Dataset

The datasets we used for mainbody detection tasks are shown in the following table.

| Dataset      | Image Number | Image Number Used in Mainbody Detection | Scenarios         | Dataset Link                                               |
| ------------ | ------------ | --------------------------------------- | ----------------- | ---------------------------------------------------------- |
| Objects365   | 170W         | 6k                                      | General Scenarios | [Link](https://www.objects365.org/overview.html)           |
| COCO2017     | 12W          | 5k                                      | General Scenarios | [Link](https://cocodataset.org/)                           |
| iCartoonFace | 2k           | 2k                                      | Cartoon Face      | [Link](https://github.com/luxiangju-PersonAI/iCartoonFace) |
| LogoDet-3k   | 3k           | 2k                                      | Logo              | [Link](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
| RPC          | 3k           | 3k                                      | Product           | [Link](https://rpc-dataset.github.io/)                     |

In the actual training process, all datasets are mixed together. Categories of all the labeled boxes are modified as `foreground`, and the detection model we trained only contains one category (`foreground`).

<a name="2"></a>

## 2. Model Selection

There are a wide variety of object detection methods, such as the commonly used two-stage detectors (FasterRCNN series, etc.), single-stage detectors (YOLO, SSD, etc.), anchor-free detectors (FCOS, etc.) and so on. PaddleDetection has its self-developed PP-YOLO models for server-side scenarios and PicoDet models for end-side scenarios (CPU and mobile), which all take the lead in the area.

Build on the studies above, PaddleClas provides lightweight and server-side main body detection models for end-side scenarios and server-side scenarios respectively. The table below presents the average mAP of the 5 datasets and the comparison of their model sizes and inference speed.

| Model                                | Model Structure | Download Link of Pre-trained Model                           | Download Link of Inference Model                             | mAP   | Size of Inference Model (MB) | Inference Time per Image (preprocessing excluded)(ms) |
| ------------------------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | ---------------------------- | ----------------------------------------------------- |
| Lightweight Mainbody Detection Model | PicoDet         | [Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_pretrained.pdparams) | [Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar) | 40.1% | 30.1                         | 29.8                                                  |
| Server-side Mainbody Detection Model | PP-YOLOv2       | [Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/ppyolov2_r50vd_dcn_mainbody_v1.0_pretrained.pdparams) | [Link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar) | 42.5% | 210.5                        | 466.6                                                 |

Notes:

- Detailed information of the CPU of the speed evaluation machine：`Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz`.The speed indicator is the testing result when mkldnn is on and the number of threads is set to 10.
- Mainbody detection has a time-consuming preprocessing procedure, with an average time of about 40 to 55 ms per image in the above machine. Therefore, it is not included in the inference time.

<a name="2.1"></a>

### 2.1 Lightweight Mainbody Detection Model

PicoDet, introduced by  [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), is an object detection algorithm applied to CPU or mobile-side scenarios. It integrates the following optimization algorithm.

- [ATSS](https://arxiv.org/abs/1912.02424)
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388)
- Cosine learning rate decay
- Cycle-EMA
- Lightweight detection head

For more details of optimized PicoDet and benchmark,  you can refer to [Tutorials of PicoDet Models](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/picodet/README.md).

To balance the detection speed and effects in lightweight mainbody detection tasks, we adopt PPLCNet_x2_5 as the backbone of the model and revise the image scale for training and inference to 640x640, with the rest configured the same as [picodet_m_shufflenetv2_416_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/picodet/picodet_m_shufflenetv2_416_coco.yml). The final detection model is obtained after the training of customized mainbody detection datasets.

<a name="2.2"></a>

### 2.2 Server-side Mainbody Detection Model

PP-YOLO is proposed by [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection). It greatly optimizes the yolov3 model from multiple perspectives such as backbone, data augmentation, regularization strategy, loss function, and post-processing. It reaches the state of the art in terms of "speed-precision". The optimization strategy is as follows.

- Better backbone: ResNet50vd-DCN
- Larger training batch size of 8 GPUs and mini-batch size of 24 on each GPU, which is corresponding to learning rate and the number of iterations.
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- Better Pre-trained Model

For more information about PP-YOLO, you can refer to [PP-YOLO tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release%2F2.1/configs/ppyolo/README.md).

In the mainbody detection task, we use `ResNet50vd-DCN` as our backbone for better performance. The config file is [ppyolov2_r50vd_dcn_365e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml), in which the dataset path is modified to the customized mainbody detection dataset. The final detection model can be downloaded [here](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar).

<a name="3"></a>

## 3. Model Training

This section mainly talks about how to train your own mainbody detection model using PaddleDetection on your own datasets.

<a name="3.1"></a>

###  3.1 Prepare For the Environment

Download PaddleDetection and install requirements.

```shell
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
# install requirements
pip install -r requirements.txt
```

For more installation tutorials, please refer to [Installation Tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL.md)

<a name="3.2"></a>

### 3.2 Prepare For the Dataset

For customized dataset, you should convert it to COCO format. Please refer to [Customized Dataset Tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/static/docs/tutorials/Custom_DataSet.md) to build your own datasets with COCO format.

In mainbody detection task, all the objects belong to foregroud. Therefore, `category_id` of all the objects in the annotation file should be modified to 1. And the `categories` map should be modified as follows, in which just class `foregroud` is included.

```
[{u'id': 1, u'name': u'foreground', u'supercategory': u'foreground'}]
```

<a name="3.3"></a>

### 3.3 Configuration Files

We use `configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml` to train the model, mode details are as follows.

  [![img](../../images/det/PaddleDetection_config.png)](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/images/det/PaddleDetection_config.png)



`ppyolov2_r50vd_dcn_365e_coco.yml` depends on other configuration files, their meanings are as follows.

```
coco_detection.yml：path of train/eval/test dataset.

runtime.yml：public runtime parameters, including whethre to use GPU, epoch number for checkpoint saving, etc.

optimizer_365e.yml：learning rate and optimizer.

ppyolov2_r50vd_dcn.yml：model architecture and backbone.

ppyolov2_reader.yml：train/eval/test reader, such as batch size, the number of concurrently loaded sub-processes, etc., and includes post-read pre-processing operations, such as resize, data enhancement, etc.
```

In mainbody detection task, you need to modify `num_classes` in `datasets/coco_detection.yml` to 1 (only `foreground` is included), while modify the paths of the training and testing datasets to those of the customized datasets.

In addition, the above files can also be modified according to real situations, for example, if the video memory is overflowing, the batch size and learning rate can be reduced in equal proportion.

<a name="3.4"></a>

### 3.4 Begin the Training Process

PaddleDetection supports many ways of training process.

- Training using single GPU

```
# not needed for windows and Mac
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml
```

- Training using multiple GPUs

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval
```

--eval: evaluation while training

- (**Recommend**) Model finetune If you want to finetune the trained model in PaddleClas on your own datasets, you can run the following command.

```
export CUDA_VISIBLE_DEVICES=0
# assign pretrain_weights, load the general mainbody-detection pretrained model
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml -o pretrain_weights=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/ppyolov2_r50vd_dcn_mainbody_v1.0_pretrained.pdparams
```

- Resume training

  you can use `-r` to load checkpoints and resume training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval -r output/ppyolov2_r50vd_dcn_365e_coco/10000
```

Note: If `Out of memory error` occurs, you can try to decrease `batch_size` in `ppyolov2_reader.yml` while reducing learning rate in equal proportion.

<a name="3.5"></a>

### 3.5 Model Prediction

Use the following command to finish the prediction process.

```
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --infer_img=your_image_path.jpg --output_dir=infer_output/ --draw_threshold=0.5 -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final
```

`--draw_threshold` is an optional parameter. According to NMS calculation, different thresholds will produce different results.  `keep_top_k`  indicates the maximum number of output targets, with a default value of 100 that can be modified according to their actual situation.

<a name="3.6"></a>

### 3.6 Model Export and Inference Deployment

Use the following to export the inference model：

```
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --output_dir=./inference -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams
```

The inference model will be saved under the directory `inference/ppyolov2_r50vd_dcn_365e_coco`, which contains`infer_cfg.yml` (optional for mainbody detection), `model.pdiparams`, `model.pdiparams.info`, `model.pdmodel`.

Note： Inference model that `PaddleDetection` exports is named `model.xxx`，if you want to keep it consistent with PaddleClas，you can rename `model.xxx` to `inference.xxx` for subsequent inference deployment of mainbody detection.

For more model export tutorials, please refer to [EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md).

The final directory contains `inference/ppyolov2_r50vd_dcn_365e_coco`,  `inference.pdiparams`, `inference.pdiparams.info`, and `inference.pdmodel`，among which`inference.pdiparams` refers to saved weight files of the inference model while `inference.pdmodel` stands for structural files.

After exporting the model, the path of the detection model can be changed to the inference model path to complete the prediction task.

Take product recognition as an example，you can modify the field `Global.det_inference_model_dir` in its config file [inference_product.yaml](../../../deploy/configs/inference_product.yaml) to the directory of exported inference model, and then finish the detection and recognition of the product with reference to  [Quick Start for Image Recognition](../quick_start/quick_start_recognition_en.md).

## FAQ

#### Q: Is it compatible with other mainbody detection models？

- A: Yes, but the current preprocessing process only supports PicoDet and YOLO models, so it is recommended to use these two for training. If you want to use other models such as Faster RCNN, you need to revise the logic of preprocessing in accordance with that of PaddleDetection. You are welcomed to resort to Github Issue or WeChat group for any needs or questions.

#### Q: Can I modify the prediction scale of mainbody detection?

- A: Yes, but there are 2 things that require attention
  - The mainbody detection model provided in PaddleClas is trained based on `640x640`  resolution, so this is also the default value of prediction process. The accuracy will be reduced if other resolutions are used.
  - When exporting the model, it is recommended to modify the resolution of the exported model to keep it consistent with the prediction process.
