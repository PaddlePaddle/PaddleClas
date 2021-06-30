# Mainbody Detection

The mainbody detection technology is currently a very widely used detection technology, which refers to the detect one or some mainbody objects in the picture, crop the corresponding area in the image and carry out recognition, thereby completing the entire recognition process. Mainbody detection is the first step of the recognition task, which can effectively improve the recognition accuracy.


This tutorial will introduce the dataset and model training for mainbody detection in PaddleClas.


## 1. Dataset

The datasets we used for mainbody detection task are shown in the following table.


| Dataset       | Image number   | Image number used in <<br>>mainbody detection   | Scenarios  | Dataset link |
| ------------  | ------------- | -------| ------- | -------- |
| Objects365 | 170W | 6k | General Scenarios | [link](https://www.objects365.org/overview.html) |
| COCO2017 | 12W | 5k  | General Scenarios | [link](https://cocodataset.org/) |
| iCartoonFace | 2k | 2k | Cartoon Face | [link](https://github.com/luxiangju-PersonAI/iCartoonFace) |
| LogoDet-3k | 3k | 2k | Logo | [link](https://github.com/Wangjing1551/LogoDet-3K-Dataset) |
| RPC | 3k | 3k  | Product | [link](https://rpc-dataset.github.io/) |


In the actual training process, all datasets are mixed together. Categories of all the labeled boxes are modified to the category `foreground`, and the detection model we trained just contains one category (`foreground`).

## 2. Model Selection


There are many types of object detection methods such as the commonly used two-stage detectors (FasterRCNN series, etc.), single-stage detectors (YOLO, SSD, etc.), anchor-free detectors (FCOS, etc.) and so on.

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
The final inference model can be downloaded [here](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar).


## 3. Model training

This section mainly talks about how to train your own mainbody detection model using PaddleDetection on your own dataset.

### 3.1 Prepare for the environment

Download PaddleDetection and install requirements。

```shell
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
# install requirements
pip install -r requirements.txt
```

For more installation tutorials, please refer to [Installation tutorial]()

更多安装教程，请参考: [安装文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/docs/tutorials/INSTALL.md)

### 3.2 Prepare for the dataset

For customized dataset, you should convert it to COCO format. Please refer to [Customized dataset tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/static/docs/tutorials/Custom_DataSet.md) to build your own dataset with COCO format.

In mainbody detection task, all the objects belong to foregroud. Therefore, `category_id` of all the objects in the annotation file should be modified to 1. And the `categories` map should be modified as follows, in which just class `foregroud` is included.

```json
[{u'id': 1, u'name': u'foreground', u'supercategory': u'foreground'}]
```

### 3.3 Configuration files

You can use `configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml` to train the model, mode details are as follows.

<div align='center'>
  <img src='../../images/det/PaddleDetection_config.png' width='400'/>
</div>

`ppyolov2_r50vd_dcn_365e_coco.yml` depends on other configuration files, their meanings are as follows.


```
coco_detection.yml：num_class of the model, and train/eval/test dataset.

runtime.yml：public runtime parameters, use_gpu, save_interval, etc.

optimizer_365e.yml：learning rate and optimizer.

ppyolov2_r50vd_dcn.yml：model architecture.

ppyolov2_reader.yml：train/eval/test reader.
```

In mainbody detection task, you need to modify `num_classes` in `datasets/coco_detection.yml` to 1 (just `foreground` is included). Dataset path should also be updated.


### 3.4 Begin the training process


PaddleDetection supports many ways of training process.

* Training using single GPU

```bash
# not needed for windows and Mac
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml
```

* Training using multiple GPU's

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval
```

--eval：eval during training

* Resume training: you can use `-r` to load checkpoints and resume training.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --eval -r output/ppyolov2_r50vd_dcn_365e_coco/10000
```

Note:
If error `out of memory` occured, you can try to decrease `batch_size` in `ppyolov2_reader.yml`.


### 3.5 Model prediction

Use the following command to finish the prediction process.



```bash
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --infer_img=your_image_path.jpg --output_dir=infer_output/ --draw_threshold=0.5 -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final
```

`--draw_threshold` is an optional parameter.

### 3.6 Export model and inference.

Use the following to export the inference model.

```bash
python tools/export_model.py -c configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml --output_dir=./inference -o weights=output/ppyolov2_r50vd_dcn_365e_coco/model_final.pdparams
```

The inference model will be saved folder `inference/ppyolov2_r50vd_dcn_365e_coco`, which contains `model.pdiparams`, `model.pdiparams.info`,`model.pdmodel` and `infer_cfg.yml`(optional for mainbody detection).

* Note: Inference model name that `PaddleDetection` exports is `model.xxx`, here if you want to keep it consistent with `PaddleClas`, you can rename `model.xxx` to `inference.xxx` for subsequent inference.

For more model export tutorial, please refer to [EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md).

Now you get the newest model on your own dataset. In the recognition process, you can replace the detection model path with yours. For quick start of recognition process, please refer to the [tutorial](../tutorials/quick_start_recognition_en.md).
