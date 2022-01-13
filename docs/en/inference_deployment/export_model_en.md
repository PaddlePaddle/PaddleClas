# Export model

PaddlePaddle supports exporting inference model for deployment. Compared with training, inference model files store network weights and network structures persistently, and PaddlePaddle supports more fast prediction engine loading inference model to deployment.

---

## Catalogue

- [1. Environmental preparation](#1)
- [2. Export classification model](#2)
- [3. Export mainbody detection model](#3)
- [4. Export recognition model](#4)
- [5. Parameter description](#5)


<a name="1"></a>

## 1. Environmental preparation

First, refer to the [Installing PaddlePaddle](../installation/install_paddle_en.md) and the [Installing PaddleClas](../installation/install_paddleclas_en.md) to prepare environment.

<a name="2"></a>

## 2. Export classification model

Change the working directory to PaddleClas:

```shell
cd /path/to/PaddleClas
```

Taking the classification model ResNet50_vd as an example, download the pre-trained model:

```shell
wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
```

The above model weights is trained by ResNet50_vd model on ImageNet1k dataset and training configuration file is `ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml`. To export the inference model, just run the following command:

```shell
python tools/export_model.py
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml \
    -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained \
    -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

<a name="3"></a>

## 3. Export mainbody detection model

About exporting mainbody detection model in details, please refer[mainbody detection](../image_recognition_pipeline/mainbody_detection_en.md).

<a name="4"></a>

## 4. Export recognition model

Change the working directory to PaddleClas:

```shell
cd /path/to/PaddleClas
```

Take the feature extraction model in products recognition as an example, download the pretrained model:

```shell
wget -P ./product_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams
```

The above model weights file is trained by ResNet50_vd on AliProduct dataset, and the training configuration file is `ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml`. The command to export inference model is as follow:

```shell
python3 tools/export_model.py \
    -c ./ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml \
    -o Global.pretrained_model=./product_pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained \
    -o Global.save_inference_dir=./deploy/models/product_ResNet50_vd_aliproduct_v1.0_infer
```

Notice, the inference model exported by above command is truncated on embedding layer, so the output of the model is n-dimensional embedding feature.

<a name="5"></a>

## 5. Parameter description

In the above model export command, the configuration file used must be the same as the training configuration file. The following fields in the configuration file are used to configure exporting model parameters.

* `Global.image_shape`：To specify the input data size of the model, which does not contain the batch dimension;
* `Global.save_inference_dir`：To specify directory of saving inference model files exported;
* `Global.pretrained_model`：To specify the path of model weight file saved during training. This path does not need to contain the suffix `.pdparams` of model weight file;

The exporting model command will generate the following three files:

* `inference.pdmodel`：To store model network structure information;
* `inference.pdiparams`：To store model network weight information;
* `inference.pdiparams.info`：To store the parameter information of the model, which can be ignored in the classification model and recognition model;

The inference model exported is used to deployment by using prediction engine. You can refer the following docs according to different deployment modes / platforms

* [Python inference](./python_deploy_en.md)
* [C++ inference](./cpp_deploy_en.md)(Only support classification)
* [Python Whl inference](./whl_deploy_en.md)(Only support classification)
* [PaddleHub Serving inference](./paddle_hub_serving_deploy_en.md)(Only support classification)
* [PaddleServing inference](./paddle_serving_deploy_en.md)
* [PaddleLite inference](./paddle_lite_deploy_en.md)(Only support classification)
