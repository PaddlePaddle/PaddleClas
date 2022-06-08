# 模型导出

PaddlePaddle 支持导出 inference 模型用于部署推理场景，相比于训练调优场景，inference 模型会将网络权重与网络结构进行持久化存储，并且 PaddlePaddle 支持使用预测引擎加载 inference 模型进行预测推理。

---


## 目录

- [1. 环境准备](#1)
- [2. 分类模型导出](#2)
- [3. 主体检测模型导出](#3)
- [4. 识别模型导出](#4)
- [5. 命令参数说明](#5)


<a name="1"></a>
## 1. 环境准备

首先请参考文档文档[环境准备](../installation/install_paddleclas.md)配置运行环境。

<a name="2"></a>
## 2. 分类模型导出

进入 PaddleClas 目录下：

```shell
cd /path/to/PaddleClas
```

以 ResNet50_vd 分类模型为例，下载预训练模型：

```shell
wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
```

上述模型是使用 ResNet50_vd 在 ImageNet 上训练的模型，使用的配置文件为 `ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml`，将该模型转为 inference 模型只需运行如下命令：

```shell
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml \
    -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained \
    -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer
```

<a name="3"></a>
## 3. 主体检测模型导出

主体检测模型的导出，可以参考[主题检测介绍](../image_recognition_pipeline/mainbody_detection.md)。

<a name="4"></a>
## 4. 识别模型导出

进入 PaddleClas 目录下：

```shell
cd /path/to/PaddleClas
```

以商品识别特征提取模型为例，下载预训练模型：

```shell
wget -P ./product_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams
```

上述模型是 ResNet50_vd 在 AliProduct 上训练的模型，训练使用的配置文件为 `ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml`，将该模型转为 inference 模型只需运行如下命令：

```shell
python3 tools/export_model.py \
    -c ./ppcls/configs/Products/ResNet50_vd_Aliproduct.yaml \
    -o Global.pretrained_model=./product_pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained \
    -o Global.save_inference_dir=./deploy/models/product_ResNet50_vd_aliproduct_v1.0_infer
```

注意，此处保存的 inference 模型在 embedding 特征层做了截断，即导出后模型最终的输出为 n 维 embedding 特征。

<a name="5"></a>
## 5. 命令参数说明

在上述模型导出命令中，所使用的配置文件需要与该模型的训练文件相同，在配置文件中有以下字段用于配置模型导出参数：

* `Global.image_shape`：用于指定模型的输入数据尺寸，该尺寸不包含 batch 维度；
* `Global.save_inference_dir`：用于指定导出的 inference 模型的保存位置；
* `Global.pretrained_model`：用于指定训练过程中保存的模型权重文件路径，该路径无需包含模型权重文件后缀名 `.pdparams`。。

上述命令将生成以下三个文件：

* `inference.pdmodel`：用于存储网络结构信息；
* `inference.pdiparams`：用于存储网络权重信息；
* `inference.pdiparams.info`：用于存储模型的参数信息，在分类模型和识别模型中可忽略。

导出的 inference 模型文件可用于预测引擎进行推理部署，根据不同的部署方式/平台，可参考：

* [Python 预测](./python_deploy.md)
* [C++ 预测](./cpp_deploy.md)（目前仅支持分类模型）
* [Python Whl 预测](./whl_deploy.md)（目前仅支持分类模型）
* [PaddleHub Serving 部署](./paddle_hub_serving_deploy.md)（目前仅支持分类模型）
* [PaddleServing 部署](./paddle_serving_deploy.md)
* [PaddleLite 部署](./paddle_lite_deploy.md)（目前仅支持分类模型）
