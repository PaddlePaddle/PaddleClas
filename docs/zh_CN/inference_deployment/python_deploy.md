# Python 预测推理

---

首先请参考文档[安装 PaddlePaddle](../installation/install_paddle.md)和文档[安装 PaddleClas](../installation/install_paddleclas.md)配置运行环境。

## 目录

- [1. 图像分类及PULC模型推理](#1)
- [2. PP-ShiTu推理](#2)
	- [2.1 主体检测模型推理](#2.1)
	- [2.2 特征提取模型推理](#2.2)
	- [2.3 主体检测、特征提取和向量检索串联](#2.3)

<a name="1"></a>
## 1. 图像分类及PULC模型推理

图像分类模型及PULC模型都是分类模型，因此推理方式一致。
首先请参考文档[模型导出](./export_model.md)准备 inference 模型，然后进入 PaddleClas 的 `deploy` 目录下：

```shell
cd /path/to/PaddleClas/deploy
```

使用以下命令进行预测：

```shell
python python/predict_cls.py -c configs/inference_cls.yaml
```

在配置文件 `configs/inference_cls.yaml` 中有以下字段用于配置预测参数：
* `Global.infer_imgs`：待预测的图片文件路径；
* `Global.inference_model_dir`：inference 模型文件所在目录，该目录下需要有文件 `inference.pdmodel` 和 `inference.pdiparams` 两个文件；
* `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认为 `False`；
* `Global.use_gpu`：是否使用 GPU 预测，默认为 `True`；
* `Global.enable_mkldnn`：是否启用 `MKL-DNN` 加速库，默认为 `False`。注意 `enable_mkldnn` 与 `use_gpu` 同时为 `True` 时，将忽略 `enable_mkldnn`，而使用 GPU 预测；
* `Global.use_fp16`：是否启用 `FP16`，默认为 `False`；
* `PreProcess`：用于数据预处理配置；
* `PostProcess`：由于后处理配置；
* `PostProcess.Topk.class_id_map_file`：数据集 label 的映射文件，默认为 `./utils/imagenet1k_label_list.txt`，该文件为 PaddleClas 所使用的 ImageNet 数据集 label 映射文件。

**注意**:
* 如果使用 VisionTransformer 系列模型，如 `DeiT_***_384`, `ViT_***_384` 等，请注意模型的输入数据尺寸，部分模型需要修改参数： `PreProcess.resize_short=384`, `PreProcess.resize=384`。
* 如果你希望提升评测模型速度，使用 GPU 评测时，建议开启 TensorRT 加速预测，使用 CPU 评测时，建议开启 MKL-DNN 加速预测。

<a name="2"></a>
## 2. PP-ShiTu推理

PP-ShiTu主要分为三大部分：主体检测模型、特征提取模型及检索模块。其中主体检测模型和特征提取模型可以单独使用。不同的模块及整体推理方式如下：

<a name="2.1"></a>
### 2.1 主体检测模型推理

进入 PaddleClas 的 `deploy` 目录下：

```shell
cd /path/to/PaddleClas/deploy
```

准备 PaddleClas 提供的主体检测 inference 模型：

```shell
mkdir -p models
# 下载通用检测 inference 模型并解压
wget -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar
tar -xf ./models/ppyolov2_r50vd_dcn_mainbody_v1.0_infer.tar -C ./models/
```

使用以下命令进行预测：

```shell
python python/predict_det.py -c configs/inference_det.yaml
```

在配置文件 `configs/inference_det.yaml` 中有以下字段用于配置预测参数：
* `Global.infer_imgs`：待预测的图片文件路径；
* `Global.use_gpu`： 是否使用 GPU 预测，默认为 `True`。


<a name="2.2"></a>
### 2.2 特征提取模型推理

下面以商品特征提取为例，介绍特征提取模型推理。首先进入 PaddleClas 的 `deploy` 目录下：

```shell
cd /path/to/PaddleClas/deploy
```

准备 PaddleClas 提供的商品特征提取 inference 模型：

```shell
mkdir -p models
# 下载商品特征提取 inference 模型并解压
wget -P ./models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/product_ResNet50_vd_aliproduct_v1.0_infer.tar
tar -xf ./models/product_ResNet50_vd_aliproduct_v1.0_infer.tar -C ./models/
```

上述预测命令可以得到一个 512 维的特征向量，直接输出在在命令行中。

<a name="2.3"></a>
### 2.3 主体检测、特征提取和向量检索串联

PP-ShiTu(主体检测、特征提取和向量检索的串联预测)，可以参考图像识别[快速体验](../quick_start/quick_start_recognition.md)。
