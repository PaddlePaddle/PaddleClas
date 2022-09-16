# Python 预测推理

首先请参考文档[环境准备](../../installation.md)配置运行环境。

## 目录

- [1. 图像分类模型推理](#1)

<a name="1"></a>

## 1. 图像分类推理

首先请参考文档[模型导出](../export_model.md)准备 inference 模型，然后进入 PaddleClas 的 `deploy` 目录下：

```shell
cd PaddleClas/deploy
```

使用以下命令进行预测：

```shell
python3.7 python/predict_cls.py -c configs/inference_cls.yaml
```

在配置文件 `configs/inference_cls.yaml` 中有以下字段用于配置预测参数：
* `Global.infer_imgs`：待预测的图片文件（夹）路径；
* `Global.inference_model_dir`：inference 模型文件所在文件夹的路径，该文件夹下需要有文件 `inference.pdmodel` 和 `inference.pdiparams` 两个文件；
* `Global.use_gpu`：是否使用 GPU 预测，默认为 `True`；
* `Global.enable_mkldnn`：是否启用 `MKL-DNN` 加速库，默认为 `False`。注意 `enable_mkldnn` 与 `use_gpu` 同时为 `True` 时，将忽略 `enable_mkldnn`，而使用 GPU 预测；
* `Global.use_fp16`：是否启用 `FP16`，默认为 `False`；
* `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认为 `False`；
* `PreProcess`：用于数据预处理配置；
* `PostProcess`：由于后处理配置；
* `PostProcess.Topk.class_id_map_file`：数据集 label 的映射文件，默认为 `../ppcls/utils/imagenet1k_label_list.txt`，该文件为 PaddleClas 所使用的 ImageNet 数据集 label 映射文件。

**注意**:
* 如果使用 VisionTransformer 系列模型，如 `DeiT_***_384`, `ViT_***_384` 等，请注意模型的输入数据尺寸，该类模型需要修改参数： `PreProcess.resize_short=384`, `PreProcess.resize=384`。
* 如果你希望提升评测模型速度，使用 GPU 评测时，建议开启 TensorRT 加速预测，使用 CPU 评测时，建议开启 MKL-DNN 加速预测。
