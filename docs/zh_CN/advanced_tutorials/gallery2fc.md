# 识别模型转分类模型

PaddleClas 提供了 `gallery2fc.py` 工具，帮助大家将识别模型转为分类模型。目前该工具仅支持转换量化后模型，因此建议使用 PaddleClas 提供的 `general_PPLCNet_x2_5_pretrained_v1.0_quant` 预训练模型，该模型为量化后的通用识别模型，backbone 为 PPLCNet_x2_5。

如需使用其他模型，关于量化的具体操作请参考文档 [模型量化](./model_prune_quantization.md)。

## 一、模型转换说明

### 1.1 准备底库数据、预训练模型

#### 1. 底库数据集

首先需要准备好底库数据，下面以 PaddleClas 提供的饮料数据集（drink_dataset_v1.0）为例进行说明，饮料数据集获取方法：

```shell
cd PaddleClas/
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
tar -xf drink_dataset_v1.0.tar
```

饮料数据集的底库图片路径为 `drink_dataset_v1.0/gallery/`，底库图片列表可在 `drink_dataset_v1.0/gallery/drink_label.txt` 中查看，关于底库数据格式说明，请参考文档[数据集格式说明](../data_preparation/recognition_dataset.md#1-数据集格式说明)。

#### 2. 预训练模型

在开始转换模型前，需要准备好预训练模型，下面以量化后的 `general_PPLCNet_x2_5` 模型为例，下载预训练模型：

```shell
cd PaddleClas/pretrained/
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/general_PPLCNet_x2_5_pretrained_v1.0_quant.pdparams
```

### 1.2 准备配置文件

在进行模型转换时，需要通过配置文件定义所需参数，本例中所用配置文件为 `ppcls/configs/GeneralRecognition/Gallery2FC_PPLCNet_x2_5.yaml`，对于配置文件字段的说明，如下所示：

* Global:
    * pretrained_model: 预训练模型路径，无需包含 `.pdparams` 后缀名；
    * image_shape: 模型输入数据尺寸，无需包含 batch size 维度；
    * save_inference_dir: 转换后模型的保存路径；
* Arch: 模型结构相关定义，可参考 [配置说明](../models_training/config_description.md#3-%E8%AF%86%E5%88%AB%E6%A8%A1%E5%9E%8B)；
* IndexProcess: 底库数据集相关定义
    * image_root: 底库数据集路径；
    * data_file: 底库数据集列表文件路径；

### 1.3 模型转换

在完成上述准备工作后，即可进行模型转换，命令如下所示：

```python
python ppcls/utils/gallery2fc.py -c ppcls/configs/GeneralRecognition/Gallery2FC_PPLCNet_x2_5.yaml
```

在上述命令执行完成后，转换并导出的模型保存在目录 `./inference/general_PPLCNet_x2_5_quant/` 下。在推理部署时，需要注意的是，模型的输出结果通常有多个，应选取分类结果作为模型输出，需要注意区分。
