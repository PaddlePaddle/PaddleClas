# 开始使用
## 注意:  本文主要介绍基于检索方式的识别
---
请参考[安装指南](./install.md)配置运行环境，并根据[快速开始](./quick_start_new_user.md)文档准备flowers102数据集，本章节下面所有的实验均以flowers102数据集为例。

PaddleClas目前支持的训练/评估环境如下：
```shell
└── CPU/单卡GPU
    ├── Linux
    └── Windows

└── 多卡GPU
    └── Linux
```

## 1. 基于CPU/单卡GPU上的训练与评估

在基于CPU/单卡GPU上训练与评估，推荐使用`tools/train.py`与`tools/eval.py`脚本。关于Linux平台多卡GPU环境下的训练与评估，请参考[2. 基于Linux+GPU的模型训练与评估](#2)。

<a name="1.1"></a>
### 1.1 模型训练

准备好配置文件之后，可以使用下面的方式启动训练。

```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.use_gpu=True
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o use_gpu=True`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`use_gpu`设置为`False`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config.md)。

训练期间也可以通过VisualDL实时观察loss变化，详见[VisualDL](../extension/VisualDL.md)。

### 1.2 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Arch.Backbone.pretrained=True
```

其中`-o Arch.Backbone.pretrained`用于设置是否加载预训练模型；为True时，会自动下载预训练模型，并加载。

<a name="1.3"></a>
### 1.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
```
只需要在继续训练时设置`Global.checkpoints`参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

<a name="1.4"></a>
### 1.4 模型评估

可以通过以下命令进行模型评估。

```bash
python tools/eval.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model="./output/RecModel/best_model"\
```
其中`-o Global.pretrained_model`用于设置需要进行评估的模型的路径

<a name="2"></a>
## 2. 基于Linux+GPU的模型训练与评估

如果机器环境为Linux+GPU，那么推荐使用`paddle.distributed.launch`启动模型训练脚本（`tools/train.py`）、评估脚本（`tools/eval.py`），可以更方便地启动多卡训练与评估。

### 2.1 模型训练

参考如下方式启动模型训练，`paddle.distributed.launch`通过设置`gpus`指定GPU运行卡号：

```bash
# PaddleClas通过launch方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml
```

### 2.2 模型微调

根据自己的数据集配置好配置文件之后，可以加载预训练模型进行微调，如下所示。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
        -o Arch.Backbone.pretrained=True
```

### 2.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
        -o Global.checkpoints="./output/RecModel/epoch_5" \
```

### 2.4 模型评估

可以通过以下命令进行模型评估。

```bash
python. -m paddle.distributed.launch \ 
    --gpus="0,1,2,3" \
    tools/eval.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model="./output/RecModel/best_model"\
```

<a name="model_inference"></a>
## 3. 使用inference模型进行模型推理
### 3.1 导出推理模型

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python tools/export_model.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model \
    -o Global.save_inference_dir=./inference \
```
其中，`--Global.pretrained_model`用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[1.3 模型恢复训练](#1.3)），`--Global.save_inference_dir`用于指定转换后模型的存储路径。
若`--save_inference_dir=./inference`，则会在`inference`文件夹下生成`inference.pdiparams`、`inference.pdmodel`和`inference.pdiparams.info`文件。

### 3.2 构建底库
通过检索方式来进行图像识别，需要构建底库。
首先, 将生成的模型拷贝到deploy目录下，并进入deploy目录：
```bash
mv ./inference ./deploy
cd deploy
```

其次，构建底库，命令如下：
```bash
python python/build_gallery.py \
       -c configs/build_flowers.yaml \
       -o Global.rec_inference_model_dir="./inference" \
       -o IndexProcess.index_path="../dataset/flowers102/index" \
       -o IndexProcess.image_root="../dataset/flowers102/" \
       -o IndexProcess.data_file="../dataset/flowers102/train_list.txt" 
```
其中
+ `Global.rec_inference_model_dir`：3.1生成的推理模型的路径
+ `IndexProcess.index_path`：gallery库index的路径
+ `IndexProcess.image_root`：gallery库图片的根目录
+ `IndexProcess.data_file`：gallery库图片的文件列表
执行完上述命令之后，会在`../dataset/flowers102`目录下面生成`index`目录，index目录下面包含3个文件`index.data`, `1index.graph`, `info.json`

### 3.3 推理预测

通过3.1生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），通过3.2构建好底库， 然后可以使用预测引擎进行推理：

```bash
python python/predict_rec.py \
    -c configs/inference_flowers.yaml \
    -o Global.infer_imgs="./images/image_00002.jpg" \
    -o Global.rec_inference_model_dir="./inference" \
    -o Global.use_gpu=True \
    -o Global.use_tensorrt=False
```
其中：
+ `Global.infer_imgs`：待预测的图片文件路径，如 `./images/image_00002.jpg`
+ `Global.rec_inference_model_dir`：预测模型文件路径，如 `./inference/`
+ `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`True`
+ `Global.use_gpu`：是否使用 GPU 预测，默认值：`True` 
执行完上述命令之后，会得到输入图片对应的特征信息； 本例子中特征维度为2048；显示如下：
```
(1, 2048)
[[0.00033124 0.00056205 0.00032261 ... 0.00030939 0.00050748 0.00030271]]
```
