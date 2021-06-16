# 开始使用
## 注意:  本文主要介绍基于检索方式的识别
---
请参考[安装指南](./install.md)配置运行环境，并根据[快速开始](./quick_start_new_user.md)文档准备flower102数据集，本章节下面所有的实验均以flower102数据集为例。

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
    -c configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
    -o pretrained_model="" \
    -o use_gpu=True
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o pretrained_model=""`表示不使用预训练模型，`-o use_gpu=True`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`use_gpu`设置为`False`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config.md)。

训练期间也可以通过VisualDL实时观察loss变化，详见[VisualDL](../extension/VisualDL.md)。

### 1.2 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python tools/train.py \
    -c configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
    -o Arch.Backbone.pretrained=True
    -o use_gpu=True
```

其中`-o pretrained_model`用于设置加载预训练模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径，也可以直接在配置文件中修改该路径。

我们也提供了大量基于`ImageNet-1k`数据集的预训练模型，模型列表及下载地址详见[模型库概览](../models/models_intro.md)。

<a name="1.3"></a>
### 1.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python tools/train.py \
    -c configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
    -o checkpoints="./output/RecModel/ppcls_epoch_5" \
    -o last_epoch=5 \
    -o use_gpu=True
```

其中配置文件不需要做任何修改，只需要在继续训练时设置`checkpoints`参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：
* 参数`-o last_epoch=5`表示将上一次训练轮次数记为`5`，即本次训练轮次数从`6`开始计算，该值默认为-1，表示本次训练轮次数从`0`开始计算。

* `-o checkpoints`参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点`5`继续训练，则`checkpoints`参数只需设置为`"./output/RecModel/ppcls_epoch_5"`，PaddleClas会自动补充后缀名。

<a name="1.4"></a>
### 1.4 模型评估

可以通过以下命令进行模型评估。

```bash
python tools/eval.py \
    -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
    -o pretrained_model="./output/RecModel/best_model"\
```

上述命令将使用`./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml`作为配置文件，对上述训练得到的模型`./output/RecModel/best_model`进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过`-o`参数更新配置，如上所示。

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
        -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml
```

其中，`-c`用于指定配置文件的路径，可通过配置文件修改相关训练配置信息，也可以通过添加`-o`参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
        -o pretrained_model="" \
        -o use_gpu=True
```
`-o`用于指定需要修改或者添加的参数，其中`-o pretrained_model=""`表示不使用预训练模型，`-o use_gpu=True`表示使用GPU进行训练。

输出日志信息的格式同上，详见[1.1 模型训练](#1.1)。

### 2.2 模型微调

根据自己的数据集配置好配置文件之后，可以加载预训练模型进行微调，如下所示。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
        -o Arch.Backbone.pretrained=True
```

30分钟玩转PaddleClas[尝鲜版](./quick_start_new_user.md)与[进阶版](./quick_start_professional.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。

### 2.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
        -o checkpoints="./output/RecModel/ppcls_epoch_5" \
        -o last_epoch=5 \
        -o use_gpu=True
```

其中配置文件不需要做任何修改，只需要在训练时设置`checkpoints`参数与`last_epoch`参数即可，该参数表示加载的断点权重文件路径，使用该参数会同时加载保存的模型参数权重和学习率、优化器等信息，详见[1.3 模型恢复训练](#1.3)。


### 2.4 模型评估

可以通过以下命令进行模型评估。

```bash
python tools/eval.py \
    -c ./configs/quick_start/ResNet50_flowers_retrieval_finetune.yaml \
    -o pretrained_model="./output/RecModel/best_model"\
```

参数说明详见[1.4 模型评估](#1.4)。

<a name="model_inference"></a>
## 3. 使用inference模型进行模型推理
### 3.1 导出推理模型

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python tools/export_model.py \
    --pretrained_model ./output/RecModel/best_model \
    --output_path ./inference \
```

其中，`--pretrained_model`用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[1.3 模型恢复训练](#1.3)），`--output_path`用于指定转换后模型的存储路径。

**注意**：
1. `--output_path`表示输出的inference模型文件夹路径，若`--output_path=./inference`，则会在`inference`文件夹下生成`inference.pdiparams`、`inference.pdmodel`和`inference.pdiparams.info`文件。
2. 可以通过设置参数`--img_size`指定模型输入图像的`shape`，默认为`224`，表示图像尺寸为`224*224`，请根据实际情况修改。

### 3.2 构建底库
通过检索方式来进行图像识别，需要构建底库。底库构建方式如下：
```bash
python python/build_gallery.py 
       -c configs/build_flowers.yaml \
       -o Global.rec_inference_model_dir "../inference" \
       -o IndexProcess.index_path "../dataset/index" \
       -o IndexProcess.image_root: "../dataset" \
       -o IndexProcess.data_file: "../dataset/train_list.txt" 
```
其中
+ `Global.rec_inference_model_dir`：3.1生成的推理模型的路径
+ `IndexProcess.index_path`：gallery库index的路径
+ `IndexProcess.image_root`：gallery库图片的根目录
+ `IndexProcess.data_file`：gallery库图片的文件列表

### 3.3 推理预测

通过3.1生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），通过3.2构建好底库， 然后可以使用预测引擎进行推理：

```bash
python python/predict_rec.py \
    -c configs/inference_flowers.yaml \
    -o Global.infer_imgs 图片路径 \
    -o Global.rec_inference_model_dir "./inference"
    -o Global.use_gpu=True \
    -o Global.use_tensorrt=False
```
其中：
+ `Global.infer_imgs`：待预测的图片文件路径，如 `./test.jpeg`
+ `Global.rec_inference_model_dir`：模型结构文件路径，如 `./inference/`
+ `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`True`
+ `Global.use_gpu`：是否使用 GPU 预测，默认值：`True` 
