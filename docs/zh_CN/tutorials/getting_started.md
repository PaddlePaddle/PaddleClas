# 开始使用
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
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=False \
    -o Global.device=gpu
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o Arch.pretrained=False`表示不使用预训练模型，`-o Global.device=gpu`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`Global.device`设置为`cpu`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config_description.md)。

运行上述命令，可以看到输出日志，示例如下：

* 如果在训练中使用了mixup或者cutmix的数据增广方式，那么日志中将不会打印top-1与top-k（默认为5）信息：
    ```
    ...
    [Train][Epoch 3/20][Avg]CELoss: 6.46287, loss: 6.46287
    ...
    [Eval][Epoch 3][Avg]CELoss: 5.94309, loss: 5.94309, top1: 0.01961, top5: 0.07941
    ...
    ```

* 如果训练过程中没有使用mixup或者cutmix的数据增广，那么除了上述信息外，日志中也会打印出top-1与top-k(默认为5)的信息：

    ```
    ...
    [Train][Epoch 3/20][Avg]CELoss: 6.12570, loss: 6.12570, top1: 0.01765, top5: 0.06961
    ...
    [Eval][Epoch 3][Avg]CELoss: 5.40727, loss: 5.40727, top1: 0.07549, top5: 0.20980
    ...
    ```

训练期间也可以通过VisualDL实时观察loss变化，详见[VisualDL](../extension/VisualDL.md)。

### 1.2 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Arch.pretrained=True \
    -o Global.device=gpu
```

其中`Arch.pretrained`设置为`True`表示加载ImageNet的预训练模型，此外，`Arch.pretrained`也可以指定具体的模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径。

我们也提供了大量基于`ImageNet-1k`数据集的预训练模型，模型列表及下载地址详见[模型库概览](../models/models_intro.md)。

<a name="1.3"></a>
### 1.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python3 tools/train.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
    -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在继续训练时设置`Global.checkpoints`参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：

* `-o Global.checkpoints`参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点`5`继续训练，则`Global.checkpoints`参数只需设置为`"../output/MobileNetV3_large_x1_0/epoch_5"`，PaddleClas会自动补充后缀名。output目录下的文件结构如下所示：

    ```shell
    output
    ├── MobileNetV3_large_x1_0
    │   ├── best_model.pdopt
    │   ├── best_model.pdparams
    │   ├── best_model.pdstates
    │   ├── epoch_1.pdopt
    │   ├── epoch_1.pdparams
    │   ├── epoch_1.pdstates
        .
        .
        .
    ```

<a name="1.4"></a>
### 1.4 模型评估

可以通过以下命令进行模型评估。

```bash
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

上述命令将使用`./configs/quick_start/MobileNetV3_large_x1_0.yaml`作为配置文件，对上述训练得到的模型`./output/MobileNetV3_large_x1_0/best_model`进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过`-o`参数更新配置，如上所示。

可配置的部分评估参数说明如下：
* `Arch.name`：模型名称
* `Global.pretrained_model`：待评估的模型预训练模型文件路径

**注意：** 在加载待评估模型时，需要指定模型文件的路径，但无需包含文件后缀名，PaddleClas会自动补齐`.pdparams`的后缀，如[1.3 模型恢复训练](#1.3)。

<a name="2"></a>
## 2. 基于Linux+GPU的模型训练与评估

如果机器环境为Linux+GPU，那么推荐使用`paddle.distributed.launch`启动模型训练脚本（`tools/train.py`）、评估脚本（`tools/eval.py`），可以更方便地启动多卡训练与评估。

### 2.1 模型训练

参考如下方式启动模型训练，`paddle.distributed.launch`通过设置`gpus`指定GPU运行卡号：

```bash
# PaddleClas通过launch方式启动多卡多进程训练

export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml
```

输出日志信息的格式同上，详见[1.1 模型训练](#1.1)。

### 2.2 模型微调

根据自己的数据集配置好配置文件之后，可以加载预训练模型进行微调，如下所示。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Arch.pretrained=True
```

其中`Arch.pretrained`为`True`或`False`，当然也可以设置加载预训练权重文件的路径，使用时需要换成自己的预训练模型权重文件路径，也可以直接在配置文件中修改该路径。

30分钟玩转PaddleClas[尝鲜版](./quick_start_new_user.md)与[进阶版](./quick_start_professional.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。


<a name="model_resume"></a>
### 2.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.checkpoints="./output/MobileNetV3_large_x1_0/epoch_5" \
        -o Global.device=gpu
```

其中配置文件不需要做任何修改，只需要在训练时设置`Global.checkpoints`参数即可，该参数表示加载的断点权重文件路径，使用该参数会同时加载保存的模型参数权重和学习率、优化器等信息，详见[1.3 模型恢复训练](#1.3)。


### 2.4 模型评估

可以通过以下命令进行模型评估。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    tools/eval.py \
        -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
        -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

参数说明详见[1.4 模型评估](#1.4)。


<a name="model_infer"></a>
## 3. 使用预训练模型进行模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python

python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg \
    -o Global.pretrained_model=./output/MobileNetV3_large_x1_0/best_model
```

参数说明：
+ `Infer.infer_imgs`：待预测的图片文件路径或者批量预测时的图片文件夹。
+ `Global.pretrained_model`：模型权重文件路径，如 `./output/MobileNetV3_large_x1_0/best_model`


<a name="model_inference"></a>
## 4. 使用inference模型进行模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/MobileNetV3_large_x1_0.yaml \
    -o Global.pretrained_model=output/MobileNetV3_large_x1_0/best_model
```


其中，`Global.pretrained_model`用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[1.3 模型恢复训练](#1.3)）。

上述命令将生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），然后可以使用预测引擎进行推理：

进入deploy目录下：

```bash
cd deploy
```

执行命令进行预测，由于默认class_id_map_file是ImageNet数据集的映射文件，所以此处需要置None。

```bash
python3 python/predict_cls.py \
    -c configs/inference_cls.yaml \
    -o Global.infer_imgs=../dataset/flowers102/jpg/image_00001.jpg \
    -o Global.inference_model_dir=../inference/ \
    -o PostProcess.Topk.class_id_map_file=None


其中：
+ `Global.infer_imgs`：待预测的图片文件路径。
+ `Global.inference_model_dir`：inference模型结构文件路径，如 `../inference/inference.pdmodel`
+ `Global.use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`False`
+ `Global.use_gpu`：是否使用 GPU 预测，默认值：`True`
+ `Global.enable_mkldnn`：是否启用`MKL-DNN`加速，默认为`False`。注意`enable_mkldnn`与`use_gpu`同时为`True`时，将忽略`enable_mkldnn`，而使用GPU运行。
+ `Global.use_fp16`：是否启用`FP16`，默认为`False`。


**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `resize=384`。

* 如果你希望提升评测模型速度，使用gpu评测时，建议开启TensorRT加速预测，使用cpu评测时，建议开启MKL-DNN加速预测。
