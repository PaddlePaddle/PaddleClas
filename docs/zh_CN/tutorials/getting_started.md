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
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="" \
    -o use_gpu=True
```

其中，`-c`用于指定配置文件的路径，`-o`用于指定需要修改或者添加的参数，其中`-o pretrained_model=""`表示不使用预训练模型，`-o use_gpu=True`表示使用GPU进行训练。如果希望使用CPU进行训练，则需要将`use_gpu`设置为`False`。

更详细的训练配置，也可以直接修改模型对应的配置文件。具体配置参数参考[配置文档](config.md)。

运行上述命令，可以看到输出日志，示例如下：

* 如果在训练中使用了mixup或者cutmix的数据增广方式，那么日志中将不会打印top-1与top-k（默认为5）信息：
    ```
    ...
    epoch:0  , train step:20   , loss: 4.53660, lr: 0.003750, batch_cost: 1.23101 s, reader_cost: 0.74311 s, ips: 25.99489 images/sec, eta: 0:12:43
    ...
    END epoch:1   valid top1: 0.01569, top5: 0.06863, loss: 4.61747,  batch_cost: 0.26155 s, reader_cost: 0.16952 s, batch_cost_sum: 10.72348 s, ips: 76.46772 images/sec.
    ...
    ```

* 如果训练过程中没有使用mixup或者cutmix的数据增广，那么除了上述信息外，日志中也会打印出top-1与top-k(默认为5)的信息：

    ```
    ...
    epoch:0  , train step:30  , top1: 0.06250, top5: 0.09375, loss: 4.62766, lr: 0.003728, batch_cost: 0.64089 s, reader_cost: 0.18857 s, ips: 49.93080 images/sec, eta: 0:06:18
    ...
    END epoch:0   train top1: 0.01310, top5: 0.04738, loss: 4.65124,  batch_cost: 0.64089 s, reader_cost: 0.18857 s, batch_cost_sum: 13.45863 s, ips: 49.93080 images/sec.
    ...
    ```

训练期间也可以通过VisualDL实时观察loss变化，详见[VisualDL](../extension/VisualDL.md)。

### 1.2 模型微调

根据自己的数据集路径设置好配置文件后，可以通过加载预训练模型的方式进行微调，如下所示。

```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./pretrained/MobileNetV3_large_x1_0_pretrained" \
    -o use_gpu=True
```

其中`-o pretrained_model`用于设置加载预训练模型权重文件的地址，使用时需要换成自己的预训练模型权重文件的路径，也可以直接在配置文件中修改该路径。

我们也提供了大量基于`ImageNet-1k`数据集的预训练模型，模型列表及下载地址详见[模型库概览](../models/models_intro.md)。

<a name="1.3"></a>
### 1.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件，继续训练：

```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o checkpoints="./output/MobileNetV3_large_x1_0/5/ppcls" \
    -o last_epoch=5 \
    -o use_gpu=True
```

其中配置文件不需要做任何修改，只需要在继续训练时设置`checkpoints`参数即可，表示加载的断点权重文件路径，使用该参数会同时加载保存的断点权重和学习率、优化器等信息。

**注意**：
* 参数`-o last_epoch=5`表示将上一次训练轮次数记为`5`，即本次训练轮次数从`6`开始计算，该值默认为-1，表示本次训练轮次数从`0`开始计算。

* `-o checkpoints`参数无需包含断点权重文件的后缀名，上述训练命令会在训练过程中生成如下所示的断点权重文件，若想从断点`5`继续训练，则`checkpoints`参数只需设置为`"./output/MobileNetV3_large_x1_0_gpupaddle/5/ppcls"`，PaddleClas会自动补充后缀名。
    ```shell
    output/
    └── MobileNetV3_large_x1_0
        ├── 0
        │   ├── ppcls.pdopt
        │   └── ppcls.pdparams
        ├── 1
        │   ├── ppcls.pdopt
        │   └── ppcls.pdparams
        .
        .
        .
    ```

<a name="1.4"></a>
### 1.4 模型评估

可以通过以下命令进行模型评估。

```bash
python tools/eval.py \
    -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./output/MobileNetV3_large_x1_0/best_model/ppcls"\
    -o load_static_weights=False
```

上述命令将使用`./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml`作为配置文件，对上述训练得到的模型`./output/MobileNetV3_large_x1_0/best_model/ppcls`进行评估。你也可以通过更改配置文件中的参数来设置评估，也可以通过`-o`参数更新配置，如上所示。

可配置的部分评估参数说明如下：
* `ARCHITECTURE.name`：模型名称
* `pretrained_model`：待评估的模型文件路径
* `load_static_weights`：待评估模型是否为静态图模型

**注意：** 如果模型为动态图模型，则在加载待评估模型时，需要指定模型文件的路径，但无需包含文件后缀名，PaddleClas会自动补齐`.pdparams`的后缀，如[1.3 模型恢复训练](#1.3)。

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
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml
```

其中，`-c`用于指定配置文件的路径，可通过配置文件修改相关训练配置信息，也可以通过添加`-o`参数来更新配置：

```bash
python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
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
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
        -o pretrained_model="./pretrained/MobileNetV3_large_x1_0_pretrained"
```

其中`pretrained_model`用于设置加载预训练权重文件的路径，使用时需要换成自己的预训练模型权重文件路径，也可以直接在配置文件中修改该路径。

30分钟玩转PaddleClas[尝鲜版](./quick_start_new_user.md)与[进阶版](./quick_start_professional.md)中包含大量模型微调的示例，可以参考该章节在特定的数据集上进行模型微调。


<a name="model_resume"></a>
### 2.3 模型恢复训练

如果训练任务因为其他原因被终止，也可以加载断点权重文件继续训练。

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
        -o checkpoints="./output/MobileNetV3_large_x1_0/5/ppcls" \
        -o last_epoch=5 \
        -o use_gpu=True
```

其中配置文件不需要做任何修改，只需要在训练时设置`checkpoints`参数与`last_epoch`参数即可，该参数表示加载的断点权重文件路径，使用该参数会同时加载保存的模型参数权重和学习率、优化器等信息，详见[1.3 模型恢复训练](#1.3)。


### 2.4 模型评估

可以通过以下命令进行模型评估。

```bash
python tools/eval.py \
    -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./output/MobileNetV3_large_x1_0/best_model/ppcls"\
    -o load_static_weights=False
```

参数说明详见[1.4 模型评估](#1.4)。


<a name="model_infer"></a>
## 3. 使用预训练模型进行模型预测

模型训练完成之后，可以加载训练得到的预训练模型，进行模型预测。在模型库的 `tools/infer/infer.py` 中提供了完整的示例，只需执行下述命令即可完成模型预测：

```python
python tools/infer/infer.py \
    -i 待预测的图片文件路径 \
    --model MobileNetV3_large_x1_0 \
    --pretrained_model "./output/MobileNetV3_large_x1_0/best_model/ppcls" \
    --use_gpu True \
    --class_num 1000
```

参数说明：
+ `image_file`(简写 i)：待预测的图片文件路径或者批量预测时的图片文件夹，如 `./test.jpeg`
+ `model`：模型名称，如 `MobileNetV3_large_x1_0`
+ `pretrained_model`：模型权重文件路径，如 `./output/MobileNetV3_large_x1_0/best_model/ppcls`
+ `use_gpu` : 是否开启GPU训练，默认值：`True`
+ `class_num` : 类别数，默认为1000，需要根据自己的数据进行修改。
+ `resize_short`: 对输入图像进行等比例缩放，表示最短边的尺寸，默认值：`256`
+ `resize`: 对`resize_short`操作后的进行居中裁剪，表示裁剪的尺寸，默认值：`224`
+ `pre_label_image` : 是否对图像数据进行预标注，默认值：`False`
+ `pre_label_out_idr` : 预标注图像数据的输出文件夹，当`pre_label_image=True`时，会在该文件夹下面生成很多个子文件夹，每个文件夹名称为类别id，其中存储模型预测属于该类别的所有图像。

**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `resize=384`。


<a name="model_inference"></a>
## 4. 使用inference模型进行模型推理

通过导出inference模型，PaddlePaddle支持使用预测引擎进行预测推理。接下来介绍如何用预测引擎进行推理：
首先，对训练好的模型进行转换：

```bash
python tools/export_model.py \
    --model MobileNetV3_large_x1_0 \
    --pretrained_model ./output/MobileNetV3_large_x1_0/best_model/ppcls \
    --output_path ./inference \
    --class_dim 1000
```

其中，参数`--model`用于指定模型名称，`--pretrained_model`用于指定模型文件路径，该路径仍无需包含模型文件后缀名（如[1.3 模型恢复训练](#1.3)），`--output_path`用于指定转换后模型的存储路径，`class_dim`表示模型所包含的类别数，默认为1000。

**注意**：
1. `--output_path`表示输出的inference模型文件夹路径，若`--output_path=./inference`，则会在`inference`文件夹下生成`inference.pdiparams`、`inference.pdmodel`和`inference.pdiparams.info`文件。
2. 可以通过设置参数`--img_size`指定模型输入图像的`shape`，默认为`224`，表示图像尺寸为`224*224`，请根据实际情况修改。

上述命令将生成模型结构文件（`inference.pdmodel`）和模型权重文件（`inference.pdiparams`），然后可以使用预测引擎进行推理：

```bash
python tools/infer/predict.py \
    --image_file 图片路径 \
    --model_file "./inference/inference.pdmodel" \
    --params_file "./inference/inference.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```
其中：
+ `image_file`：待预测的图片文件路径，如 `./test.jpeg`
+ `model_file`：模型结构文件路径，如 `./inference/inference.pdmodel`
+ `params_file`：模型权重文件路径，如 `./inference/inference.pdiparams`
+ `use_tensorrt`：是否使用 TesorRT 预测引擎，默认值：`True`
+ `use_gpu`：是否使用 GPU 预测，默认值：`True`
+ `enable_mkldnn`：是否启用`MKL-DNN`加速，默认为`False`。注意`enable_mkldnn`与`use_gpu`同时为`True`时，将忽略`enable_mkldnn`，而使用GPU运行。
+ `resize_short`: 对输入图像进行等比例缩放，表示最短边的尺寸，默认值：`256`
+ `resize`: 对`resize_short`操作后的进行居中裁剪，表示裁剪的尺寸，默认值：`224`

**注意**: 如果使用`Transformer`系列模型，如`DeiT_***_384`, `ViT_***_384`等，请注意模型的输入数据尺寸，需要设置参数`resize_short=384`, `resize=384`。

* 如果你希望评测模型速度，建议使用该脚本(`tools/infer/predict.py`)，同时开启TensorRT加速预测。
