# Getting Started
---
Please refer to [Installation](install_en.md) to setup environment at first, and prepare flower102 dataset by following the instruction mentioned in the [Quick Start](quick_start_en.md).

## 1. Training and Evaluation on CPU or Single GPU

If training and evaluation are performed on CPU or single GPU, it is recommended to use the `tools/train.py` and `tools/eval.py`.
For training and evaluation in multi-GPU environment on Linux, please refer to [2. Training and evaluation on Linux+GPU](#2-training-and-evaluation-on-linuxgpu).

<a name="1.1"></a>
## 1.1 Model training

After preparing the configuration file, The training process can be started in the following way.

```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="" \
    -o use_gpu=False
```

Among them, `-c` is used to specify the path of the configuration file, `-o` is used to specify the parameters needed to be modified or added, `-o pretrained_model=""` means to not using pre-trained models.
`-o use_gpu=True` means to use GPU for training. If you want to use the CPU for training, you need to set `use_gpu` to `False`.


Of course, you can also directly modify the configuration file to update the configuration. For specific configuration parameters, please refer to [Configuration Document](config_en.md).

* The output log examples are as follows:
    * If mixup or cutmix is used in training, only loss, lr (learning rate) and training time of the minibatch will be printed in the log.

    ```
    train step:890  loss:  6.8473 lr: 0.100000 elapse: 0.157s
    ```

    * If mixup or cutmix is not used during training, in addition to loss, lr (learning rate) and the training time of the minibatch, top-1 and top-k( The default is 5) will also be printed in the log.

    ```
    epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193s
    ```

During training, you can view loss changes in real time through `VisualDL`,  see [VisualDL](../extension/VisualDL.md) for details.

### 1.2 Model finetuning

After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./pretrained/MobileNetV3_large_x1_0_pretrained" \
    -o use_gpu=True
```

Among them, `-o pretrained_model` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

We also provide a lot of pre-trained models trained on the ImageNet-1k dataset. For the model list and download address, please refer to the [model library overview](../models/models_intro_en.md).

### 1.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
python tools/train.py \
    -c configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o checkpoints="./output/MobileNetV3_large_x1_0/5/ppcls" \
    -o last_epoch=5 \
    -o use_gpu=True
```

The configuration file does not need to be modified. You only need to add the `checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter.

**Note**:
* The parameter `-o last_epoch=5` means to record the number of the last training epoch as `5`, that is, the number of this training epoch starts from `6`, , and the parameter defaults to `-1`, which means the number of this training epoch starts from `0`.

* The `-o checkpoints` parameter does not need to include the suffix of the checkpoints. The above training command will generate the checkpoints as shown below during the training process. If you want to continue training from the epoch `5`, Just set the `checkpoints` to `./output/MobileNetV3_large_x1_0_gpupaddle/5/ppcls`, PaddleClas will automatically fill in the `pdopt` and `pdparams` suffixes.

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


### 1.4 Model evaluation

The model evaluation process can be started as follows.

```bash
python tools/eval.py \
    -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./output/MobileNetV3_large_x1_0/best_model/ppcls"\
    -o load_static_weights=False
```

The above command will use `./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml` as the configuration file to evaluate the model `./output/MobileNetV3_large_x1_0/best_model/ppcls`. You can also set the evaluation by changing the parameters in the configuration file, or you can update the configuration with the `-o` parameter, as shown above.

Some of the configurable evaluation parameters are described as follows:
* `ARCHITECTURE.name`: Model name
* `pretrained_model`: The path of the model file to be evaluated
* `load_static_weights`: Whether the model to be evaluated is a static graph model


**Note:** If the model is a dygraph type, you only need to specify the prefix of the model file when loading the model, instead of specifying the suffix, such as [1.3 Resume Training](#13-resume-training).

<a name="2"></a>
### 2. Training and evaluation on Linux+GPU

If you want to run PaddleClas on Linux with GPU, it is highly recommended to use `paddle.distributed.launch` to start the model training script(`tools/train.py`) and evaluation script(`tools/eval.py`), which can start on multi-GPU environment more conveniently.

### 2.1 Model training

After preparing the configuration file, The training process can be started in the following way. `paddle.distributed.launch` specifies the GPU running card number by setting `selected_gpus`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml
```

The configuration can be updated by adding the `-o` parameter.

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
        -o pretrained_model="" \
        -o use_gpu=True
```

The format of output log information is the same as above, see [1.1 Model training](#11-model-training) for details.

### 2.2 Model finetuning

After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
        -o pretrained_model="./pretrained/MobileNetV3_large_x1_0_pretrained"
```

Among them, `pretrained_model` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

There contains a lot of examples of model finetuning in [Quick Start](./quick_start_en.md). You can refer to this tutorial to finetune the model on a specific dataset.

<a name="model_resume"></a>
### 2.3 Resume Training

If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
        -o checkpoints="./output/MobileNetV3_large_x1_0/5/ppcls" \
        -o last_epoch=5 \
        -o use_gpu=True
```

The configuration file does not need to be modified. You only need to add the `checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter. About `last_epoch` parameter, please refer [1.3 Resume training](#13-resume-training) for details.

### 2.4 Model evaluation

The model evaluation process can be started as follows.

```bash
python tools/eval.py \
    -c ./configs/quick_start/MobileNetV3_large_x1_0_finetune.yaml \
    -o pretrained_model="./output/MobileNetV3_large_x1_0/best_model/ppcls"\
    -o load_static_weights=False
```

About parameter description, see [1.4 Model evaluation](#14-model-evaluation) for details.

<a name="model_infer"></a>
## 3. Use the pre-trained model to predict
After the training is completed, you can predict by using the pre-trained model obtained by the training, as follows:

```python
python tools/infer/infer.py \
    -i image path \
    --model MobileNetV3_large_x1_0 \
    --pretrained_model "./output/MobileNetV3_large_x1_0/best_model/ppcls" \
    --use_gpu True \
    --load_static_weights False
```

Among them:
+ `image_file`(i): The path of the image file to be predicted, such as `./test.jpeg`;
+ `model`: Model name, such as `MobileNetV3_large_x1_0`;
+ `pretrained_model`: Weight file path, such as `./pretrained/MobileNetV3_large_x1_0_pretrained/`;
+ `use_gpu`: Whether to use the GPU, default by `True`;
+ `load_static_weights`: Whether to load the pre-trained model obtained from static image training, default by `False`;
+ `pre_label_image`: Whether to pre-label the image data, default value: `False`;
+ `pre_label_out_idr`: The output path of pre-labeled image data. When `pre_label_image=True`, a lot of subfolders will be generated under the path, each subfolder represent a category, which stores all the images predicted by the model to belong to the category.

About more detailed infomation, you can refer to [infer.py](../../../tools/infer/infer.py).

<a name="model_inference"></a>
## 4. Use the inference model to predict

PaddlePaddle supports inference using prediction engines, which will be introduced next.

Firstly, you should export inference model using `tools/export_model.py`.

```bash
python tools/export_model.py \
    --model MobileNetV3_large_x1_0 \
    --pretrained_model ./output/MobileNetV3_large_x1_0/best_model/ppcls \
    --output_path ./inference \
    --class_dim 1000
```

Among them, the `--model` parameter is used to specify the model name, `--pretrained_model` parameter is used to specify the model file path, the path does not need to include the model file suffix name, and `--output_path` is used to specify the storage path of the converted model, class_dim means number of class for the model, default as 1000.

**Note**:
1. If `--output_path=./inference`, then three files will be generated in the folder `inference`, they are `inference.pdiparams`, `inference.pdmodel` and `inference.pdiparams.info`.
2. In the file `export_model.py:line53`, the `shape` parameter is the shape of the model input image, the default is `224*224`. Please modify it according to the actual situation, as shown below:

```python
50 # Please modify the 'shape' according to actual needs
51 @to_static(input_spec=[
52     paddle.static.InputSpec(
53         shape=[None, 3, 224, 224], dtype='float32')
54 ])
```

The above command will generate the model structure file (`inference.pdmodel`) and the model weight file (`inference.pdiparams`), and then the inference engine can be used for inference:

```bash
python tools/infer/predict.py \
    --image_file image path \
    --model_file "./inference/inference.pdmodel" \
    --params_file "./inference/inference.pdiparams" \
    --use_gpu=True \
    --use_tensorrt=False
```
Among them:
+ `image_file`: The path of the image file to be predicted, such as `./test.jpeg`;
+ `model_file`: Model file path, such as `./MobileNetV3_large_x1_0/inference.pdmodel`;
+ `params_file`: Weight file path, such as `./MobileNetV3_large_x1_0/inference.pdiparams`;
+ `use_tensorrt`: Whether to use the TesorRT, default by `True`;
+ `use_gpu`: Whether to use the GPU, default by `True`
+ `enable_mkldnn`: Wheter to use `MKL-DNN`, default by `False`. When both `use_gpu` and `enable_mkldnn` are set to `True`, GPU is used to run and `enable_mkldnn` will be ignored.


If you want to evaluate the speed of the model, it is recommended to use [predict.py](../../../tools/infer/predict.py), and enable TensorRT to accelerate.
