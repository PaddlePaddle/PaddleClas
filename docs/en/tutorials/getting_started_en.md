# Getting Started
---
Please refer to [Installation](install.md) to setup environment at first, and prepare ImageNet1K data by following the instruction mentioned in the [data](data.md)

## 1. Training and Evaluation on Windows or CPU

If training and evaluation are performed on Windows system or CPU, it is recommended to use the `tools/train_multi_platform.py` and `tools/eval_multi_platform.py` scripts.


## 1.1 Model training

After preparing the configuration file, The training process can be started in the following way.

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o model_save_dir=./output/ \
    -o use_gpu=True
```

Among them, `-c` is used to specify the path of the configuration file, `-o` is used to specify the parameters needed to be modified or added, `-o model_save_dir=./output/` means to modify the `model_save_dir` in the configuration file to ` ./output/`. `-o use_gpu=True` means to use GPU for training. If you want to use the CPU for training, you need to set `use_gpu` to `False`.


Of course, you can also directly modify the configuration file to update the configuration. For specific configuration parameters, please refer to [Configuration Document](config.md).

* The output log examples are as follows:
    * If mixup or cutmix is used in training, only loss, lr (learning rate) and training time of the minibatch will be printed in the log.

    ```
    train step:890  loss:  6.8473 lr: 0.100000 elapse: 0.157s
    ```

    * If mixup or cutmix is not used during training, in addition to loss, lr (learning rate) and the training time of the minibatch, top-1 and top-k( The default is 5) will also be printed in the log.

    ```
    epoch:0    train    step:13    loss:7.9561    top1:0.0156    top5:0.1094    lr:0.100000    elapse:0.193s
    ```

During training, you can view loss changes in real time through `VisualDL`. The command is as follows.

```bash
visualdl --logdir ./scalar --host <host_IP> --port <port_num>
```

### 1.2 Model finetuning

* After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o pretrained_model="./pretrained/ResNet50_pretrained"
```

Among them, `pretrained_model` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

### 1.3 Resume Training

* If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
python tools/train_multi_platform.py \
    -c configs/ResNet/ResNet50.yaml \
    -o checkpoints="./output/ResNet/0/ppcls"
```

The configuration file does not need to be modified. You only need to add the `checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, earning rate, optimizer and other information will be loaded using this parameter.


### 1.4 Model evaluation

* The model evaluation process can be started as follows.

```bash
python tools/eval_multi_platform.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models
```

You can modify the `ARCHITECTURE.name` field and `pretrained_model` field in `configs/eval.yaml` to configure the evaluation model, and you also can update the configuration through the -o parameter.


**Note:** When loading the pretrained model, you need to specify the prefix of the pretrained model. For example, the pretrained model path is `output/ResNet50_vd/19`, and the pretrained model filename is `output/ResNet50_vd/19/ppcls.pdparams`, the parameter `pretrained_model` needs to be specified as `output/ResNet50_vd/19/ppcls`, PaddleClas will automatically fill in the `.pdparams` suffix.

### 2. Training and evaluation on Linux+GPU

If you want to run PaddleClas on Linux with GPU, it is highly recommended to use the model training and evaluation scripts provided by PaddleClas: `tools/train.py` and `tools/eval.py`.

### 2.1 Model training

After preparing the configuration file, The training process can be started in the following way.

```bash
# PaddleClas starts multi-card and multi-process training through launch
# Specify the GPU running card number by setting FLAGS_selected_gpus
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml
```

The configuration can be updated by adding the `-o` parameter.

```bash
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c ./configs/ResNet/ResNet50_vd.yaml \
        -o use_mix=1 \
        --vdl_dir=./scalar/
```

The format of output log information is the same as above.



### 2.2 Model finetuning

* After configuring the configuration file, you can finetune it by loading the pretrained weights, The command is as shown below.

```
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c configs/ResNet/ResNet50.yaml \
        -o pretrained_model="./pretrained/ResNet50_pretrained"
```

Among them, `pretrained_model` is used to set the address to load the pretrained weights. When using it, you need to replace it with your own pretrained weights' path, or you can modify the path directly in the configuration file.

* There contains a lot of examples of model finetuning in [The quick start tutorial](./quick_start_en.md). You can refer to this tutorial to finetune the model on a specific dataset.

### 2.3 Resume Training

* If the training process is terminated for some reasons, you can also load the checkpoints to continue training.

```
python -m paddle.distributed.launch \
    --selected_gpus="0,1,2,3" \
    tools/train.py \
        -c configs/ResNet/ResNet50.yaml \
        -o checkpoints="./output/ResNet/0/ppcls"
```

The configuration file does not need to be modified. You only need to add the `checkpoints` parameter during training, which represents the path of the checkpoints. The parameter weights, learning rate, optimizer and other information will be loaded using this parameter.

### 2.4 Model evaluation

* The model evaluation process can be started as follows.

```bash
python tools/eval_multi_platform.py \
    -c ./configs/eval.yaml \
    -o ARCHITECTURE.name="ResNet50_vd" \
    -o pretrained_model=path_to_pretrained_models
```

You can modify the `ARCHITECTURE.name` field and `pretrained_model` field in `configs/eval.yaml` to configure the evaluation model, and you also can update the configuration through the -o parameter.


## 3. Model inference

PaddlePaddle provides three ways to perform model inference. Next, how to use the inference engine to perforance model inference will be introduced.

Firstly, you should export inference model using `tools/export_model.py`.

```bash
python tools/export_model.py \
    --model=model_name \
    --pretrained_model=pretrained_model_dir \
    --output_path=save_inference_dir

```

Secondly, Inference engine can be started using the following commands.

```bash
python tools/infer/predict.py \
    -m model_path \
    -p params_path \
    -i image path \
    --use_gpu=1 \
    --use_tensorrt=True
```
please refer to [inference](../extension/paddle_inference_en.md) for more details.
