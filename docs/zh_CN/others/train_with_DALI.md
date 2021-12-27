# 使用 DALI 加速训练
----
## 目录
* [1. 前言](#1)
* [2. 安装 DALI](#2)
* [3. 使用 DALI](#3)
* [4. 使用 FP16 训练](#4)

 <a name='1'></a>

## 1. 前言
[NVIDIA 数据加载库](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)（The NVIDIA Data Loading Library，DALI）是用于数据加载和预处理的开源库，用于加速深度学习训练、推理过程，它可以直接构建飞桨 Paddle 的 DataLoader 数据读取器。

由于深度学习程序在训练阶段依赖大量数据，这些数据需要经过加载、预处理等操作后，才能送入训练程序，而这些操作通常在 CPU 完成，因此限制了训练速度进一步提高，特别是在 batch_size 较大时，数据读取可能成为训练速度的瓶颈。 DALI 可以基于 GPU 的高并行特性实现数据加载及预处理操作，可以进一步提高训练速度。

 <a name='2'></a>

## 2.安装 DALI
目前 DALI 仅支持 Linux x64 平台，且 CUDA 版本大于等于 10.2。

* 对于 CUDA 10:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

* 对于 CUDA 11.0:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

关于更多 DALI 安装的信息，可以参考[DALI 官方](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)。

 <a name='3'></a>

## 3. 使用 DALI
PaddleClas 支持使用 DALI 对图像预处理进行加速，由于 DALI 仅支持 GPU 训练，因此需要设置 GPU，且 DALI 需要占用 GPU 显存，需要为 DALI 预留显存。使用 DALI 训练只需在训练配置文件中设置字段 `use_dali=True`，或通过以下命令启动训练即可：

```shell
# 设置用于训练的 GPU 卡号
export CUDA_VISIBLE_DEVICES="0"

python ppcls/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml -o Global.use_dali=True
```

也可以使用多卡训练：

```shell
# 设置用于训练的 GPU 卡号
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 设置用于神经网络训练的显存大小，可根据具体情况设置，一般可设置为 0.8 或 0.7，剩余显存则预留 DALI 使用
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/train.py \
        -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
        -o Global.use_dali=True
```

<a name='4'></a>

## 4. 使用 FP16 训练
在上述基础上，使用 FP16 半精度训练，可以进一步提高速度，可以参考下面的配置与运行命令。

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.8

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_fp16_dygraph.yaml
```
