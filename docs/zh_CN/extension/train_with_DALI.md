# 使用DALI加速训练

## 前言
[NVIDIA数据加载库](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)（The NVIDIA Data Loading Library，DALI）是用于数据加载和预处理的开源库，用于加速深度学习训练、推理过程，它可以直接构建飞桨Paddle的DataLoader数据读取器。

由于深度学习程序在训练阶段依赖大量数据，这些数据需要经过加载、预处理等操作后，才能送入训练程序，而这些操作通常在CPU完成，因此限制了训练速度进一步提高，特别是在batch_size较大时，数据读取可能成为训练速度的瓶颈。DALI可以基于GPU的高并行特性实现数据加载及预处理操作，可以进一步提高训练速度。

## 安装DALI
目前DALI仅支持Linux x64平台，且CUDA版本大于等于10.0。

* 对于CUDA 10:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

* 对于CUDA 11.0:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

关于更多DALI安装的信息，可以参考[DALI官方](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)。

## 使用DALI
PaddleClas支持在静态图训练方式中使用DALI加速，由于DALI仅支持GPU训练，因此需要设置GPU，且DALI需要占用GPU显存，需要为DALI预留显存。使用DALI训练只需在训练配置文件中设置字段`use_dali=True`，或通过以下命令启动训练即可：

```shell
# 设置用于训练的GPU卡号
export CUDA_VISIBLE_DEVICES="0"

# 设置用于神经网络训练的显存大小，可根据具体情况设置，一般可设置为0.8或0.7
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python tools/static/train.py -c configs/ResNet/ResNet50.yaml -o use_dali=True
```

也可以使用多卡训练：

```shell
# 设置用于训练的GPU卡号
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 设置用于神经网络训练的显存，可根据具体情况设置，一般可设置为0.8或0.7
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/static/train.py \
        -c ./configs/ResNet/ResNet50.yaml \
        -o use_dali=True
```

## 使用FP16训练

在上述基础上，使用FP16半精度训练，可以进一步提高速度，只需在启动训练命令中添加字段`AMP.use_pure_fp16=True`：

```shell
python tools/static/train.py -c configs/ResNet/ResNet50.yaml -o use_dali=True -o AMP.use_pure_fp16=True
```

使用FP16半精度训练将导致训练精度下降或收敛变慢的问题。
