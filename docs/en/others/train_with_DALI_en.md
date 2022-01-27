# Train with DALI

---

## Catalogue

* [1. Preface](#1)
* [2. Installing DALI](#2)
* [3. Using DALI](#3)
* [4. Train with FP16](#4)

<a name='1'></a>

## 1. Preface

[The NVIDIA Data Loading Library](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It can build Dataloader of PaddlePaddle.

Since the Deep Learning relies on a large amount of data in the training stage, these data need to be loaded and preprocessed. These operations are usually executed on the CPU, which limits the further improvement of the training speed, especially when the batch_size is large, which become the bottleneck of training speed. DALI can use GPU to accelerate these operations, thereby further improve the training speed.

<a name='2'></a>

## 2. Installing DALI

DALI only support Linux x64 and version of CUDA is 10.2 or later.

* For CUDA 10:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100

* For CUDA 11.0:

    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

For more information about installing DALI, please refer to [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).

<a name='3'></a>

## 3. Using DALI

Paddleclas supports training with DALI. Since DALI only supports GPU training, `CUDA_VISIBLE_DEVICES` needs to be set, and DALI needs to occupy GPU memory, so it needs to reserve GPU memory for Dali. To train with DALI, just set the fields in the training config `use_dali = True`, or start the training by the following command:

```shell
# set the GPUs that can be seen
export CUDA_VISIBLE_DEVICES="0"

python ppcls/train.py -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml -o Global.use_dali=True
```

And you can train with muti-GPUs:

```shell
# set the GPUs that can be seen
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# set the GPU memory used for neural network training, generally 0.8 or 0.7, and the remaining GPU memory is reserved for DALI
export FLAGS_fraction_of_gpu_memory_to_use=0.80

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/train.py \
        -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
        -o Global.use_dali=True
```

<a name='4'></a>

## 4. Train with FP16

On the basis of the above, using FP16 half-precision can further improve the training speed, you can refer to the following command.

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.8

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_fp16_dygraph.yaml
```
