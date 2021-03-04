
## Introduction

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model.
Quantization is a technique that reduces this redundancy by reducing the full precision data to a fixed number,
so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided [APIs of Quantization](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/) to compress the PaddleClas models.

It is recommended that you could understand following pages before reading this example：
- [The training strategy of PaddleClas models](../../../docs/en/tutorials/quick_start_en.md)
- [PaddleSlim Document](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)

## Quick Start
Quantization is mostly suitable for the deployment of lightweight models on mobile terminals.
After training, if you want to further compress the model size and accelerate the prediction, you can use quantization methods to compress the model according to the following steps.

1. Install PaddleSlim
2. Prepare trained model
3. Quantization-Aware Training
4. Export inference model
5. Deploy quantization inference model


### 1. Install PaddleSlim

* Install by pip.

```bash
pip3.7 install paddleslim==2.0.0
```

* Install from source code to get the lastest features.

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```


### 2. Download Pretrain Model
PaddleClas provides a series of trained [models](../../../docs/en/models/models_intro_en.md).
If the model to be quantified is not in the list, you need to follow the [Regular Training](../../../docs/en/tutorials/getting_started_en.md) method to get the trained model.


### 3. Quant-Aware Training
Quantization training includes offline quantization training and online quantization training.
Online quantization training is more effective. It is necessary to load the pre-trained model.
After the quantization strategy is defined, the model can be quantified.

The code for quantization training is located in `deploy/slim/quant/quant.py`. The training command is as follow:

* CPU/Single GPU training

```bash
python3.7 deploy/slim/quant/quant.py \
    -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
    -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
```

* Distributed training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    deploy/slim/quant/quant.py \
        -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
        -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
```

* The command of quantizing `MobileNetV3_large_x1_0` model is as follow:

```bash
# download pre-trained model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams

# run training
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    deploy/slim/quant/quant.py \
        -c configs/MobileNetV3/MobileNetV3_large_x1_0.yaml \
        -o pretrained_model="./MobileNetV3_large_x1_0_pretrained"
        -o LEARNING_RATE.params.lr=0.13 \
        -o epochs=100
```


### 4. Export inference model

After getting the model quantization aware trained, we can export it as inference model for predictive deployment:

```bash
python3.7 deploy/slim/quant/export_model.py \
    -m MobileNetV3_large_x1_0 \
    -p output/MobileNetV3_large_x1_0/best_model/ppcls \
    -o ./MobileNetV3_large_x1_0_infer/ \
    --img_size=224 \
    --class_dim=1000
```

### 5. Deploy
The type of quantized model's parameters derived from the above steps is still FP32, but the numerical range of the parameters is int8.
The derived model can be converted through the `opt tool` of PaddleLite.

For quantitative model deployment, please refer to [Mobile terminal model deployment](../../lite/readme_en.md)

## Notices:

* In quantitative training, it is suggested to load the pre-trained model obtained from conventional training to accelerate the convergence of quantitative training.
* In quantitative training, it is suggested that the initial learning rate should be changed to `1 / 20 ~ 1 / 10` of the conventional training, and the training epoch number should be changed to `1 / 5 ~ 1 / 2` of the conventional training. In terms of learning rate strategy, other configuration information is not recommended to be changed.
