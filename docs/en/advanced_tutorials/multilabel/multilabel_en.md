# Multilabel classification quick start

Based on the [NUS-WIDE-SCENE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) dataset which is a subset of NUS-WIDE dataset, you can experience multilabel of PaddleClas, include training, evaluation and prediction. Please refer to [Installation](install.md) to install at first.

## Preparation

* Enter PaddleClas directory

```
cd path_to_PaddleClas
```

* Create and enter `dataset/NUS-WIDE-SCENE` directory, download and decompress NUS-WIDE-SCENE dataset

```shell
mkdir dataset/NUS-WIDE-SCENE
cd dataset/NUS-WIDE-SCENE
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/NUS-SCENE-dataset.tar
tar -xf NUS-SCENE-dataset.tar
```

* Return `PaddleClas` root home

```
cd ../../
```

## Environment

### Download pretrained model

You can use the following commands to download the pretrained model of ResNet50_vd.

```bash
mkdir pretrained
cd pretrained
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams
cd ../
```

## Training

```shell
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch \
    --gpus="0" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_multilabel.yaml
```

After training for 10 epochs, the best accuracy over the validation set should be around 0.72.

## Evaluation

```bash
python tools/eval.py \
    -c ./configs/quick_start/ResNet50_vd_multilabel.yaml \
    -o pretrained_model="./output/ResNet50_vd/best_model/ppcls" \
    -o load_static_weights=False
```

The metric of evaluation is based on mAP, which is commonly used in multilabel task to show model perfermance. The mAP over validation set should be around 0.57.

## Prediction

```bash
python tools/infer/infer.py \
    -i "./dataset/NUS-WIDE-SCENE/NUS-SCENE-dataset/images/0199_434752251.jpg" \
    --model ResNet50_vd \
    --pretrained_model "./output/ResNet50_vd/best_model/ppcls" \
    --use_gpu True \
    --load_static_weights False \
    --multilabel True \
    --class_num 33
```

You will get multiple output such as the following:
```    
    class id: 3, probability: 0.6025
    class id: 23, probability: 0.5491
    class id: 32, probability: 0.7006
```