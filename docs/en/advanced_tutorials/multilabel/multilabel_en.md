# Multilabel classification quick start

Based on the [NUS-WIDE-SCENE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) dataset which is a subset of NUS-WIDE dataset, you can experience multilabel of PaddleClas, include training, evaluation and prediction. Please refer to [Installation](../../installation/) to install at first.

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

## Training

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml
```

After training for 10 epochs, the best accuracy over the validation set should be around 0.95.

## Evaluation

```bash
python tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

## Prediction

```bash
python3 tools/infer.py
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

You will get multiple output such as the following:
```
[{'class_ids': [6, 13, 17, 23, 26, 30], 'scores': [0.95683, 0.5567, 0.55211, 0.99088, 0.5943, 0.78767], 'file_name': './deploy/images/0517_2715693311.jpg', 'label_names': []}]  
```

## Prediction based on prediction engine

### Export model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

The default path of the inference model is under the current path `./inference`

### Prediction based on prediction engine

Enter the deploy directory:

```bash
cd ./deploy
```

Prediction based on prediction engine:

```
python3 python/predict_cls.py \
     -c configs/inference_multilabel_cls.yaml
```

You will get multiple output such as the following:

```
0517_2715693311.jpg:    class id(s): [6, 13, 17, 23, 26, 30], score(s): [0.96, 0.56, 0.55, 0.99, 0.59, 0.79], label_name(s): []
```
