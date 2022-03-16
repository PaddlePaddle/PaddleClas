# Quick Start of Multi-label Classification

Experience the training, evaluation, and prediction of multi-label classification based on the [NUS-WIDE-SCENE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) dataset, which is a subset of the NUS-WIDE dataset. Please first install PaddlePaddle and PaddleClas, see [Paddle Installation](../installation/install_paddle_en.md) and [PaddleClas installation](../installation/install_paddleclas_en.md) for more details.

## Catalogue

- [1. Data and Model Preparation](#1)
- [2. Model Training](#2)
- [3. Model Evaluation](#3)
- [4. Model Prediction](#4)
- [5. Predictive engine-based Prediction](#5)
  - [5.1 Export inference model](#5.1)
  - [5.2 Predictive engine-based Prediction](#5.2)


<a name="1"></a>
## 1. Data and Model Preparation

- Go to `PaddleClas`.

```
cd path_to_PaddleClas
```

- Create and go to `dataset/NUS-WIDE-SCENE`, download and unzip the NUS-WIDE-SCENE dataset.

```
mkdir dataset/NUS-WIDE-SCENE
cd dataset/NUS-WIDE-SCENE
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/NUS-SCENE-dataset.tar
tar -xf NUS-SCENE-dataset.tar
```

- Return to `PaddleClas` root directory

```
cd ../../
```


<a name="2"></a>
## 2. Model Training

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml
```

After training 10 epochs, the best correctness of the validation set should be around 0.95.


<a name="3"></a>
## 3. Model Evaluation

```
python3 tools/eval.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```


<a name="4"></a>
## 4. Model Prediction

```
python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

Obtain an output silimar to the following:

```
[{'class_ids': [6, 13, 17, 23, 26, 30], 'scores': [0.95683, 0.5567, 0.55211, 0.99088, 0.5943, 0.78767], 'file_name': './deploy/images/0517_2715693311.jpg', 'label_names': []}]
```


<a name="5"></a>
## 5. Predictive engine-based Prediction


<a name="5.1"></a>
### 5.1 Export inference model

```
python3 tools/export_model.py \
    -c ./ppcls/configs/quick_start/professional/MobileNetV1_multilabel.yaml \
    -o Arch.pretrained="./output/MobileNetV1/best_model"
```

The path of the inference model is by default under the current path `. /inference`.


<a name="5.2"></a>
### 5.2 Predictive engine-based Prediction

Go to the `deploy` first：

```
cd ./deploy
```

Inference and prediction through predictive engines:

```
python3 python/predict_cls.py \
     -c configs/inference_multilabel_cls.yaml
```

Obtain an output silimar to the following:

```
0517_2715693311.jpg:    class id(s): [6, 13, 17, 23, 26, 30], score(s): [0.96, 0.56, 0.55, 0.99, 0.59, 0.79], label_name(s): []
```
