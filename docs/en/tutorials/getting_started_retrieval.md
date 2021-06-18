# Getting Started
## Note: This tutorial focuses on retrival-based image recognition 
---
Please take a referencre to [Installation Guide](./install.md) to configure the PaddleClas environment.Follow [Quick Start](./quick_start_new_user.md) to prepare flowers102 dataset.All experiments in the following sections of this chapter are based on the flowers102 dataset.


PaddleClas currently supports the following training/evaluation environments:
```shell
└── CPU/Single GPU
    ├── Linux
    └── Windows

└── Multi GPU
    └── Linux
```

## 1. Training and evaluation on CPU/single GPU

For training and evaluation on CPU/single GPU, it is recommended to use `tools/train.py` and `tools/eval.py`.For training and evaluation in a multi-GPU environment on Linux platforms, please refer to[2. Linux+GPU based model training & evaluation](#2).

<a name="1.1"></a>
### 1.1 Model Training

Once prepared the configuration file, you can start the training as following

```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.use_gpu=True
```

 `-c` is used to specify the path to the configuration file, `-o` is used to specify the parameters that need to be modified or added, where `-o use_gpu=True` indicates that the GPU is used for training. If you want to use the CPU for training, you need to set `use_gpu` to `False`.

For more detailed training configuration, you can also modify the corresponding configuration file of the model directly. For specific configuration parameters, please refer to the [ConfigurationFile].(config.md)。

Loss changes can be observed in real time during training via VisualDL, refer to [VisualDL](../extension/VisualDL.md)。

### 1.2 Model Fintune

After setting up the profile according to your own dataset path, you can fine-tune it by loading the pre-trained model as shown below.


```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Arch.Backbone.pretrained=True
```

 `-o Arch.Backbone.pretrained` is used to set whether or not to load the pre-trained model; when it is True, the pre-trained model will be automatically downloaded and loaded.

<a name="1.3"></a>
### 1.3 Model Recovery Training

If the training task is terminated for other reasons, it can be recovered by loading the breakpoint weights file to continue training.

```
python tools/train.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.checkpoints="./output/RecModel/epoch_5" \
```
Simply set the `Global.checkpoints` parameter when continuing training, indicating the path to the loaded breakpoint weights file, using this parameter will load both the saved breakpoint weights and information about the learning rate, optimizer, etc.

<a name="1.4"></a>
### 1.4 Model Evaluation

Model evaluation can be carried out with the following commands.

```bash
python tools/eval.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model="./output/RecModel/best_model"\
```
where `-o Global.pretrained_model` is used to set the path of the model to be evaluated


<a name="2"></a>
## 2. Linux+GPU based model training & evaluation

If the machine environment is Linux+GPU, then it is recommended to use `paddle.distributed.launch` to start the model training script (`tools/train.py`) and the evaluation script (`tools/eval.py`), which makes it easier to start multi-GPU training and evaluation.

### 2.1 Model Training

Referring to the following way to start model training, `paddle.distributed.launch` specifies the GPU run card number by setting `gpus` to.

```bash
# PaddleClas launches multi-card multi-process training via launch

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml
```

### 2.2 Model Finetune

After configuring the profile according to your own dataset, you can load the pre-trained model for fine-tuning, as shown below.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
        -o Arch.Backbone.pretrained=True
```

### 2.3 Model Recovery Training

If the training is terminated for other reasons, the breakpoint weights file can also be loaded to continue training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
        -o Global.checkpoints="./output/RecModel/epoch_5" \
```

### 2.4 Model Evaluation

Model evaluation can be carried out with the following commands.

```bash
python. -m paddle.distributed.launch \ 
    --gpus="0,1,2,3" \
    tools/eval.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model="./output/RecModel/best_model"\
```

<a name="model_inference"></a>
## 3. Inference Using Inference Model
### 3.1 Export Inference Models

By exporting inference models, PaddlePaddle supports predictive inference using a prediction engine. The following introduces how inference is performed with the prediction engine.
First, the trained model is transformed.

```bash
python tools/export_model.py \
    -c ppcls/configs/quick_start/ResNet50_vd_finetune_retrieval.yaml \
    -o Global.pretrained_model=./output/RecModel/best_model \
    -o Global.save_inference_dir=./inference \
```

 `--Global.pretrained_model` is used to specify the model file path, which don't need contain the model file suffix (e.g. [1.3 Model Recovery Training](#1.3)), and `--Global.save_inference_dir` is used to specify the path where the converted model is stored.
If `--save_inference_dir=. /inference`, then the `inference.pdiparams`, `inference.pdmodel` and `inference.pdiparams.info` files will be generated in the `inference` folder.


### 3.2 Build Database

To perform image recognition by retrieval, you need to build the retrival database.
First, copy the generated model to the deploy directory and go back to the deploy directory.
```bash
mv ./inference ./deploy
cd deploy
```

Next, build the base library with the following command.
```bash
python python/build_gallery.py \
       -c configs/build_flowers.yaml \
       -o Global.rec_inference_model_dir="./inference" \
       -o IndexProcess.index_path="../dataset/flowers102/index" \
       -o IndexProcess.image_root="../dataset/flowers102/" \
       -o IndexProcess.data_file="../dataset/flowers102/train_list.txt" 
```

+ `Global.rec_inference_model_dir`: the path to the inference model generated by 3.1
+ `IndexProcess.index_path`: path to the index of the gallery library
+ `IndexProcess.image_root`: the root directory of the gallery's images
+ `IndexProcess.data_file`: the list of files for the gallery images
After executing the above command, a list of files will be created in `... /dataset/flowers102` directory, the index directory contains 3 files `index.data`, `1index.graph`, `info.json`


### 3.3 Inference & Prediction

The model structure file (`inference.pdmodel`) and the model weights file (`inference.pdiparams`) are generated by 3.1, the retrival database is built by 3.2, and the prediction engine can then be used to make inferences: 

```bash
python python/predict_rec.py \
    -c configs/inference_flowers.yaml \
    -o Global.infer_imgs="./images/image_00002.jpg" \
    -o Global.rec_inference_model_dir="./inference" \
    -o Global.use_gpu=True \
    -o Global.use_tensorrt=False
```

+ `Global.infer_imgs`: the path to the image file to be predicted, e.g. `. /images/image_00002.jpg`
+ `Global.rec_inference_model_dir`: the path to the prediction model file, e.g. `. /inference/`
+ `Global.use_tensorrt`: whether to use the TesorRT prediction engine, default value: `True`
+ `Global.use_gpu`: whether to use GPU prediction, default value: `True` 

After executing the above commands, you will get the feature information corresponding to the input image, in this case the feature dimension is 2048, and the log will show this:
```
(1, 2048)
[[0.00033124 0.00056205 0.00032261 ... 0.00030939 0.00050748 0.00030271]]
```
