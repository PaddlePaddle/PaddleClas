# RAM, RAM++ 图文标签模型

## 目录

* [1. 模型介绍](#1)
* [2. 数据和模型准备](#2)
* [3. 模型训练](#3)
* [4. 模型评估](#4)
* [5. 模型预测](#5)
* [6. 基于预测引擎预测](#6)
  * [6.1 导出 inference model](#6.1)
  * [6.2 基于 Python 预测引擎推理](#6.2)
* [7. 引用](#7)

<a name="1"></a>
## 1. 模型介绍

RAM以及RAM++（下文简称RAM类模型）主要用于标注类任务，其中两个模型的主要贡献为提出了集训练-推理-tag一体化的框架。其通过堆叠vision encoder以及text encoder，实现多种下游任务。核心方法包括：

1. 结合CLIP架构，提出 Image-Tag recognition Decoder，image-text alignment encoder，image-tag interaction encoder，image-tag-text generation decoder以及generation encoder 5个组件分别实现text-image对齐，text-tag对齐。
2. RAM++进一步使用语言大模型（large language model，LLM）的语义信息，提升text-image对齐的能力。

使用RAM类模型时，作者在多个分类任务上取得了最先进的结果：

| Model | BackBone   | Size   | Inference Prompt | OpenImages-MAP |
|-------|------------|--------|------------------|----------------|
| RAM   | Swin-large | 5.63GB | LLM Tag Dec      | 82.2           |
| RAM++ | Swin-base  | 3.01GB | LLM Tag Dec      | 86.6           |

`PaddleClas` paddleclas分别实现了基于不同backbone的RAM类模型:
```yaml
# model architecture
Arch:
  name: ram_plus
  vit: swin_l
  med_config: 'ppcls/configs/ram/ram_bert.yaml'
  clip_pretraind: ./ViT-B-32.pdparams #for CLIP a necessary part for training ram
  stage: train
  image_size: 384
  vit_grad_ckpt: False
  vit_ckpt_layer: 0
  prompt: 'a picture of '
  threshold: 0.68
  delete_tag_index: []
  tag_list: 'ppcls/utils/ram/ram_tag_list.txt'
  tag_list_chinese: 'ppcls/utils/ram/ram_tag_list_chinese.txt'
  clip_version: 'vit-b-32-224'
  q2l_config: 'ppcls/configs/ram/ram_q2l.yaml'
  ram_class_threshold_path: 'ppcls/utils/RAM/ram_tag_list_threshold.txt'
```
参数注释：
  - name  模型参数，使用RAM模型可以指定为ram，使用RAM++模型可以指定为ram_plus，默认为ram 
  - vit  视觉主干网络参数，包括 vit：vision transformer，swin_b： swin base模型，swin_l, swin large模型
  - med_config RAM类模型所使用的Bert模型配置文件，默认配置路径：'ppcls/configs/ram/config_bert.yaml'
  - clip_pretraind 训练所使用的CLIP预训练参数路径，当需要训练RAM类模型时，不能为None
  - stage  指定RAM，RAM++模型是否进行训练，stage = train表示需要训练，训练时clip_pretraind不能为None，stage = eval表示无需训练
  - image_size  图片分辨率
  - prompt RAM训练时所使用的文本提示前缀
  - threshold 输出TAG所需的阈值数值，表示当该tag对应概率大于该值，则认为属于该tag
  - delete_tag_index 屏蔽tag所用参数，例如传递[1,3,2]则表示屏蔽index为1，2，3的tag标签
  - tag_list  英文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  - tag_list_chinese 中文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  - clip_version  所使用的CLIP结构，默认 vit-b-32-224
  - q2l_config  基于bert 的text-tag alignment encoder模型配置文件默认  ppcls/configs/ram/config_q2l.yaml 
  - ram_class_threshold_path  tag生成阈值文件默认ppcls/utils/RAM/ram_tag_list_threshold.txt
注意，RAM类模型的推理和训练，需要使用tools/train_multimodal.py, tools/infer_multimodal.py 以及predict_multimodal.py接口，支持多模态输入的动态图训练，推理以及静态图推理。

<a name="2"></a>
## 2. 数据和模型准备

* 前往官方[repo](https://github.com/xinyu1205/recognize-anything/tree/main)下载对应数据集json文件。同时按照json文件目录格式，准备相应的数据。目录格式为：
```json
{
    {
        "image_path": "visual-genome/VG_100K/1.jpg", 
        "caption": ["trees line the sidewalk"],
        "union_label_id": [4480], 
        "parse_label_id": [[4253, 2461, 2966]]
    }
}
```
其中务必保证数据集文件路径符合image_path
* 本文档中，为RAM类模型提供了统一的动态训练以及推理配置文件，结构如下：
```yaml
Global:
  checkpoints: null
  pretrained_model: "ram.pdparams" # pretrain model for ram and ram plus, default random initilize
  output_dir: ./output/
  device: gpu
  save_interval: 1
  eval_during_train: True
  eval_interval: 1
  epochs: 120
  print_batch_step: 10
  use_visualdl: False
  # used for static mode and model export
  image_shape: [3, 384, 384]
  save_inference_dir: ./inference


# mixed precision
AMP:
  use_amp: True
  use_fp16_test: False
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  use_promote: False
  # O1: mixed fp16, O2: pure fp16
  level: O1


# model architecture
Arch:
  name: ram
  vit: swin_l
  med_config: 'ppcls/configs/ram/ram_bert.yaml'
  clip_pretraind: ./ViT-B-32.pdparams # for CLIP a necessary part for training ram
  stage: train
  image_size: 384
  vit_grad_ckpt: False
  vit_ckpt_layer: 0
  prompt: 'a picture of '
  threshold: 0.68
  delete_tag_index: []
  tag_list: 'ppcls/utils/ram/ram_tag_list.txt'
  tag_list_chinese: 'ppcls/utils/ram/ram_tag_list_chinese.txt'
  clip_version: 'vit-b-32-224'
  q2l_config: 'ppcls/configs/ram/ram_q2l.yaml'
  ram_class_threshold_path: 'ppcls/utils/RAM/ram_tag_list_threshold.txt'
 
# loss function config for traing/eval process
Loss:
  Train:
    - RAMLoss:
        weight: 1.0
        mode: "pretrain"
  Eval:
    - RAMLoss:
        weight: 1.0
        mode: "pretrain"

Optimizer:
  name: AdamW
  beta1: 0.9
  beta2: 0.999
  epsilon: 1e-8
  weight_decay: 0.05
  layerwise_decay: 0.6
  filter_bias_and_bn: True
  lr:
    name: Cosine
    learning_rate: 0.0004
    eta_min: 1e-6
    warmup_epoch: 10
    warmup_start_lr: 5e-7

# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: RAMPretrainDataset
      ann_file: [./visual-genome/vg_ram.json]
      transform_ops_ram:
        - ResizeImage:
            size: 384
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
      transform_ops_clip:        
        - ResizeImage:
            resize_short: 224
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:

    sampler:
      name: DistributedBatchSampler
      batch_size: 52
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

  Eval:
    dataset: 
      name: RAMPretrainDataset
      ann_file: [./visual-genome/vg_ram.json]
      transform_ops_ram:
        - ResizeImage:
            size: 384
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
      transform_ops_clip:        
        - ResizeImage:
            resize_short: 224
            interpolation: bicubic
            backend: pil
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:

    sampler:
      name: DistributedBatchSampler
      batch_size: 52
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  batch_size: 1
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 384
    - CropImage:
        size: 384
    - NormalizeImage:
          scale: 1.0/255.0
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          order: ''
    - ToCHWImage:
  PostProcess:
    name: RamOutPut
    language: "cn"
    tag_list: "ppcls/utils/RAM/ram_tag_list.txt"
    tag_list_chinese: "ppcls/utils/RAM/ram_tag_list_chinese.txt"
    ram_class_threshold_path: "ppcls/utils/RAM/ram_tag_list_threshold.txt"

```
用户可以根据自身需求，更改相应配置。注意arch参数请参照本文档。

## 3. 模型训练
以RAM为例：
```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/RAM.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/RAM.yaml
```
以RAM++为例：
```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/RAM_plus.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/RAM_plus.yaml
```
**注意:**
1. 目前多标签分类的损失函数默认使用`AsymmetricLoss`。
2. 目前多标签分类的评估指标默认使用`MAP(integral)`。

<a name="4"></a>

## 4. 模型评估

```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/RAM.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```

<a name="5"></a>
## 5. 模型预测

```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/RAM.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```

得到类似下面的输出：
```
{'class_ids': [[0, 593], [0, 871], [0, 998], [0, 2071], [0, 3336], [0, 3862]], 'scores': [871], 'label_names': ['棕色 | 鸡 | 公鸡 | 母鸡  | 红色 | 站/矗立/摊位']}
```

<a name="6"></a>
## 6. 基于预测引擎预测

<a name="6.1"></a>
### 6.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/ram/RAM.yaml \
    -o Global.pretrained_model="./output/ram/"
```
inference model 的路径默认在当前路径下 `./inference`
`./inference` 文件夹下应有如下文件结构：

```
├── inference
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
```

<a name="6.2"></a>

### 6.2 基于 Python 预测引擎推理

切换到depoly目录下，并且使用deploy中的脚本进行推理前需要确认paddleclas为非本地安装, 如不是请进行切换，不然会出现包的导入错误。 

```shell
# 本地安装
pip install -e .
# 非本地安装
python setup.py install

# 进入deploy目录下
cd deploy
```

<a name="6.2.1"></a>  

#### 6.2.1 预测单张图像

运行下面的命令，对图像 `docs/images/inference_deployment/whl_demo.jpg` 进行分类。

```shell
# linux使用`python3`，windows使用`python (-m)`来执行脚本
# 使用下面的命令使用 GPU 进行预测
python3 python/predict_multimodal.py \
    -c deploy/configs/inference_ram.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
# 使用下面的命令使用 CPU 进行预测
#更改 `config_infer.yaml` 配置文件后
python3 python/predict_multimodal.py \
    -c deploy/configs/inference_ram.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
```

输出结果如下：

```
whl_demo.jpg-class_ids: [[0, 593], [0, 871], [0, 998], [0, 2071], [0, 3336], [0, 3862]],
whl_demo.jpg-scores: [871], 
whl_demo.jpg-label_names: ['棕色 | 鸡 | 公鸡 | 母鸡  | 红色 | 站/矗立/摊位']
```


<a name="7"></a>
## 7. 引用
```
@article{huang2023inject,
  title={Inject Semantic Concepts into Image Tagging for Open-Set Recognition},
  author={Huang, Xinyu and Huang, Yi-Jie and Zhang, Youcai and Tian, Weiwei and Feng, Rui and Zhang, Yuejie and Xie, Yanchun and Li, Yaqian and Zhang, Lei},
  journal={arXiv preprint arXiv:2310.15200},
  year={2023}
}

@article{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model},
  author={Zhang, Youcai and Huang, Xinyu and Ma, Jinyu and Li, Zhaoyang and Luo, Zhaochuan and Xie, Yanchun and Qin, Yuzhuo and Luo, Tong and Li, Yaqian and Liu, Shilong and others},
  journal={arXiv preprint arXiv:2306.03514},
  year={2023}
}
```