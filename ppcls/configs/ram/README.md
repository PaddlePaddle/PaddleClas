# RAM, RAM_plus 图文标签模型

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

RAM以及RAM_plus为类CLIP模型，其中RAM的核心为提出了集训练-推理-打标签一体化的框架，通过堆叠vision encoder以及text encoder，实现多种打标签的下游任务，其核心方法：

1. RAM基于CLIP，提出Image-Tag recognition Decoder，image-text alignment encoder，image-tag interaction encoder，image-tag-text generation decoder以及generation encoder 5个组件分别实现text-image对齐，text-tag对齐。
2. RAM_plus作为RAM的升级版，进一步引出语言大模型（large language model，LLM）的语义信息，提升text-image对齐的能力。
3. RAM以及RAM_plus能够实现segmentation，tag等多种下游任务。

使用RAM以及RAM_plus时，作者在多个分类任务上取得了最先进的结果：
1. 在OpenImages零样本tag任务上，达到86.6%的mAP；
3. 在ImageNet零样本多标签分类上，达到72.4%的mAP。

`PaddleClas` paddleclas分别实现了基于不同backbone的ram，ram_plus:
```yaml
# model architecture
Arch:
  name: ram_plus # ram模型可以指定为ram，ram_plus可以指定为ram_plus默认为ram 
  vit: swin_l # 视觉主干网络参数，包括 vit：vision transformer，swin_b： swin base模型，swin_l, swin large模型
  image_size: 384 #图片分辨率
  pretrain_clip: ./ViT-B-32.pdparams # 训练所使用的CLIP 预训练参数路径，当需要训练ram时，不能为None
  stage: train # 指定ram，ram_plus模型是否进行训练，stage = train表示需要训练，此时pretrain_clip不能为None，stage = eval表示无需训练
  med_config: ppcls/configs/RAM/config_bert.yaml # bert based text encoder配置文件，默认 ppcls/configs/RAM/config_bert.yaml
  tag_list: ppcls/utils/RAM/ram_tag_list.txt # 英文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  tag_list_chinese: ppcls/utils/RAM/ram_tag_list.txt # 中文tag标签文件路劲，默认ppcls/utils/RAM/ram_tag_list.txt
  q2l_config: ppcls/configs/ram/config_q2l.yaml # 基于bert 的text-tag alignment encoder模型配置文件默认  ppcls/configs/ram/config_q2l.yaml 
  clip_version: vit-b-32-224 #训练时所需的CLIP架构
  ram_class_threshold_path: ppcls/utils/RAM/ram_tag_list_threshold.txt # tag生成阈值文件默认ppcls/utils/RAM/ram_tag_list_threshold.txt
```
注意，以上模型使用时，需要使用 tools/train_multimodal.py, tools/infer_multimodal.py 以及predict_multimodal.py分别进行支持多模态输入的动态图训练，推理以及静态图推理。

<a name="2"></a>
## 2. 数据和模型准备

* 基于 https://github.com/xinyu1205/recognize-anything/tree/main 下载对应数据集json文件，以及按照json文件目录格式，准备相应的数据。目录格式为：
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
其中务必保证数据集文件路径符合image-path
* 基于ppcls/configs/config_train.yaml，调整相关参数：

```yaml
# global configs
Global:
  checkpoints: null
  pretrained_model: "ram_plus.pdparams"
  output_dir: ./output/
  device: cpu
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
  use_amp: False
  use_fp16_test: False
  scale_loss: 128.0
  use_dynamic_loss_scaling: True
  use_promote: False
  # O1: mixed fp16, O2: pure fp16
  level: O1


# model architecture
Arch:
  name: ram_plus
  vit: swin_l
  pretrain_clip: ./ViT-B-32.pdparams
  stage: train
 
# loss function config for ram and ram_loss please note that RAMLoss only support ram and ram_plus
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
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    decay_epochs: [30, 60, 90]
    values: [0.01, 0.001, 0.0001, 0.00001]
  regularizer:
    name: 'L2'
    coeff: 0.0001


# data loader for ram and ram_plus dataset
DataLoader:
  Train:
    dataset:
      name: RAM_pretrain_dataset
      ann_file: [./visual-genome/vg_ram.json] # 步骤1中所涉及的文件参数，可以同时输入多个文件

    sampler:
      name: DistributedBatchSampler
      batch_size: 1
      drop_last: False
      shuffle: True
    loader:
      num_workers: 1
      use_shared_memory: True

  Eval:
    dataset: 
      name: RAM_pretrain_dataset
      ann_file: [./visual-genome/vg_ram.json] # 步骤1中所涉及的文件参数，可以同时输入多个文件
    sampler:
      name: DistributedBatchSampler
      batch_size: 1
      drop_last: False
      shuffle: False
    loader:
      num_workers: 1
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
  PostProcess: # RAM 以及RAM_plus 专用后处理，用于根据模型的输出组词
    name: RamOutPut
    language: "cn" # tag输出的语言，"cn"表示中文，"en"表示英文，"all"表示同时输出中英文，仅支持"cn","en" and "all"
    tag_list: "ppcls/utils/RAM/ram_tag_list.txt" # 英文标签文件
    tag_list_chinese: "ppcls/utils/RAM/ram_tag_list_chinese.txt" #中文标签文件
    ram_class_threshold_path: "ppcls/utils/RAM/ram_tag_list_threshold.txt" # 组词阈值文件，参照ram paper
```

## 3. 模型训练

```shell
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train_multimodal.py \
        -c ./ppcls/configs/ram/config_train.yaml
# 单卡
python3 tools/train_multimodal.py \
        -c ./ppcls/configs//ram/config_train.yaml
```

**注意:**
1. 目前多标签分类的损失函数默认使用`AsymmetricLoss`。
2. 目前多标签分类的评估指标默认使用`MAP(integral)`。

<a name="4"></a>

## 4. 模型评估

```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/config_train.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```

<a name="5"></a>
## 5. 模型预测

```bash
python3 tools/infer_multimodal.py \
    -c ./ppcls/configs/ram/config_train.yaml \
    -o Global.pretrained_model="./output/ram/best_model"
```

得到类似下面的输出：
```
{'class_ids': Tensor(shape=[6, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0   , 593 ],
        [0   , 871 ],
        [0   , 998 ],
        [0   , 2071],
        [0   , 3336],
        [0   , 3862]]),
'scores': Tensor(shape=[1, 4585], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-0.81945127, -1.42842841, -1.68710935, ..., -1.40692604,
         -1.96643567, -0.52991331]]), 
        
'label_names': ['棕色 | 鸡 | 公鸡 | 母鸡  | 红色 | 站/矗立/摊位']}
```

<a name="6"></a>
## 6. 基于预测引擎预测

<a name="6.1"></a>
### 6.1 导出 inference model

```bash
python3 tools/export_model.py \
    -c ./ppcls/configs/ram/config_train.yaml \
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
    -c ppcls/configs/ram/config_infer.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
# 使用下面的命令使用 CPU 进行预测
#更改 `config_infer.yaml` 配置文件后
python3 python/predict_multimodal.py \
    -c ppcls/configs/ram/config_infer.yaml \
    -o Global.inference_model_dir=../inference/ \
    -o Global.infer_imgs=docs/images/inference_deployment/whl_demo.jpg 
```

输出结果如下：

```
whl_demo.jpg-class_ids: Tensor(shape=[6, 2], dtype=int64, place=Place(cpu), stop_gradient=True,
       [[0   , 593 ],
        [0   , 871 ],
        [0   , 998 ],
        [0   , 2071],
        [0   , 3336],
        [0   , 3862]]),
whl_demo.jpg-scores: Tensor(shape=[1, 4585], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-0.81945127, -1.42842841, -1.68710935, ..., -1.40692604,
         -1.96643567, -0.52991331]]), 
        
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