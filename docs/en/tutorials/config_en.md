#Configuration

---

## Introduction

This document introduces the configuration(filed in `config/*.yaml`) of PaddleClas.

* Note: Some parameters do not appear in the yaml file (because they are not used for this file). During training or validation, you can use the command `-o` to update or add the specified parameters. For the example `-o checkpoints=./ckp_path/ppcls`, it means that the parameter `checkpoints` will be updated or added using the value `./ckp_path/ppcls`.

### Basic

| name | detail | default value | optional value |
|:---:|:---:|:---:|:---:|
| mode | mode | "train" | ["train"," valid"] |
| checkpoints | checkpoint model path for resuming training process | "" | Str |
| last_epoch | last epoch for the training，used with checkpoints | -1 | int |
| pretrained_model | pretrained model path | "" | Str |
| load_static_weights | whether the pretrained model is saved in static mode | False | bool |
| model_save_dir | model stored path | "" | Str |
| classes_num | class number | 1000 | int |
| total_images | total images | 1281167 | int |
| save_interval | save interval | 1 | int |
| validate | whether to validate when training | TRUE | bool |
| valid_interval | valid interval | 1 | int |
| epochs | epoch |  | int |
| topk | K value | 5 | int |
| image_shape | image size | [3，224，224] | list, shape: (3,) |
| use_mix | whether to use mixup | False | ['True', 'False'] |
| ls_epsilon | label_smoothing epsilon value| 0 | float |
| use_distillation | whether to use SSLD distillation training | False | bool |


## ARCHITECTURE

| name | detail | default value | optional value |
|:---:|:---:|:---:|:---:|
| name | model name | "ResNet50_vd" | one of 23 architectures |
| params | model parameters | {} | extra dictionary for the model structure, parameters such as `padding_type` in EfficientNet can be set here |


### LEARNING_RATE

| name | detail | default value |Optional value |
|:---:|:---:|:---:|:---:|
| function | decay type | "Linear" | ["Linear", "Cosine", <br> "Piecewise", "CosineWarmup"] |
| params.lr | initial learning rate | 0.1 | float |
| params.decay_epochs | milestone in piecewisedecay |  | list |
| params.gamma | gamma in piecewisedecay | 0.1 | float |
| params.warmup_epoch | warmup epoch | 5 | int |
| parmas.steps | decay steps in lineardecay | 100 | int |
| params.end_lr | end lr in lineardecay | 0 | float |

### OPTIMIZER

| name | detail | default value | optional value |
|:---:|:---:|:---:|:---:|
| function | optimizer name | "Momentum" | ["Momentum", "RmsProp"] |
| params.momentum | momentum value | 0.9 | float |
| regularizer.function | regularizer method name | "L2" | ["L1", "L2"] |
| regularizer.factor | regularizer factor | 0.0001 | float |

### reader

| name | detail |
|:---:|:---:|
| batch_size | batch size |
| num_workers | worker number |
| file_list | train list path |
| data_dir | train  dataset path |
| shuffle_seed | seed |

processing

| function name | attribute name | detail |
|:---:|:---:|:---:|
| DecodeImage | to_rgb | decode to RGB |
|  | to_np | to numpy |
|  | channel_first | Channel first |
| RandCropImage | size | random crop |
| RandFlipImage | | random flip |
| NormalizeImage | scale | normalize image |
|  | mean | mean |
|  | std | std |
|  | order | order |
| ToCHWImage |  | to CHW |
| CropImage | size | crop size |
| ResizeImage | resize_short | resize according to short size |

mix preprocessing

| name| detail|
|:---:|:---:|
| MixupOperator.alpha | alpha value in mixup|
