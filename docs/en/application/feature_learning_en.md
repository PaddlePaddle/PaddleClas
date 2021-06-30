# Feature Learning

This part mainly explains the training mode of feature learning, which is `RecModel` training mode in code. The main purpose of feature learning is to support the application, such as vehicle recognition (vehicle fine-grained classification, vehicle Reid), logo recognition,  cartoon character recognition , product recognition, which needs to learn robust features to identify objects. Different from training classification network on Imagenet, this feature learning part mainly has the following features:

- Support to truncate the `backbone`, which means feature of any intermediate layer can be extracted

- Support to add configurable  layers after `backbone` output, namely `Neck`

- Support `Arcface Loss` and other `metric learning`loss functions to improve feature learning ability

# 1 Pipeline

![](../../images/recognition/rec_pipeline.png)

The overall structure of feature learning is shown in the figure above, which mainly includes `Data Augmentation`, `Backbone`, `Neck`, `Metric Learning` and so on. The `Neck` part is a freely added  layers, such as  `Embedding layer`. Of course, this module can be omitted if not needed. During training, the loss of `Metric Learning`  is used to optimize the model. Generally speaking, the output of the `Neck`  is used as the feature output when in inference stage.

## 2 Config Description

The feature learning config file description can be found in [yaml description](../tutorials/config_en.md).

## 3 Pretrained Model

The following are the pretrained models trained on different dataset.

- Vehicle Fine-Grained Classification：[CompCars](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/vehicle_cls_ResNet50_CompCars_v1.1_pretrained.pdparams)
- Vehicle ReID：[VERI-Wild](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/vehicle_reid_ResNet50_VERIWild_v1.0_pretrained.pdparams)
- Cartoon Character Recognition：[iCartoon](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/cartoon_rec_ResNet50_iCartoon_v1.0_pretrained.pdparams)
- Logo Recognition：[Logo 3K](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/logo_rec_ResNet50_Logo3K_v1.0_pretrained.pdparams)
- Product Recognition： [Inshop](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Inshop_pretrained_v1.0.pdparams)、[Aliproduct](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/pretrain/product_ResNet50_vd_Aliproduct_v1.0_pretrained.pdparams) 

