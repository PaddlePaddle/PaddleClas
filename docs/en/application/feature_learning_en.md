# Feature Learning

This part mainly explains the training mode of feature learning, which is `RecModel` training mode in code. The main purpose of feature learning is to support the application, such as vehicle recognition (vehicle fine-grained classification, vehicle Reid), logo recognition,  cartoon character recognition , product recognition, which needs to learn robust features to identify objects. Different from training classification network on Imagenet, this feature learning part mainly has the following features:

- Support to truncate the `backbone`, which means feature of any intermediate layer can be extracted

- Support to add configurable  layers after `backbone` output, namely `Neck`

- Support `Arcface Loss` and other `metric learning`loss functions to improve feature learning ability

# Pipeline

![](../../images/recognition/rec_pipeline.png)

The overall structure of feature learning is shown in the figure above, which mainly includes `Data Augmentation`, `Backbone`, `Neck`, `Metric Learning` and so on. The `Neck` part is a freely added  layers, such as  `Embedding layer`. Of course, this module can be omitted if not needed. During training, the loss of `Metric Learning`  is used to optimize the model. Generally speaking, the output of the `Neck`  is used as the feature output when in inference stage.

## Config Description

The feature learning config file description can be found in [yaml description](../tutorials/config_en.md).
