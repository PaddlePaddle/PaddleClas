# PULC 模型库

------

此处提供了 PULC 模型库的相关指标和模型的下载链接，其中预训练模型可以用来微调训练，推理模型可以直接用来预测和部署。


|模型名称|模型简介|模型精度 |模型大小|推理耗时|下载地址|
| --- | --- | --- | --- | --- | --- |
| person_exists |[PULC有人/无人分类模型](PULC_person_exists.md)| 96.23 |7.0M|2.58ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/person_exists_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/person_exists_pretrained.pdparams)|
| person_attribute |[PULC人体属性识别模型](PULC_person_attribute.md)| 78.59 |7.2M|2.01ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/person_attribute_pretrained.pdparams)|
| safety_helmet |[PULC佩戴安全帽分类模型](PULC_safety_helmet.md)| 99.38 |7.1M|2.03ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/safety_helmet_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/safety_helmet_pretrained.pdparams)|
| traffic_sign |[PULC交通标志分类模型](PULC_traffic_sign.md)| 98.35 |8.2M|2.10ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/traffic_sign_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/traffic_sign_pretrained.pdparams)|
| vehicle_attribute |[PULC车辆属性识别模型](PULC_vehicle_attribute.md)| 90.81 |7.2M|2.36ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/vehicle_attribute_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/vehicle_attribute_pretrained.pdparams)|
| car_exists |[PULC有车/无车分类模型](PULC_car_exists.md) | 95.92 | 7.1M | 2.38ms |[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/car_exists_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/car_exists_pretrained.pdparams)|
| text_image_orientation |[PULC含文字图像方向分类模型](PULC_text_image_orientation.md)| 99.06 | 7.1M | 2.16ms |[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/text_image_orientation_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/text_image_orientation_pretrained.pdparams)|
| textline_orientation |[PULC文本行方向分类模型](PULC_textline_orientation.md)| 96.01 |7.0M|2.72ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/textline_orientation_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/textline_orientation_pretrained.pdparams)|
| language_classification |[PULC语种分类模型](PULC_language_classification.md)| 99.26 |7.1M|2.58ms|[推理模型](https://paddleclas.bj.bcebos.com/models/PULC/inference/language_classification_infer.tar) / [预训练模型](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/language_classification_pretrained.pdparams)|


**备注：**

* 以上所有的模型的 backbone 均为 PPLCNet_x1_0，部分模型大小不同是由于分类的输出大小不同导致的，推理耗时是基于Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz 测试得到，其中测试过程开启 MKLDNN 加速策略，线程数为10。速度测试过程会有轻微波动。

* person_exists、safety_helmet、car_exists 的评测指标为 TprAtFpr，person_attribute、vehicle_attribute的评测指标为ma、traffic_sign、text_image_orientation、textline_orientation、language_classification的评测指标为Top-1 Acc。
