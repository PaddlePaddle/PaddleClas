# PULC Model Zoo

------

The PULC model zoo is provided here, mainly providing indicators, model storage size, and download links of the model. The pre-trained model can be used for fine-tuning training, and the inference model can be directly used for prediction and deployment.


|Model name| Model Description | Metrics |Storage Size| Latency| Download Address|
| --- | --- | --- | --- | --- | --- |
| person_exists |[Human Exists Classification](PULC_person_exists_en.md)| 96.23 |7.0M|2.58ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/person_exists_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/person_exists_pretrained.pdparams)|
| person_attribute |[Pedestrian Attribute Classification](PULC_person_attribute_en.md)| 78.59 |7.2M|2.01ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/person_attribute_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/person_attribute_pretrained.pdparams)|
| safety_helmet |[Classification of Wheather Wearing Safety Helmet](PULC_safety_helmet_en.md)| 99.38 |7.1M|2.03ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/safety_helmet_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/safety_helmet_pretrained.pdparams)|
| traffic_sign |[Traffic Sign Classification](PULC_traffic_sign_en.md)| 98.35 |8.2M|2.10ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/traffic_sign_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/traffic_sign_pretrained.pdparams)|
| vehicle_attribute |[Vehicle Attribute Classification](PULC_vehicle_attribute_en.md)| 90.81 |7.2M|2.36ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/vehicle_attribute_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/vehicle_attribute_pretrained.pdparams)|
| car_exists |[Car Exists Classification](PULC_car_exists_en.md) | 95.92 | 7.1M | 2.38ms |[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/car_exists_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/car_exists_pretrained.pdparams)|
| text_image_orientation |[Text Image Orientation Classification](PULC_text_image_orientation_en.md)| 99.06 | 7.1M | 2.16ms |[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/text_image_orientation_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/text_image_orientation_pretrained.pdparams)|
| textline_orientation |[Text-line Orientation Classification](PULC_textline_orientation_en.md)| 96.01 |7.0M|2.72ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/textline_orientation_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/textline_orientation_pretrained.pdparams)|
| language_classification |[Language Classification](PULC_language_classification_en.md)| 99.26 |7.1M|2.58ms|[inference model](https://paddleclas.bj.bcebos.com/models/PULC/inference/language_classification_infer.tar) / [pretrained model](https://paddleclas.bj.bcebos.com/models/PULC/pretrained/language_classification_pretrained.pdparams)|


**Note：**

* The backbone of all the above models is PPLCNet_x1_0. The different sizes of some models are caused by the different output sizes of the classification layer. The inference time is tested on the Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz. During the test process, the MKLDNN acceleration strategy is turned on, and the number of threads is 10. There will be slight fluctuations during the speed test process.

* The evaluation indicators of person_exists, safety_helmet, and car_exists are TprAtFpr. The evaluation indicators of person_attribute and vehicle_attribute are ma. The evaluation indicators of traffic_sign, text_image_orientation, textline_orientation and language_classification are Top-1 Acc.
