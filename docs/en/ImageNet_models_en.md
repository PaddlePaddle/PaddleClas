### ImageNet Model zoo overview

Based on the ImageNet-1k classification dataset, the 35 classification network structures supported by PaddleClas and the corresponding 164 image classification pretrained models are shown below. Training trick, a brief introduction to each series of network structures, and performance evaluation will be shown in the corresponding chapters. The  evaluation environment is as follows.

* CPU evaluation environment is based on Snapdragon 855 (SD855).
* The GPU evaluation speed is measured by running 500 times under the FP32+TensorRT configuration (excluding the warmup time of the first 10 times).


Curves of accuracy to the inference time of common server-side models are shown as follows.

![](../images/models/T4_benchmark/t4.fp32.bs1.main_fps_top1.png)


Curves of accuracy to the inference time and storage size of common mobile-side models are shown as follows.

![](../images/models/mobile_arm_storage.png)

![](../images/models/mobile_arm_top1.png)

<a name="SSLD_pretrained_series"></a>
### SSLD pretrained models
Accuracy and inference time of the prtrained models based on SSLD distillation are as follows. More detailed information can be refered to [SSLD distillation tutorial](../en/advanced_tutorials/distillation/distillation_en.md).

* Server-side distillation pretrained models

| Model                  | Top-1 Acc | Reference<br>Top-1 Acc | Acc gain | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|---------------------|-----------|-----------|---------------|----------------|-----------|----------|-----------|-----------------------------------|
| ResNet34_vd_ssld         | 0.797    | 0.760  | 0.037  | 2.434               | 6.222              | 7.39     | 21.82     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_ssld_pretrained.pdparams)         |
| ResNet50_vd_<br>ssld | 0.830    | 0.792    | 0.039 | 3.531               | 8.090              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_ssld_pretrained.pdparams) |
| ResNet101_vd_<br>ssld   | 0.837    | 0.802    | 0.035 |  6.117               | 13.762             | 16.1     | 44.57     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_ssld_pretrained.pdparams)   |
| Res2Net50_vd_<br>26w_4s_ssld | 0.831    | 0.798    | 0.033 |  4.527              | 9.657             | 8.37     | 25.06     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_ssld_pretrained.pdparams) |
| Res2Net101_vd_<br>26w_4s_ssld | 0.839    | 0.806    | 0.033 | 8.087              | 17.312             | 16.67    | 45.22     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_ssld_pretrained.pdparams) |
| Res2Net200_vd_<br>26w_4s_ssld | 0.851    | 0.812    | 0.049 | 14.678              | 32.350             | 31.49    | 76.21     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_ssld_pretrained.pdparams) |
| HRNet_W18_C_ssld | 0.812    | 0.769   | 0.043 | 7.406          | 13.297         | 4.14     | 21.29     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_ssld_pretrained.pdparams) |
| HRNet_W48_C_ssld | 0.836    | 0.790   | 0.046  | 13.707         | 34.435         | 34.58    | 77.47     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_ssld_pretrained.pdparams) |
| SE_HRNet_W64_C_ssld | 0.848    |  -    |  - |  31.697      |     94.995      | 57.83    | 128.97    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SE_HRNet_W64_C_ssld_pretrained.pdparams) |


* Mobile-side distillation pretrained models

| Model                  | Top-1 Acc | Reference<br>Top-1 Acc | Acc gain | SD855 time(ms)<br>bs=1 | Flops(G) | Params(M) | 模型大小(M) | Download Address  |
|---------------------|-----------|-----------|---------------|----------------|-----------|----------|-----------|-----------------------------------|
| MobileNetV1_<br>ssld   | 0.779    | 0.710    | 0.069 |  32.523              | 1.11     | 4.19      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_ssld_pretrained.pdparams)                 |
| MobileNetV2_<br>ssld                 | 0.767    | 0.722  | 0.045  | 23.318              | 0.6      | 3.44      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams)                 |
| MobileNetV3_<br>small_x0_35_ssld          | 0.556    | 0.530 | 0.026   | 2.635                 | 0.026    | 1.66      | 6.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_ssld_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0_ssld      | 0.790    | 0.753  | 0.036  | 19.308           | 0.45     | 5.47      | 21      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)      |
| MobileNetV3_small_<br>x1_0_ssld      | 0.713    | 0.682  |  0.031  | 6.546                 | 0.123    | 2.94      | 12      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)      |
| GhostNet_<br>x1_3_ssld                    | 0.794    | 0.757   | 0.037 | 19.983                | 0.44     | 7.3       | 29      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_ssld_pretrained.pdparams)  


* Note: `Reference Top-1 Acc` means accuracy of pretrained models which are trained on ImageNet1k dataset.


<a name="ResNet_and_Vd_series"></a>
### ResNet and Vd series

Accuracy and inference time metrics of ResNet and Vd series models are shown as follows. More detailed information can be refered to [ResNet and Vd series tutorial](../en/models/ResNet_and_vd_en.md).

| Model                  | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|---------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------|
| ResNet18            | 0.7098    | 0.8992    | 1.45606               | 3.56305              | 3.66     | 11.69     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_pretrained.pdparams)            |
| ResNet18_vd         | 0.7226    | 0.9080    | 1.54557               | 3.85363              | 4.14     | 11.71     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_vd_pretrained.pdparams)         |
| ResNet34            | 0.7457    | 0.9214    | 2.34957               | 5.89821              | 7.36     | 21.8      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_pretrained.pdparams)            |
| ResNet34_vd         | 0.7598    | 0.9298    | 2.43427               | 6.22257              | 7.39     | 21.82     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_pretrained.pdparams)         |
| ResNet34_vd_ssld         | 0.7972    | 0.9490    | 2.43427               | 6.22257              | 7.39     | 21.82     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_ssld_pretrained.pdparams)         |
| ResNet50            | 0.7650    | 0.9300    | 3.47712               | 7.84421              | 8.19     | 25.56     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_pretrained.pdparams)            |
| ResNet50_vc         | 0.7835    | 0.9403    | 3.52346               | 8.10725              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vc_pretrained.pdparams)         |
| ResNet50_vd         | 0.7912    | 0.9444    | 3.53131               | 8.09057              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams)         |
| ResNet101           | 0.7756    | 0.9364    | 6.07125               | 13.40573             | 15.52    | 44.55     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_pretrained.pdparams)           |
| ResNet101_vd        | 0.8017    | 0.9497    | 6.11704               | 13.76222             | 16.1     | 44.57     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_pretrained.pdparams)        |
| ResNet152           | 0.7826    | 0.9396    | 8.50198               | 19.17073             | 23.05    | 60.19     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_pretrained.pdparams)           |
| ResNet152_vd        | 0.8059    | 0.9530    | 8.54376               | 19.52157             | 23.53    | 60.21     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_vd_pretrained.pdparams)        |
| ResNet200_vd        | 0.8093    | 0.9533    | 10.80619              | 25.01731             | 30.53    | 74.74     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet200_vd_pretrained.pdparams)        |
| ResNet50_vd_<br>ssld | 0.8300    | 0.9640    | 3.53131               | 8.09057              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_ssld_pretrained.pdparams) |
| ResNet101_vd_<br>ssld   | 0.8373    | 0.9669    | 6.11704               | 13.76222             | 16.1     | 44.57     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_ssld_pretrained.pdparams)   |


<a name="Mobile_series"></a>
### Mobile series

Accuracy and inference time metrics of Mobile series models are shown as follows. More detailed information can be refered to [Mobile series tutorial](../en/models/Mobile_en.md).

| Model                              | Top-1 Acc | Top-5 Acc | SD855 time(ms)<br>bs=1 | Flops(G) | Params(M) | Model storage size(M) | Download Address                                                                                                      |
|----------------------------------|-----------|-----------|------------------------|----------|-----------|---------|-----------------------------------------------------------------------------------------------------------|
| MobileNetV1_<br>x0_25                | 0.5143    | 0.7546    | 3.21985                | 0.07     | 0.46      | 1.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_25_pretrained.pdparams)                |
| MobileNetV1_<br>x0_5                 | 0.6352    | 0.8473    | 9.579599               | 0.28     | 1.31      | 5.2     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_5_pretrained.pdparams)                 |
| MobileNetV1_<br>x0_75                | 0.6881    | 0.8823    | 19.436399              | 0.63     | 2.55      | 10      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_x0_75_pretrained.pdparams)                |
| MobileNetV1                      | 0.7099    | 0.8968    | 32.523048              | 1.11     | 4.19      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_pretrained.pdparams)                      |
| MobileNetV1_<br>ssld                 | 0.7789    | 0.9394    | 32.523048              | 1.11     | 4.19      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV1_ssld_pretrained.pdparams)                 |
| MobileNetV2_<br>x0_25                | 0.5321    | 0.7652    | 3.79925                | 0.05     | 1.5       | 6.1     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams)                |
| MobileNetV2_<br>x0_5                 | 0.6503    | 0.8572    | 8.7021                 | 0.17     | 1.93      | 7.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x0_75                | 0.6983    | 0.8901    | 15.531351              | 0.35     | 2.58      | 10      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams)                |
| MobileNetV2                      | 0.7215    | 0.9065    | 23.317699              | 0.6      | 3.44      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams)                      |
| MobileNetV2_<br>x1_5                 | 0.7412    | 0.9167    | 45.623848              | 1.32     | 6.76      | 26      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x2_0                 | 0.7523    | 0.9258    | 74.291649              | 2.32     | 11.13     | 43      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams)                 |
| MobileNetV2_<br>ssld                 | 0.7674    | 0.9339    | 23.317699              | 0.6      | 3.44      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams)                 |
MobileNetV3_<br>large_x1_25          | 0.7641    | 0.9295    | 28.217701              | 0.714    | 7.44      | 29      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0           | 0.7532    | 0.9231    | 19.30835               | 0.45     | 5.47      | 21      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_75          | 0.7314    | 0.9108    | 13.5646                | 0.296    | 3.91      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>large_x0_5           | 0.6924    | 0.8852    | 7.49315                | 0.138    | 2.67      | 11      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_35          | 0.6432    | 0.8546    | 5.13695                | 0.077    | 2.1       | 8.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_25          | 0.7067    | 0.8951    | 9.2745                 | 0.195    | 3.62      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_0           | 0.6824    | 0.8806    | 6.5463                 | 0.123    | 2.94      | 12      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_75          | 0.6602    | 0.8633    | 5.28435                | 0.088    | 2.37      | 9.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_5           | 0.5921    | 0.8152    | 3.35165                | 0.043    | 1.9       | 7.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_35          | 0.5303    | 0.7637    | 2.6352                 | 0.026    | 1.66      | 6.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_35_ssld          | 0.5555    | 0.7771    | 2.6352                 | 0.026    | 1.66      | 6.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x0_35_ssld_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0_ssld      | 0.7896    | 0.9448    | 19.30835               | 0.45     | 5.47      | 21      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)      |
| MobileNetV3_small_<br>x1_0_ssld      | 0.7129    | 0.9010    | 6.5463                 | 0.123    | 2.94      | 12      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)      |
| ShuffleNetV2                     | 0.6880    | 0.8845    | 10.941                 | 0.28     | 2.26      | 9       | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams)                     |
| ShuffleNetV2_<br>x0_25               | 0.4990    | 0.7379    | 2.329                  | 0.03     | 0.6       | 2.7     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_33               | 0.5373    | 0.7705    | 2.64335                | 0.04     | 0.64      | 2.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_5                | 0.6032    | 0.8226    | 4.2613                 | 0.08     | 1.36      | 5.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x1_5                | 0.7163    | 0.9015    | 19.3522                | 0.58     | 3.47      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x2_0                | 0.7315    | 0.9120    | 34.770149              | 1.12     | 7.32      | 28      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams)                |
| ShuffleNetV2_<br>swish               | 0.7003    | 0.8917    | 16.023151              | 0.29     | 2.26      | 9.1     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams)               |
| GhostNet_<br>x0_5                    | 0.6688    | 0.8695    | 5.7143                 | 0.082    | 2.6       | 10      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x0_5_pretrained.pdparams)               |
| GhostNet_<br>x1_0                    | 0.7402    | 0.9165    | 13.5587                | 0.294    | 5.2       | 20      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams)               |
| GhostNet_<br>x1_3                    | 0.7579    | 0.9254    | 19.9825                | 0.44     | 7.3       | 29      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_pretrained.pdparams)               |
| GhostNet_<br>x1_3_ssld                    | 0.7938    | 0.9449    | 19.9825                | 0.44     | 7.3       | 29      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_3_ssld_pretrained.pdparams)               |


<a name="SEResNeXt_and_Res2Net_series"></a>
### SEResNeXt and Res2Net series

Accuracy and inference time metrics of SEResNeXt and Res2Net series models are shown as follows. More detailed information can be refered to [SEResNext and_Res2Net series tutorial](../en/models/SEResNext_and_Res2Net_en.md).


| Model                 | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|---------------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| Res2Net50_<br>26w_4s          | 0.7933    | 0.9457    | 4.47188               | 9.65722              | 8.52     | 25.7      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_26w_4s_pretrained.pdparams)          |
| Res2Net50_vd_<br>26w_4s       | 0.7975    | 0.9491    | 4.52712               | 9.93247              | 8.37     | 25.06     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_vd_26w_4s_pretrained.pdparams)       |
| Res2Net50_<br>14w_8s          | 0.7946    | 0.9470    | 5.4026                | 10.60273             | 9.01     | 25.72     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net50_14w_8s_pretrained.pdparams)          |
| Res2Net101_vd_<br>26w_4s      | 0.8064    | 0.9522    | 8.08729               | 17.31208             | 16.67    | 45.22     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net101_vd_26w_4s_pretrained.pdparams)      |
| Res2Net200_vd_<br>26w_4s      | 0.8121    | 0.9571    | 14.67806              | 32.35032             | 31.49    | 76.21     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_pretrained.pdparams)      |
| Res2Net200_vd_<br>26w_4s_ssld | 0.8513    | 0.9742    | 14.67806              | 32.35032             | 31.49    | 76.21     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Res2Net200_vd_26w_4s_ssld_pretrained.pdparams) |
| ResNeXt50_<br>32x4d           | 0.7775    | 0.9382    | 7.56327               | 10.6134              | 8.02     | 23.64     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_32x4d_pretrained.pdparams)           |
| ResNeXt50_vd_<br>32x4d        | 0.7956    | 0.9462    | 7.62044               | 11.03385             | 8.5      | 23.66     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_32x4d_pretrained.pdparams)        |
| ResNeXt50_<br>64x4d           | 0.7843    | 0.9413    | 13.80962              | 18.4712              | 15.06    | 42.36     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_64x4d_pretrained.pdparams)           |
| ResNeXt50_vd_<br>64x4d        | 0.8012    | 0.9486    | 13.94449              | 18.88759             | 15.54    | 42.38     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt50_vd_64x4d_pretrained.pdparams)        |
| ResNeXt101_<br>32x4d          | 0.7865    | 0.9419    | 16.21503              | 19.96568             | 15.01    | 41.54     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x4d_pretrained.pdparams)          |
| ResNeXt101_vd_<br>32x4d       | 0.8033    | 0.9512    | 16.28103              | 20.25611             | 15.49    | 41.56     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_32x4d_pretrained.pdparams)       |
| ResNeXt101_<br>64x4d          | 0.7835    | 0.9452    | 30.4788               | 36.29801             | 29.05    | 78.12     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_64x4d_pretrained.pdparams)          |
| ResNeXt101_vd_<br>64x4d       | 0.8078    | 0.9520    | 30.40456              | 36.77324             | 29.53    | 78.14     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_vd_64x4d_pretrained.pdparams)       |
| ResNeXt152_<br>32x4d          | 0.7898    | 0.9433    | 24.86299              | 29.36764             | 22.01    | 56.28     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_32x4d_pretrained.pdparams)          |
| ResNeXt152_vd_<br>32x4d       | 0.8072    | 0.9520    | 25.03258              | 30.08987             | 22.49    | 56.3      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_32x4d_pretrained.pdparams)       |
| ResNeXt152_<br>64x4d          | 0.7951    | 0.9471    | 46.7564               | 56.34108             | 43.03    | 107.57    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_64x4d_pretrained.pdparams)          |
| ResNeXt152_vd_<br>64x4d       | 0.8108    | 0.9534    | 47.18638              | 57.16257             | 43.52    | 107.59    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt152_vd_64x4d_pretrained.pdparams)       |
| SE_ResNet18_vd            | 0.7333    | 0.9138    | 1.7691                | 4.19877              | 4.14     | 11.8      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet18_vd_pretrained.pdparams)            |
| SE_ResNet34_vd            | 0.7651    | 0.9320    | 2.88559               | 7.03291              | 7.84     | 21.98     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet34_vd_pretrained.pdparams)            |
| SE_ResNet50_vd            | 0.7952    | 0.9475    | 4.28393               | 10.38846             | 8.67     | 28.09     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNet50_vd_pretrained.pdparams)            |
| SE_ResNeXt50_<br>32x4d        | 0.7844    | 0.9396    | 8.74121               | 13.563               | 8.02     | 26.16     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_32x4d_pretrained.pdparams)        |
| SE_ResNeXt50_vd_<br>32x4d     | 0.8024    | 0.9489    | 9.17134               | 14.76192             | 10.76    | 26.28     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt50_vd_32x4d_pretrained.pdparams)     |
| SE_ResNeXt101_<br>32x4d       | 0.7939    | 0.9443    | 18.82604              | 25.31814             | 15.02    | 46.28     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SE_ResNeXt101_32x4d_pretrained.pdparams)       |
| SENet154_vd               | 0.8140    | 0.9548    | 53.79794              | 66.31684             | 45.83    | 114.29    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SENet154_vd_pretrained.pdparams)               |


<a name="DPN_and_DenseNet_series"></a>
### DPN and DenseNet series

Accuracy and inference time metrics of DPN and DenseNet series models are shown as follows. More detailed information can be refered to [DPN and DenseNet series tutorial](../en/models/DPN_DenseNet_en.md).


| Model                 | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|-------------|-----------|-----------|-----------------------|----------------------|----------|-----------|--------------------------------------------------------------------------------------|
| DenseNet121 | 0.7566    | 0.9258    | 4.40447               | 9.32623              | 5.69     | 7.98      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams) |
| DenseNet161 | 0.7857    | 0.9414    | 10.39152              | 22.15555             | 15.49    | 28.68     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams) |
| DenseNet169 | 0.7681    | 0.9331    | 6.43598               | 12.98832             | 6.74     | 14.15     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams) |
| DenseNet201 | 0.7763    | 0.9366    | 8.20652               | 17.45838             | 8.61     | 20.01     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams) |
| DenseNet264 | 0.7796    | 0.9385    | 12.14722              | 26.27707             | 11.54    | 33.37     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams) |
| DPN68       | 0.7678    | 0.9343    | 11.64915              | 12.82807             | 4.03     | 10.78     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN68_pretrained.pdparams)       |
| DPN92       | 0.7985    | 0.9480    | 18.15746              | 23.87545             | 12.54    | 36.29     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN92_pretrained.pdparams)       |
| DPN98       | 0.8059    | 0.9510    | 21.18196              | 33.23925             | 22.22    | 58.46     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN98_pretrained.pdparams)       |
| DPN107      | 0.8089    | 0.9532    | 27.62046              | 52.65353             | 35.06    | 82.97     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN107_pretrained.pdparams)      |
| DPN131      | 0.8070    | 0.9514    | 28.33119              | 46.19439             | 30.51    | 75.36     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DPN131_pretrained.pdparams)      |

<a name="HRNet_series"></a>
### HRNet series

Accuracy and inference time metrics of HRNet series models are shown as follows. More detailed information can be refered to [Mobile series tutorial](../en/models/HRNet_en.md).


| Model         | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                 |
|-------------|-----------|-----------|------------------|------------------|----------|-----------|--------------------------------------------------------------------------------------|
| HRNet_W18_C | 0.7692    | 0.9339    | 7.40636          | 13.29752         | 4.14     | 21.29     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_pretrained.pdparams) |
| HRNet_W18_C_ssld | 0.81162    | 0.95804    | 7.40636          | 13.29752         | 4.14     | 21.29     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_ssld_pretrained.pdparams) |
| HRNet_W30_C | 0.7804    | 0.9402    | 9.57594          | 17.35485         | 16.23    | 37.71     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W30_C_pretrained.pdparams) |
| HRNet_W32_C | 0.7828    | 0.9424    | 9.49807          | 17.72921         | 17.86    | 41.23     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams) |
| HRNet_W40_C | 0.7877    | 0.9447    | 12.12202         | 25.68184         | 25.41    | 57.55     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W40_C_pretrained.pdparams) |
| HRNet_W44_C | 0.7900    | 0.9451    | 13.19858         | 32.25202         | 29.79    | 67.06     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W44_C_pretrained.pdparams) |
| HRNet_W48_C | 0.7895    | 0.9442    | 13.70761         | 34.43572         | 34.58    | 77.47     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_pretrained.pdparams) |
| HRNet_W48_C_ssld | 0.8363    | 0.9682    | 13.70761         | 34.43572         | 34.58    | 77.47     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_ssld_pretrained.pdparams) |
| HRNet_W64_C | 0.7930    | 0.9461    | 17.57527         | 47.9533          | 57.83    | 128.06    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W64_C_pretrained.pdparams) |
| SE_HRNet_W64_C_ssld | 0.8475    |  0.9726    |    31.69770      |     94.99546      | 57.83    | 128.97    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SE_HRNet_W64_C_ssld_pretrained.pdparams) |


<a name="Inception_series"></a>
### Inception series

Accuracy and inference time metrics of Inception series models are shown as follows. More detailed information can be refered to [Inception series tutorial](../en/models/Inception_en.md).


| Model                 | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|--------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|---------------------------------------------------------------------------------------------|
| GoogLeNet          | 0.7070    | 0.8966    | 1.88038               | 4.48882              | 2.88     | 8.46      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GoogLeNet_pretrained.pdparams)          |
| Xception41         | 0.7930    | 0.9453    | 4.96939               | 17.01361             | 16.74    | 22.69     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams)         |
| Xception41_deeplab | 0.7955    | 0.9438    | 5.33541               | 17.55938             | 18.16    | 26.73     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_deeplab_pretrained.pdparams) |
| Xception65         | 0.8100    | 0.9549    | 7.26158               | 25.88778             | 25.95    | 35.48     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams)         |
| Xception65_deeplab | 0.8032    | 0.9449    | 7.60208               | 26.03699             | 27.37    | 39.52     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_deeplab_pretrained.pdparams) |
| Xception71         | 0.8111    | 0.9545    | 8.72457               | 31.55549             | 31.77    | 37.28     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams)         |
| InceptionV3        | 0.7914    | 0.9459    | 6.64054              | 13.53630              | 11.46    | 23.83     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/InceptionV3_pretrained.pdparams)        |
| InceptionV4        | 0.8077    | 0.9526    | 12.99342              | 25.23416             | 24.57    | 42.68     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams)        |


<a name="EfficientNet_and_ResNeXt101_wsl_series"></a>
### EfficientNet and ResNeXt101_wsl series

Accuracy and inference time metrics of EfficientNet and ResNeXt101_wsl series models are shown as follows. More detailed information can be refered to [EfficientNet and ResNeXt101_wsl series tutorial](../en/models/EfficientNet_and_ResNeXt101_wsl_en.md).


| Model                       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                               |
|---------------------------|-----------|-----------|------------------|------------------|----------|-----------|----------------------------------------------------------------------------------------------------|
| ResNeXt101_<br>32x8d_wsl      | 0.8255    | 0.9674    | 18.52528         | 34.25319         | 29.14    | 78.44     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x8d_wsl_pretrained.pdparams)      |
| ResNeXt101_<br>32x16d_wsl     | 0.8424    | 0.9726    | 25.60395         | 71.88384         | 57.55    | 152.66    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x16d_wsl_pretrained.pdparams)     |
| ResNeXt101_<br>32x32d_wsl     | 0.8497    | 0.9759    | 54.87396         | 160.04337        | 115.17   | 303.11    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x32d_wsl_pretrained.pdparams)     |
| ResNeXt101_<br>32x48d_wsl     | 0.8537    | 0.9769    | 99.01698256      | 315.91261        | 173.58   | 456.2     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeXt101_32x48d_wsl_pretrained.pdparams)     |
| Fix_ResNeXt101_<br>32x48d_wsl | 0.8626    | 0.9797    | 160.0838242      | 595.99296        | 354.23   | 456.2     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Fix_ResNeXt101_32x48d_wsl_pretrained.pdparams) |
| EfficientNetB0            | 0.7738    | 0.9331    | 3.442            | 6.11476          | 0.72     | 5.1       | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_pretrained.pdparams)            |
| EfficientNetB1            | 0.7915    | 0.9441    | 5.3322           | 9.41795          | 1.27     | 7.52      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB1_pretrained.pdparams)            |
| EfficientNetB2            | 0.7985    | 0.9474    | 6.29351          | 10.95702         | 1.85     | 8.81      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB2_pretrained.pdparams)            |
| EfficientNetB3            | 0.8115    | 0.9541    | 7.67749          | 16.53288         | 3.43     | 11.84     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB3_pretrained.pdparams)            |
| EfficientNetB4            | 0.8285    | 0.9623    | 12.15894         | 30.94567         | 8.29     | 18.76     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB4_pretrained.pdparams)            |
| EfficientNetB5            | 0.8362    | 0.9672    | 20.48571         | 61.60252         | 19.51    | 29.61     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB5_pretrained.pdparams)            |
| EfficientNetB6            | 0.8400    | 0.9688    | 32.62402         | -                | 36.27    | 42        | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB6_pretrained.pdparams)            |
| EfficientNetB7            | 0.8430    | 0.9689    | 53.93823         | -                | 72.35    | 64.92     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB7_pretrained.pdparams)            |
| EfficientNetB0_<br>small      | 0.7580    | 0.9258    | 2.3076           | 4.71886          | 0.72     | 4.65      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB0_small_pretrained.pdparams)      |


<a name="ResNeSt_and_RegNet_series"></a>
### ResNeSt and RegNet series

Accuracy and inference time metrics of ResNeSt and RegNet series models are shown as follows. More detailed information can be refered to [ResNeSt and RegNet series tutorial](../en/models/ResNeSt_RegNet_en.md).


| Model                    | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                                 |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| ResNeSt50_<br>fast_1s1x64d | 0.8035    | 0.9528    | 3.45405                | 8.72680                | 8.68     | 26.3      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_fast_1s1x64d_pretrained.pdparams) |
| ResNeSt50              | 0.8083    | 0.9542    | 6.69042    | 8.01664                | 10.78    | 27.5      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNeSt50_pretrained.pdparams)              |
| RegNetX_4GF            | 0.785     | 0.9416    |    6.46478              |      11.19862           | 8        | 22.1      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams)            |


<a name="Transformer"></a>
### ViT_DeiT series

Accuracy and inference time metrics of ViT and DeiT series models are shown as follows. More detailed information can be refered to [Transformer series tutorial](../en/models/ViT_and_DeiT_en.md).


| Model                    | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address |
|------------------------|-----------|-----------|------------------|------------------|----------|------------------------|------------------------|
| ViT_small_<br/>patch16_224 | 0.7769  | 0.9342   | -                | -                |      |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams) |
| ViT_base_<br/>patch16_224 | 0.8195   | 0.9617   | -    | -                |     | 86 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams) |
| ViT_base_<br/>patch16_384 | 0.8414  | 0.9717   |    -              |      -           |         |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams) |
| ViT_base_<br/>patch32_384 | 0.8176   | 0.9613   | - | - |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams) |
| ViT_large_<br/>patch16_224 | 0.8323  | 0.9650   | - | - |  | 307 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams) |
| ViT_large_<br/>patch16_384 | 0.8513  | 0.9736  | - | - |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams) |
| ViT_large_<br/>patch32_384 | 0.8153   | 0.9608  | - | - |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams) |
|                            |           |           |                  |                  |          |           |                                                              |


| Model                    | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address |
|------------------------|-----------|-----------|------------------|------------------|----------|------------------------|------------------------|
| DeiT_tiny_<br>patch16_224 | 0.718 | 0.910 | -                | -                |      | 5 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams) |
| DeiT_small_<br>patch16_224 | 0.796 | 0.949 | -    | -                |     | 22 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>patch16_224 | 0.817 | 0.957 |    -              |      -           |         | 86 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>patch16_384 | 0.830 | 0.962 | - | - |  | 87 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_384_pretrained.pdparams) |
| DeiT_tiny_<br>distilled_patch16_224 | 0.741 | 0.918 | - | - |  | 6 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_distilled_patch16_224_pretrained.pdparams) |
| DeiT_small_<br>distilled_patch16_224 | 0.809 | 0.953 | - | - |  | 22 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>distilled_patch16_224 | 0.831 | 0.964 | - | - |  | 87 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams) |
| DeiT_base_<br>distilled_patch16_384 | 0.851 | 0.973 | - | - |  | 88 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams) |
|  |  |  |  |  |  |  |  |


<a name="RepVGG_series"></a>

### RepVGG

Accuracy and inference time metrics of RepVGG series models are shown as follows. More detailed information can be refered to [RepVGG series tutorial](../en/models/RepVGG_en.md).


| Model                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| RepVGG_A0   | 0.7131    | 0.9016    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A0_pretrained.pdparams) |
| RepVGG_A1   | 0.7380    | 0.9146    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A1_pretrained.pdparams) |
| RepVGG_A2   | 0.7571    | 0.9264    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_A2_pretrained.pdparams) |
| RepVGG_B0   | 0.7450    | 0.9213    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B0_pretrained.pdparams) |
| RepVGG_B1   | 0.7773    | 0.9385    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1_pretrained.pdparams) |
| RepVGG_B2   | 0.7813    | 0.9410    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2_pretrained.pdparams) |
| RepVGG_B1g2 | 0.7732    | 0.9359    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g2_pretrained.pdparams) |
| RepVGG_B1g4 | 0.7675    | 0.9335    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B1g4_pretrained.pdparams) |
| RepVGG_B2g4 | 0.7881    | 0.9448    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B2g4_pretrained.pdparams) |
| RepVGG_B3g4 | 0.7965    | 0.9485    |  |  |  |  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RepVGG_B3g4_pretrained.pdparams) |

<a name="MixNet"></a>

### MixNet

Accuracy and inference time metrics of MixNet series models are shown as follows. More detailed information can be refered to [MixNet series tutorial](../en/models/MixNet_en.md).

| Model    | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(M) | Params(M) | Download Address                                             |
| -------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| MixNet_S | 0.7628    | 0.9299    |                  |                  | 252.977  | 4.167     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_S_pretrained.pdparams) |
| MixNet_M | 0.7767    | 0.9364    |                  |                  | 357.119  | 5.065     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_M_pretrained.pdparams) |
| MixNet_L | 0.7860    | 0.9437    |                  |                  | 579.017  | 7.384     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MixNet_L_pretrained.pdparams) |

<a name="ReXNet"></a>

### ReXNet

Accuracy and inference time metrics of ReXNet series models are shown as follows. More detailed information can be refered to [ReXNet series tutorial](../en/models/ReXNet_en.md).

| Model      | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                             |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| ReXNet_1_0 | 0.7746    | 0.9370    |                  |                  | 0.415    | 4.838     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams) |
| ReXNet_1_3 | 0.7913    | 0.9464    |                  |                  | 0.683    | 7.611     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_3_pretrained.pdparams) |
| ReXNet_1_5 | 0.8006    | 0.9512    |                  |                  | 0.900    | 9.791     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_5_pretrained.pdparams) |
| ReXNet_2_0 | 0.8122    | 0.9536    |                  |                  | 1.561    | 16.449    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_2_0_pretrained.pdparams) |
| ReXNet_3_0 | 0.8209    | 0.9612    |                  |                  | 3.445    | 34.833    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_3_0_pretrained.pdparams) |

<a name="Others"></a>

<a name="SwinTransformer_series"></a>
### SwinTransformer

Accuracy and inference time metrics of SwinTransformer series models are shown as follows. More detailed information can be refered to[SwinTransformer series tutorial](../en/models/SwinTransformer_en.md).

| Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| SwinTransformer_tiny_patch4_window7_224    | 0.8069 | 0.9534 |                  |                  | 4.5  | 28   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_small_patch4_window7_224   | 0.8275 | 0.9613 |                  |                  | 8.7  | 50   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_small_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_base_patch4_window7_224    | 0.8300 | 0.9626 |                  |                  | 15.4 | 88   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_pretrained.pdparams) |
| SwinTransformer_base_patch4_window12_384   | 0.8439 | 0.9693 |                  |                  | 47.1 | 88   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window12_384_pretrained.pdparams) |
| SwinTransformer_base_patch4_window7_224<sup>[1]</sup>     | 0.8487 | 0.9746 |                  |                  | 15.4 | 88   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window7_224_22kto1k_pretrained.pdparams) |
| SwinTransformer_base_patch4_window12_384<sup>[1]</sup>    | 0.8642 | 0.9807 |                  |                  | 47.1 | 88   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_base_patch4_window12_384_22kto1k_pretrained.pdparams) |
| SwinTransformer_large_patch4_window7_224<sup>[1]</sup>    | 0.8596 | 0.9783 |                  |                  | 34.5 | 197  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window7_224_22kto1k_pretrained.pdparams) |
| SwinTransformer_large_patch4_window12_384<sup>[1]</sup>   | 0.8719 | 0.9823 |                  |                  | 103.9 | 197 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams) |

[1] Based on the pre-trained model of the ImageNet22k dataset, it is obtained by finetuning from the ImageNet1k data set.

<a name="LeViT_series"></a>
### LeViT

Accuracy and inference time metrics of LeViT series models are shown as follows. More detailed information can be refered to[LeViT series tutorial](../en/models/LeViT_en.md).

| Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(M) | Params(M) | Download Address                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| LeViT_128S | 0.7598    | 0.9269    |                  |                  | 305    | 7.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams) |
| LeViT_128 | 0.7810    | 0.9371    |                  |                  | 406    | 9.2     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams) |
| LeViT_192 | 0.7934    | 0.9446    |                  |                  | 658    | 11     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams) |
| LeViT_256 | 0.8085    | 0.9497    |                  |                  | 1120    | 19    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams) |
| LeViT_384 | 0.8191   | 0.9551    |                  |                  | 2353    | 39    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams) |

**Note**：The difference in accuracy from Reference is due to the difference in data preprocessing and the absence of distilled head as output.

<a name="Twins_series"></a>
### Twins

Accuracy and inference time metrics of Twins series models are shown as follows. More detailed information can be refered to[Twins series tutorial](../en/models/Twins_en.md).

| Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| pcpvt_small | 0.8082    | 0.9552    |                  |                  |3.7    | 24.1    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams) |
| pcpvt_base | 0.8242    | 0.9619    |                  |                  | 6.4    | 43.8    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams) |
| pcpvt_large | 0.8273    | 0.9650    |                  |                  | 9.5    | 60.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams) |
| alt_gvt_small | 0.8140    | 0.9546    |                  |                  |2.8   | 24   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams) |
| alt_gvt_base | 0.8294   | 0.9621    |                  |                  | 8.3   | 56   | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams) |
| alt_gvt_large | 0.8331   | 0.9642    |                  |                  | 14.8   | 99.2    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams) |

**Note**：The difference in accuracy from Reference is due to the difference in data preprocessing.

<a name="HarDNet_series"></a>
### HarDNet

Accuracy and inference time metrics of HarDNet series models are shown as follows. More detailed information can be refered to[HarDNet series tutorial](../en/models/HarDNet_en.md).


| Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                    |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| HarDNet39_ds | 0.7133    |0.8998    |                  |                  | 0.4   |  3.5    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet39_ds_pretrained.pdparams) |
| HarDNet68_ds |0.7362    | 0.9152   |                  |                  | 0.8   | 4.2 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_ds_pretrained.pdparams) |
| HarDNet68| 0.7546   | 0.9265   |                  |                  | 4.3   | 17.6    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet68_pretrained.pdparams) |
| HarDNet85 | 0.7744   | 0.9355   |                  |                  | 9.1   | 36.7  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HarDNet85_pretrained.pdparams) |

<a name="DLA_series"></a>
### DLA

Accuracy and inference time metrics of DLA series models are shown as follows. More detailed information can be refered to[DLA series tutorial](../en/models/DLA_en.md).

| Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| DLA102 | 0.7893    |0.9452    |                  |                  | 7.2   |  33.3    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams) |
| DLA102x2 |0.7885    | 0.9445  |                  |                  | 9.3   | 41.4 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x2_pretrained.pdparams) |
| DLA102x| 0.781   | 0.9400   |                  |                  | 5.9  | 26.4    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x_pretrained.pdparams) |
| DLA169 | 0.7809  | 0.9409   |                  |                  | 11.6  | 53.5  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA169_pretrained.pdparams) |
| DLA34 | 0.7603   | 0.9298    |                  |                  | 3.1   |  15.8    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams) |
| DLA46_c |0.6321   | 0.853   |                  |                  | 0.5   | 1.3 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA46_c_pretrained.pdparams) |
| DLA60 | 0.7610   | 0.9292   |                  |                  | 4.2   | 22.0    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60_pretrained.pdparams) |
| DLA60x_c | 0.6645   | 0.8754   |                  |                  | 0.6   | 1.3  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_c_pretrained.pdparams) |
| DLA60x | 0.7753  | 0.9378  |                  |                  | 3.5   | 17.4  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_pretrained.pdparams) |

<a name="RedNet_series"></a>
### RedNet

Accuracy and inference time metrics of RedNet series models are shown as follows. More detailed information can be refered to[RedNet series tutorial](../en/models/RedNet_en.md).

| Model      | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                     |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| RedNet26 | 0.7595   |0.9319  |                  |                  | 1.7   |  9.2    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet26_pretrained.pdparams) |
| RedNet38 |0.7747  | 0.9356  |                  |                  | 2.2   | 12.4 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet38_pretrained.pdparams) |
| RedNet50| 0.7833  | 0.9417   |                  |                  | 2.7   | 15.5    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet50_pretrained.pdparams) |
| RedNet101 | 0.7894  | 0.9436   |                  |                  | 4.7  | 25.7 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet101_pretrained.pdparams) |
| RedNet152 | 0.7917  | 0.9440   |                  |                  | 6.8  | 34.0  | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RedNet152_pretrained.pdparams) |

<a name="TNT_series"></a>
### TNT

Accuracy and inference time metrics of TNT series models are shown as follows. More detailed information can be refered to[TNT series tutorial](../en/models/TNT_en.md).

|  Model       | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                      |
| ---------- | --------- | --------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------ |
| TNT_small | 0.8121   |0.9563  |                  |                  | 5.2   |  23.8    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams) |               |  

**Note**：The `mean` and `std` in `NormalizeImage` in the data preprocessing part of the TNT model are both 0.5.

### Others

Accuracy and inference time metrics of AlexNet, SqueezeNet series, VGG series and DarkNet53 models are shown as follows. More detailed information can be refered to [Others](../en/models/Others_en.md).


| Model                     | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address |
|------------------------|-----------|-----------|------------------|------------------|----------|-----------|------------------------------------------------------------------------------------------------------|
| AlexNet       | 0.567 | 0.792 | 1.44993         | 2.46696         | 1.370 | 61.090 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams) |
| SqueezeNet1_0 | 0.596 | 0.817 | 0.96736 | 2.53221         | 1.550 | 1.240 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_0_pretrained.pdparams) |
| SqueezeNet1_1 | 0.601 | 0.819 | 0.76032       | 1.877      | 0.690   | 1.230 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SqueezeNet1_1_pretrained.pdparams) |
| VGG11 | 0.693 | 0.891 | 3.90412 | 9.51147 | 15.090 | 132.850 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG11_pretrained.pdparams) |
| VGG13 | 0.700 | 0.894 | 4.64684 | 12.61558 | 22.480 | 133.030 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG13_pretrained.pdparams) |
| VGG16 | 0.720 | 0.907 | 5.61769 | 16.40064 | 30.810 | 138.340 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG16_pretrained.pdparams) |
| VGG19 | 0.726 | 0.909 | 6.65221 | 20.4334 | 39.130 | 143.650 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VGG19_pretrained.pdparams) |
| DarkNet53 | 0.780 | 0.941 | 4.10829 | 12.1714 | 18.580 | 41.600 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams) |
