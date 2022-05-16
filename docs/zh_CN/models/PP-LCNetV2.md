# PP-LCNetV2 系列

---

## 概述

PP-LCNetV2 是在 [PP-LCNet 系列模型](./PP-LCNet.md)的基础上，所提出的针对 Intel CPU 硬件平台设计的计算机视觉骨干网络，该模型更为

在不使用额外数据的前提下，PPLCNetV2_base 模型在图像分类 ImageNet 数据集上能够取得超过 77% 的 Top1 Acc，同时在 Intel CPU 平台仅有 4.4 ms 以下的延迟，如下表所示，其中延时测试基于 Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz 硬件平台，OpenVINO 2021.4.2推理平台。

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|-------|-----------|----------|---------------|---------------|-------------|
| PPLCNetV2_base  | 6.6 | 604  | 77.04 | 93.27 | 4.32 |

关于 PP-LCNetV2 系列模型的更多信息，敬请关注。
