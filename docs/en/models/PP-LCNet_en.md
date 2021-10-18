# PP-LCNet Series

## Abstract

In the field of computer vision, the quality of backbone network determines the outcome of the whole vision task. In previous studies, researchers generally focus on the optimization of FLOPs or Params, but inference speed actually serves as an importance indicator of model quality in real-world scenarios. Nevertheless, it is difficult to balance inference speed and accuracy. In view of various CPU-based applications in industry, we are now working to raise the adaptability of the backbone network to Intel CPU, so as to obtain a faster and more accurate lightweight backbone network. At the same time, the performance of downstream vision tasks such as object detection and semantic segmentation are also improved.

## Introduction

Recent years witnessed the emergence of many lightweight backbone networks. In past two years, in particular, there were abundant networks searched by NAS that either enjoy advantages on FLOPs or Params, or have an edge in terms of inference speed on ARM devices. However, few of them dedicated to specified optimization of Intel CPU, resulting their imperfect inference speed on the intel CPU side. Based on this, we specially design the backbone network PP-LCNet for Intel CPU devices with its acceleration library MKLDNN. Compared with other lightweight SOTA models, this backbone network can further improve the performance of the model without increasing the inference time, significantly outperforming the existing SOTA models. A comparison chart with other models is shown below.
<div align=center><img src="../../images/PP-LCNet/PP-LCNet-Acc.png" width="500" height="400"/></div>

## Method

The overall structure of the network is shown in the figure below.
<div align=center><img src="../../images/PP-LCNet/PP-LCNet.png" width="700" height="400"/></div>

Build on extensive experiments, we found that many seemingly less time-consuming operations will increase the latency  on Intel CPU-based devices, especially when the MKLDNN acceleration library is enabled. Therefore, we finally chose a block with the leanest possible structure and the fastest possible speed to form our BaseNet (similar to MobileNetV1). Based on BaseNet, we summarized four strategies that can improve the accuracy of the model without increasing the latency, and we combined these four strategies to form PP-LCNet. Each of these four strategies is introduced as below:

### Better Activation Function

Since the adoption of ReLU activation function by convolutional neural network, the network performance has been improved substantially, and variants of the ReLU activation function have appeared in recent years, such as Leaky-ReLU, P-ReLU, ELU, etc. In 2017, Google Brain searched to obtain the swish activation function, which performs well on lightweight networks. In 2019, the authors of MobileNetV3 further optimized this activation function to H-Swish, which removes the exponential operation, leading to faster speed and an almost unaffected network accuracy. After many experiments, we also recognized its excellent performance on lightweight networks. Therefore, this activation function is adopted in PP-LCNet.

### SE Modules at Appropriate Positions

The SE module is a channel attention mechanism proposed by SENet, which can effectively improve the accuracy of the model. However, on the Intel CPU side, the module also presents a large latency, leaving us the task of balancing accuracy and speed. The search of the location of the SE module in NAS search-based networks such as MobileNetV3 brings no general conclusions, but we found through our experiments that the closer the SE module is to the tail of the network the greater the improvement in model accuracy. The following table also shows some of our experimental results：

| SE Location       | Top-1 Acc(\%) | Latency(ms) |
|-------------------|---------------|-------------|
| 1100000000000     | 61.73           | 2.06         |
| 0000001100000     | 62.17           | 2.03         |
| <b>0000000000011<b>     | <b>63.14<b>           | <b>2.05<b>         |
| 1111111111111     | 64.27           | 3.80         |

The option in the third row of the table was chosen for the location of the SE module in PP-LCNet.

### Larger Convolution Kernels

In the paper of MixNet, the author analyzes the effect of convolutional kernel size on model performance and concludes that larger convolutional kernels within a certain range can improve the performance of the model, but beyond this range will be detrimental to the model’s performance. So the author forms MixConv with split-concat paradigm combined, which can improve the performance of the model but is not conducive to inference. We experimentally summarize the role of some larger convolutional kernels at different positions that are similar to those of the SE module, and find that larger convolutional kernels display more prominent roles in the middle and tail of the network. The following table shows the effect of the position of the 5x5 convolutional kernels on the accuracy：

| SE Location       | Top-1 Acc(\%) | Latency(ms) |
|-------------------|---------------|-------------|
| 1111111111111     | 63.22           | 2.08         |
| 1111111000000     | 62.70           | 2.07        |
| <b>0000001111111<b>     | <b>63.14<b>           | <b>2.05<b>         |


Experiments show that a larger convolutional kernel placed at the middle and tail of the network can achieve the same accuracy as placed at all positions, coupled with faster inference. The option in the third row of the table was the final choice of PP-LCNet.

### Larger dimensional 1 × 1 conv layer after GAP

Since the introduction of GoogLeNet, GAP (Global-Average-Pooling) is often directly followed by a classification layer, which fails to result in further integration and processing of features extracted after GAP in the lightweight network. If a larger 1x1 convolutional layer (equivalent to the FC layer) is used after GAP, the extracted features, instead of directly passing through the classification layer, will first be integrated, and then classified. This can greatly improve the accuracy rate without affecting the inference speed of the model. The above four improvements were made to BaseNet to obtain PP-LCNet. The following table further illustrates the impact of each scheme on the results：

| Activation | SE-block | Large-kernal | last-1x1-conv | Top-1 Acc(\%) | Latency(ms) |
|------------|----------|--------------|---------------|---------------|-------------|
| 0       | 1       | 1               | 1                | 61.93 | 1.94 |
| 1       | 0       | 1               | 1                | 62.51 | 1.87 |
| 1       | 1       | 0               | 1                | 62.44 | 2.01 |
| 1       | 1       | 1               | 0                | 59.91 | 1.85 |
| <b>1<b>       | <b>1<b>       | <b>1<b>               | <b>1<b>                | <b>63.14<b> | <b>2.05<b> |


## Experiments

### Image Classification

For image classification, ImageNet dataset is adopted. Compared with the current mainstream lightweight network, PP-LCNet can obtain faster inference speed with the same accuracy. When using Baidu’s self-developed SSLD distillation strategy, the accuracy is further improved, with the Top-1 Acc of ImageNet exceeding 80% at an inference speed of about 5ms on the Intel CPU side.

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) | 
|-------|-----------|----------|---------------|---------------|-------------|
| PP-LCNet-0.25x  | 1.5 | 18  | 51.86 | 75.65 | 1.74 |
| PP-LCNet-0.35x  | 1.6 | 29  | 58.09 | 80.83 | 1.92 |
| PP-LCNet-0.5x   | 1.9 | 47  | 63.14 | 84.66 | 2.05 |
| PP-LCNet-0.75x  | 2.4 | 99  | 68.18 | 88.30 | 2.29 |
| PP-LCNet-1x     | 3.0 | 161 | 71.32 | 90.03 | 2.46 |
| PP-LCNet-1.5x   | 4.5 | 342 | 73.71 | 91.53 | 3.19 |
| PP-LCNet-2x     | 6.5 | 590 | 75.18 | 92.27 | 4.27 |
| PP-LCNet-2.5x   | 9.0 | 906 | 76.60 | 93.00 | 5.39 |
| PP-LCNet-0.25x\* | 1.9 | 47  | 66.10 | 86.46 | 2.05 |
| PP-LCNet-0.25x\* | 3.0 | 161 | 74.39 | 92.09 | 2.46 |
| PP-LCNet-0.25x\* | 9.0 | 906 | 80.82 | 95.33 | 5.39 |
    
\* denotes the model after using SSLD distillation.

Performance comparison with other lightweight networks:

| Model | Params(M) | FLOPs(M) | Top-1 Acc(\%) | Top-5 Acc(\%) | Latency(ms) |
|-------|-----------|----------|---------------|---------------|-------------|
| MobileNetV2-0.25x  | 1.5 | 34  | 53.21 | 76.52 | 2.47 |
| MobileNetV3-small-0.35x  | 1.7 | 15  | 53.03 | 76.37 | 3.02 |
| ShuffleNetV2-0.33x  | 0.6 | 24  | 53.73 | 77.05 | 4.30 |
| <b>PP-LCNet-0.25x<b>  | <b>1.5<b> | <b>18<b>  | <b>51.86<b> | <b>75.65<b> | <b>1.74<b> |
| MobileNetV2-0.5x  | 2.0 | 99  | 65.03 | 85.72 | 2.85 |
| MobileNetV3-large-0.35x  | 2.1 | 41  | 64.32 | 85.46 | 3.68 |
| ShuffleNetV2-0.5x  | 1.4 | 43  | 60.32 | 82.26 | 4.65 |
| <b>PP-LCNet-0.5x<b>   | <b>1.9<b> | <b>47<b>  | <b>63.14<b> | <b>84.66<b> | <b>2.05<b> |
| MobileNetV1-1x  | 4.3 | 578  | 70.99 | 89.68 | 3.38 |
| MobileNetV2-1x  | 3.5 | 327  | 72.15 | 90.65 | 4.26 |
| MobileNetV3-small-1.25x  | 3.6 | 100  | 70.67 | 89.51 | 3.95 |
| <b>PP-LCNet-1x<b>     |<b> 3.0<b> | <b>161<b> | <b>71.32<b> | <b>90.03<b> | <b>2.46<b> |


### Object Detection

For object detection, we adopt Baidu’s self-developed PicoDet, which focuses on lightweight object detection scenarios. The following table shows the comparison between the results of PP-LCNet and MobileNetV3 on the COCO dataset. PP-LCNet has an obvious advantage in both accuracy and speed.

| Backbone | mAP(%) | Latency(ms) |
|-------|-----------|----------|
MobileNetV3-large-0.35x | 19.2 | 8.1 |
<b>PP-LCNet-0.5x<b> | <b>20.3<b> | <b>6.0<b> |
MobileNetV3-large-0.75x | 25.8 | 11.1 |
<b>PP-LCNet-1x<b> | <b>26.9<b> | <b>7.9<b> | 


### Semantic Segmentation

For semantic segmentation, DeeplabV3+ is adopted. The following table presents the comparison between PP-LCNet and MobileNetV3 on the Cityscapes dataset, and PP-LCNet also stands out in terms of accuracy and speed.

| Backbone | mIoU(%) | Latency(ms) |
|-------|-----------|----------|
MobileNetV3-large-0.5x | 55.42 | 135 |
<b>PP-LCNet-0.5x<b> | <b>58.36<b> | <b>82<b> |
MobileNetV3-large-0.75x | 64.53 | 151 |
<b>PP-LCNet-1x<b> | <b>66.03<b> | <b>96<b> |

## Conclusion

Rather than holding on to perfect FLOPs and Params as academics do, PP-LCNet focuses on analyzing how to add Intel CPU-friendly modules to improve the performance of the model, which can better balance accuracy and inference time. The experimental conclusions therein are available to other researchers in network structure design, while providing NAS search researchers with a smaller search space and general conclusions. The finished PP-LCNet can also be better accepted and applied in industry.

## Reference

Reference to cite when you use PP-LCNet in a paper:
```
@misc{cui2021pplcnet,
      title={PP-LCNet: A Lightweight CPU Convolutional Neural Network}, 
      author={Cheng Cui and Tingquan Gao and Shengyu Wei and Yuning Du and Ruoyu Guo and Shuilong Dong and Bin Lu and Ying Zhou and Xueying Lv and Qiwen Liu and Xiaoguang Hu and Dianhai Yu and Yanjun Ma},
      year={2021},
      eprint={2109.15099},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
