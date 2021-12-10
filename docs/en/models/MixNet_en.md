# MixNet series
---
## Catalogue

* [1. Overview](#1)
* [2. Accuracy, FLOPs and Parameters](#2)

<a name='1'></a>
## 1. Overview

MixNet is a lightweight network proposed by Google. The main idea of MixNet is to explore the combination of different size of kernels. The author found that the current network has the following two problems:

- Small convolution kernel has small receptive field and few parameters, but the accuracy is not high.
- The larger convolution kernel has larger receptive field and higher accuracy, but the parameters also increase a lot .

 In order to solve the above two problems, MDConv(mixed depthwise convolution) is proposed.  In this method, different size of kernels  are mixed in a convolution operation block. And based on AutoML,  a series of networks called MixNets are proposed, which have achieved good results on Imagenet. [paper](https://arxiv.org/pdf/1907.09595.pdf)

<a name='2'></a>
## 2. Accuracy, FLOPs and Parameters

|  Models  | Top1  | Top5  | Reference<br>top1 | FLOPs<br>(M) | Params<br/>(G |
| :------: | :---: | :---: | :---------------: | :----------: | ------------- |
| MixNet_S | 76.28 | 92.99 |       75.8        |   252.977    | 4.167         |
| MixNet_M | 77.67 | 93.64 |       77.0        |   357.119    | 5.065         |
| MixNet_L | 78.60 | 94.37 |       78.9        |   579.017    | 7.384         |

Inference speed and other information are coming soon.
