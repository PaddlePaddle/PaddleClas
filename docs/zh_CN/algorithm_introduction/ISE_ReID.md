# ISE
---
## 目录

- [1. 介绍](#1)
- [2. 在Market1501和MSMT17上的结果](#2)
- [3. 测试](#3)
- [4. 引用](#4)

<a name='1'></a>
## 1. 介绍

ISE (Implicit Sample Extension)是一种简单、高效、有效的无监督行人再识别学习算法。ISE在聚类蔟边界周围生成样本，我们称之为支持样本。ISE的样本生成过程依赖于两个关键机制，即渐进线性插值策略（progressive linear interpolation）和标签保留的损失函数（label-preserving loss function）。ISE生成的支持样本提供了额外补充信息，可以很好地处理“子类和混合”的聚类错误。ISE在Market1501和MSMT17数据集上取得了优于其他无监督方法的性能。

> [**Implicit Sample Extension for Unsupervised Person Re-Identification**](https://arxiv.org/abs/2204.06892v1)<br>
> Xinyu Zhang, Dongdong Li, Zhigang Wang, Jian Wang, Errui Ding, Javen Qinfeng Shi, Zhaoxiang Zhang, Jingdong Wang<br>
> CVPR2022

![image](../../images/ISE_ReID/ISE_pipeline.png)


<a name='2'></a>
## 2. 在Market1501和MSMT17上的结果

在Market1501和MSMT17上的主要结果。“PIL”表示渐进线性插值策略。“LP”表示标签保留的损失函数。

| 方法 | Market1501 | 下载链接 | MSMT17 | 下载链接 |
| --- | -- | -- | -- | - |
| Baseline | 82.5 (92.5) | - | 30.1 (58.6) | - |
| ISE (+PIL) | 83.9 (93.9) | - | 33.5 (63.9) | - |
| ISE (+LP)  | 83.6 (92.7) | - | 31.4 (59.9) | - |
| ISE (Ours) (+PIL+LP) | **84.7 (94.0)** | [ISE_M](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ISE_M_model.pdparams) | **35.0 (64.7)** | [ISE_MS](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ISE_MS_model.pdparams) |


<a name="3"></a>
## 3. 测试
我们很快会提供训练代码，首先我们提供了测试代码和模型。

**测试：** 可简使用如下脚本进行模型评估。

```
python tools/eval.py -c ./ppcls/configs/Person/ResNet50_UReID_infer.yaml
```
**步骤：**
1. 首先下载模型，并放入：```./pd_model_trace/ISE/```。
2. 改变```./ppcls/configs/Person/ResNet50_UReID_infer.yaml```中的数据集名称。
3. 运行上述脚本。


<a name="4"></a>
## 4. 引用

如果ISE在您的研究中有启发，请考虑引用我们的论文:

```
@inproceedings{zhang2022Implicit,
  title={Implicit Sample Extension for Unsupervised Person Re-Identification},
  author={Xinyu Zhang, Dongdong Li, Zhigang Wang, Jian Wang, Errui Ding, Javen Qinfeng Shi, Zhaoxiang Zhang, Jingdong Wang},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
