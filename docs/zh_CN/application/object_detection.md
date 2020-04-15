# 通用目标检测

## 服务器端实用目标检测方案

### 简介

* 近年来，学术界和工业界广泛关注图像中目标检测任务。基于SSLD蒸馏方案训练得到的ResNet50_vd预训练模型(ImageNet1k验证集上Top1 Acc为82.39%)，结合PaddleDetection中的丰富算子，飞桨提供了一种面向服务器端实用的目标检测方案PSS-DET(Practical Server Side Detection)。基于COCO2017目标检测数据集，V100单卡预测速度为为61FPS时，COCO mAP可达41.6%；预测速度为20FPS时，COCO mAP可达47.8%。

### 消融实验

* 我们以标准的Faster RCNN ResNet50_vd FPN为例，下表给出了PSS-DET不同的模块的速度与精度收益。

| Trick | Train scale | Test scale |  COCO mAP | Infer speed/FPS |
|- |:-: |:-: | :-: | :-: |
| `baseline` | 640x640 | 640x640 | 36.4% | 43.589 |
| +`test proposal=pre/post topk 500/300` | 640x640 | 640x640 | 36.2% | 52.512 |
| +`fpn channel=64` | 640x640 | 640x640 | 35.1% | 67.450 |
| +`ssld pretrain` | 640x640 | 640x640 | 36.3% | 67.450 |
| +`ciou loss` | 640x640 | 640x640 | 37.1% | 67.450 |
| +`DCNv2` | 640x640 | 640x640 | 39.4% | 60.345 |
| +`3x, multi-scale training` | 640x640 | 640x640 | 41.0% | 60.345 |
| +`auto augment` | 640x640 | 640x640 | 41.4% | 60.345 |
| +`libra sampling` | 640x640 | 640x640 | 41.6% | 60.345 |


基于该实验结论，我们结合Cascade RCNN，使用更大的训练与评估尺度(1000x1500)，最终在单卡V100上速度为20FPS，COCO mAP达47.8%。下图给出了目前类似速度的目标检测方法的速度与精度指标。


![pssdet](../../images/det/pssdet.png)

**注意**
> 这里为了更方便地对比，我们将V100的预测耗时乘以1.2倍，近似转化为Titan V的预测耗时。

更加详细的代码、配置与预训练模型的地址可以参考[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det)。


## 移动端实用目标检测方案

* 目前正在更新中，敬请期待！