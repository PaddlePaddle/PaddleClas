# General object detection

## Practical Server-side detection method base on RCNN

### Introduction


* In recent years, object detection tasks have attracted widespread attention. [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) open-sourced the ResNet50_vd_SSLD pretrained model based on ImageNet(Top1 Acc 82.4%). And based on the pretrained model, PaddleDetection provided the PSS-DET (Practical Server-side detection) with the help of the rich operators in PaddleDetection. The inference speed can reach 61FPS on single V100 GPU when COCO mAP is 41.6%, and 20FPS when COCO mAP is 47.8%.

* We take the standard `Faster RCNN ResNet50_vd FPN` as an example. The following table shows ablation study of PSS-DET.

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


Based on the ablation experiments, Cascade RCNN and larger inference scale(1000x1500) are used for better performance. The final COCO mAP is 47.8%
and the following figure shows `mAP-Speed` curves for some common detectors.


![pssdet](../../images/det/pssdet.png)


**Note**
> For fair comparison, inference time for PSS-DET models on V100 GPU is transformed to Titan V GPU by multiplying by 1.2 times.

For more detailed information, you can refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_server_side_det).


## Practical Mobile-side detection method base on RCNN

* This part is comming soon!
