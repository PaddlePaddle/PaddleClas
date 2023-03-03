# MobileNet 前端部署示例

本节介绍部署PaddleClas的图像分类mobilenet模型在浏览器中运行，以及@paddle-js-models/mobilenet npm包中的js接口。

## 1. 前端部署图像分类模型

图像分类模型web demo使用[**参考文档**](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/application/js/web_demo)

## 2. MobileNet js接口

```
import * as mobilenet from "@paddle-js-models/mobilenet";
# mobilenet模型加载和初始化
await mobilenet.load()
# mobilenet模型执行预测，并获得分类的类别
const res = await mobilenet.classify(img);
console.log(res);
```

**load()函数参数**

> * **Config**(dict): 图像分类模型配置参数，默认值为 {Path: 'https://paddlejs.bj.bcebos.com/models/fuse/mobilenet/mobileNetV2_fuse_activation/model.json', fill: '#fff', mean: [0.485, 0.456, 0.406],std: [0.229, 0.224, 0.225]}; 其中，modelPath为js模型路径，fill 为图像预处理padding的值，mean和std分别为预处理的均值和标准差。


**classify()函数参数**
> * **img**(HTMLImageElement): 输入图像参数，类型为HTMLImageElement。
