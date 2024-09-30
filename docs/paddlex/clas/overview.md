## 1.简介

[PaddleX](https://github.com/PaddlePaddle/PaddleX)，依托于PaddleClas的先进技术，构建了图像分类和图像检索领域的**低代码全流程**开发范式。通过低代码开发，可实现简单且高效的模型使用、组合与定制。这将显著**减少模型开发的时间消耗**，**降低其开发难度**，大大加快模型在行业中的应用和推广速度。特色如下：

* 🎨 模型丰富一键调用：将通用图像分类、图像多标签分类、通用图像识别、人脸识别涉及的**92个模型**整合为4条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持目标检测、图像分割、文本图像智能分析、通用OCR、时序预测等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合使用**。

* 🚀提高效率降低门槛：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能部署、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。
  
## 2.能力支持

PaddleX中图像分类和图像检索的4条产线均支持本地**快速推理**，部分产线支持**在线体验**，您可以快速体验各个产线的预训练模型效果，如果您对产线的预训练模型效果满意，可以直接对产线进行[高性能部署](/docs_new/pipeline_deploy/high_performance_deploy.md)/[服务化部署](/docs_new/pipeline_deploy/service_deploy.md)/[端侧部署](/docs_new/pipeline_deploy/lite_deploy.md)，如果不满意，您也可以使用产线的**二次开发**能力，提升效果。完整的产线开发流程请参考[PaddleX产线使用概览](/docs_new/pipeline_usage/pipeline_develop_guide.md)或各产线使用[教程](#-文档)。



此外，PaddleX为开发者提供了基于[云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)的全流程开发工具, 详细请参考[教程《零门槛开发产业级AI模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)


<table >
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
<tr>
        <td>通用语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>通用图像识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>人脸识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
        <tr>
        <td>行人属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td>车辆属性识别</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>

    
</table>


> ❗注：以上功能均基于GPU/CPU实现。PaddleX还可在昆仑、昇腾、寒武纪和海光等主流硬件上进行快速推理和二次开发。下表详细列出了模型产线的支持情况，具体支持的模型列表请参阅[模型列表(MLU)](./docs_new/support_list/model_list_mlu.md)/[模型列表(NPU)](./docs_new/support_list/model_list_npu.md)/[模型列表(XPU)](./docs_new/support_list/model_list_xpu.md)/[模型列表DCU](./docs_new/support_list/model_list_dcu.md)。我们正在适配更多的模型，并在主流硬件上推动高性能和服务化部署的实施。

<details>
  <summary>👉 国产化硬件能力支持</summary>

<table>
  <tr>
    <th>产线名称</th>
    <th>昇腾 910B</th>
    <th>昆仑 R200/R300</th>
    <th>寒武纪 MLU370X8</th>
    <th>海光 Z100</th>
  </tr>
  <tr>
    <td>通用OCR</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>表格识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用目标检测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用实例分割</td>
    <td>✅</td>
    <td>🚧</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用图像分类</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>通用语义分割</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>时序预测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>时序异常检测</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>时序分类</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
</table>
</details>

## 3.模型列表

## 语义分割模块
|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|Deeplabv3_Plus-R50 |80.36|61.0531|1513.58|94.9 M|
|Deeplabv3_Plus-R101|81.10|100.026|2460.71|162.5 M|
|Deeplabv3-R50|79.90|82.2631|1735.83|138.3 M|
|Deeplabv3-R101|80.85|121.492|2685.51|205.9 M|
|OCRNet_HRNet-W18|80.67|48.2335|906.385|43.1 M|
|OCRNet_HRNet-W48|82.15|78.9976|2226.95|249.8 M|
|PP-LiteSeg-T|73.10|7.6827|138.683|28.5 M|
|PP-LiteSeg-B|75.25|-|-|47.0 M|
|SegFormer-B0 (slice)|76.73|11.1946|268.929|13.2 M|
|SegFormer-B1 (slice)|78.35|17.9998|403.393|48.5 M|
|SegFormer-B2 (slice)|81.60|48.0371|1248.52|96.9 M|
|SegFormer-B3 (slice)|82.47|64.341|1666.35|167.3 M|
|SegFormer-B4 (slice)|82.38|82.4336|1995.42|226.7 M|
|SegFormer-B5 (slice)|82.58|97.3717|2420.19|229.7 M|

**注：以上精度指标为 **[Cityscapes](https://www.cityscapes-dataset.com/)** 数据集 mloU。**

|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|SeaFormer_base(slice)|40.92|24.4073|397.574|30.8 M|
|SeaFormer_large (slice)|43.66|27.8123|550.464|49.8 M|
|SeaFormer_small (slice)|38.73|19.2295|358.343|14.3 M|
|SeaFormer_tiny (slice)|34.58|13.9496|330.132|6.1M |

**注：以上精度指标为 **[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)** 数据集, slice 表示对输入图像进行了切图操作。**

## 图像异常检测模块
|模型名称|Avg（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|STFPM|96.2|-|-|21.5 M|

**注：以上精度指标为 **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** 验证集 平均异常分数。**

### 行人属性模块
|模型名称|mA（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|PP-LCNet_x1_0_pedestrian_attribute|92.2|3.84845|9.23735|6.7M  |

**注：以上精度指标为 PaddleX 内部自建数据集mA。**

### 车辆属性模块
|模型名称|mA（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|PP-LCNet_x1_0_vehicle_attribute|91.7|3.84845|9.23735|6.7 M|

**注：以上精度指标为 VeRi 数据集 mA。**


>**注：以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**