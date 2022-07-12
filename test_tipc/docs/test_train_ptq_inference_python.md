# Linux GPU/CPU KL离线量化推理测试

Linux GPU/CPU KL离线量化推理测试的主程序为`test_ptq_inference_python.sh`，可以测试基于Python的模型KL离线量化推理等基本功能。

## 1. 测试结论汇总

- KL离线量化：

|    算法名称     |                模型名称                 |   CPU   |
| :-------------: | :-------------------------------------: | :----------: |
|   MobileNetV3   |         MobileNetV3_large_x1_0          | KL离线量化 |
|    PP-ShiTu     |     GeneralRecognition_PPLCNet_x2_5     | KL离线量化 |
|     PPHGNet     |              PPHGNet_small              | KL离线量化 |
|     PPHGNet     |              PPHGNet_tiny               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_25              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_35              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_5               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_75              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x1_0               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x1_5               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x2_0               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x2_5               | KL离线量化 |
|    PPLCNetV2    |             PPLCNetV2_base              | KL离线量化 |
|     ResNet      |                ResNet50                 | KL离线量化 |
|     ResNet      |               ResNet50_vd               | KL离线量化 |
| SwinTransformer | SwinTransformer_tiny_patch4_window7_224 | KL离线量化 |

- 推理相关：

|    算法名称     |                模型名称                 |   CPU   |
| :-------------: | :-------------------------------------: | :----------: |
|   MobileNetV3   |         MobileNetV3_large_x1_0          | KL离线量化 |
|    PP-ShiTu     |     GeneralRecognition_PPLCNet_x2_5     | KL离线量化 |
|     PPHGNet     |              PPHGNet_small              | KL离线量化 |
|     PPHGNet     |              PPHGNet_tiny               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_25              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_35              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_5               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x0_75              | KL离线量化 |
|     PPLCNet     |              PPLCNet_x1_0               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x1_5               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x2_0               | KL离线量化 |
|     PPLCNet     |              PPLCNet_x2_5               | KL离线量化 |
|    PPLCNetV2    |             PPLCNetV2_base              | KL离线量化 |
|     ResNet      |                ResNet50                 | KL离线量化 |
|     ResNet      |               ResNet50_vd               | KL离线量化 |
| SwinTransformer | SwinTransformer_tiny_patch4_window7_224 | KL离线量化 |


## 2. 测试流程

一下测试流程以 MobileNetV3_large_x1_0 模型为例。

### 2.1 准备环境

- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
  ```bash
  # 需要安装2.2及以上版本的Paddle
  # 安装GPU版本的Paddle
  python3.7 -m pip install paddlepaddle-gpu==2.2.0
  # 安装CPU版本的Paddle
  python3.7 -m pip install paddlepaddle==2.2.0
  ```

- 安装PaddleSlim
  ```bash
  python3.7 -m pip install paddleslim==2.2.0
  ```

- 安装依赖
  ```bash
  python3.7 -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 安装AutoLog（规范化日志输出工具）
  ```bash
  python3.7 -m pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
  ```

### 2.2 准备数据和模型

```bash
bash test_tipc/prepare.sh test_tipc/config/MobileNetV3/MobileNetV3_large_x1_0_train_ptq_infer_python.txt whole_infer
```

离线量化的操作流程，可参考[文档](../../deploy/slim/README.md)。

### 2.3 功能测试

以`MobileNetV3_large_x1_0`的`Linux GPU/CPU KL离线量化训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_ptq_inference_python.sh test_tipc/config/MobileNetV3/MobileNetV3_large_x1_0_train_ptq_infer_python.txt whole_infer
```

输出结果如下，表示命令运行成功。

```log
Run successfully with command - MobileNetV3_large_x1_0 - python3.7 deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/MobileNetV3/MobileNetV3_large_x1_0.yaml -o Global.save_inference_dir=./MobileNetV3_large_x1_0_infer!
Run successfully with command - MobileNetV3_large_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.use_fp16=False -o Global.inference_model_dir=.././MobileNetV3_large_x1_0_infer//quant_post_static_model -o Global.batch_size=1 -o Global.infer_imgs=../deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg -o Global.benchmark=False > .././test_tipc/output/MobileNetV3_large_x1_0/whole_infer/infer_gpu_usetrt_False_precision_False_batchsize_1.log 2>&1 !
Run successfully with command - MobileNetV3_large_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=False -o Global.enable_mkldnn=False -o Global.cpu_num_threads=1 -o Global.inference_model_dir=.././MobileNetV3_large_x1_0_infer//quant_post_static_model -o Global.batch_size=1 -o Global.infer_imgs=../deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg -o Global.benchmark=False   > .././test_tipc/output/MobileNetV3_large_x1_0/whole_infer/infer_cpu_usemkldnn_False_threads_1_batchsize_1.log 2>&1 !
```
同时，测试过程中的日志保存在`PaddleClas/test_tipc/output/MobileNetV3_large_x1_0/whole_infer`下。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
