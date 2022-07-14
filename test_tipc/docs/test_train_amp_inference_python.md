# Linux GPU/CPU 混合精度训练推理测试

Linux GPU/CPU 混合精度训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型混合精度(默认O2)训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

|    算法名称     |                模型名称                 |   单机单卡   |   单机多卡   |
| :-------------: | :-------------------------------------: | :----------: | :----------: |
|   MobileNetV3   |         MobileNetV3_large_x1_0          | 混合精度训练 | 混合精度训练 |
|    PP-ShiTu     |     GeneralRecognition_PPLCNet_x2_5     | 混合精度训练 | 混合精度训练 |
|     PPHGNet     |              PPHGNet_small              | 混合精度训练 | 混合精度训练 |
|     PPHGNet     |              PPHGNet_tiny               | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x0_25              | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x0_35              | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x0_5               | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x0_75              | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x1_0               | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x1_5               | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x2_0               | 混合精度训练 | 混合精度训练 |
|     PPLCNet     |              PPLCNet_x2_5               | 混合精度训练 | 混合精度训练 |
|    PPLCNetV2    |             PPLCNetV2_base              | 混合精度训练 | 混合精度训练 |
|     ResNet      |                ResNet50                 | 混合精度训练 | 混合精度训练 |
|     ResNet      |               ResNet50_vd               | 混合精度训练 | 混合精度训练 |
| SwinTransformer | SwinTransformer_tiny_patch4_window7_224 | 混合精度训练 | 混合精度训练 |

- 推理相关：

|    算法名称     |                模型名称                 | device_CPU | device_GPU | batchsize |
| :-------------: | :-------------------------------------: | :--------: | :--------: | :-------: |
|   MobileNetV3   |         MobileNetV3_large_x1_0          |    支持    |    支持    |     1     |
|    PP-ShiTu     |     GeneralRecognition_PPLCNet_x2_5     |    支持    |    支持    |     1     |
|     PPHGNet     |              PPHGNet_small              |    支持    |    支持    |     1     |
|     PPHGNet     |              PPHGNet_tiny               |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x0_25              |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x0_35              |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x0_5               |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x0_75              |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x1_0               |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x1_5               |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x2_0               |    支持    |    支持    |     1     |
|     PPLCNet     |              PPLCNet_x2_5               |    支持    |    支持    |     1     |
|    PPLCNetV2    |             PPLCNetV2_base              |    支持    |    支持    |     1     |
|     ResNet      |                ResNet50                 |    支持    |    支持    |     1     |
|     ResNet      |               ResNet50_vd               |    支持    |    支持    |     1     |
| SwinTransformer | SwinTransformer_tiny_patch4_window7_224 |    支持    |    支持    |     1     |

## 2. 测试流程

以下测试流程以 PPLCNet_x1_0 模型为例。

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
bash test_tipc/prepare.sh test_tipc/configs/PPLCNet/PPLCNet_x1_0_train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```
### 2.3 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_train_inference_python.sh ${your_params_file} lite_train_lite_infer
```

以`PPLCNet_x1_0`的`Linux GPU/CPU 混合精度训练推理测试`为例，命令如下所示。

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/PPLCNet/PPLCNet_x1_0_train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```log
Run successfully with command - PPLCNet_x1_0 - python3.7 tools/train.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.seed=1234 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.loader.use_shared_memory=False -o AMP.scale_loss=65536 -o AMP.use_dynamic_loss_scaling=True -o AMP.level=O2 -o Optimizer.multi_precision=True -o Global.eval_during_train=False -o Global.device=gpu  -o Global.output_dir=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null -o Global.epochs=2     -o DataLoader.Train.sampler.batch_size=8   !
Run successfully with command - PPLCNet_x1_0 - python3.7 tools/eval.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.pretrained_model=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null/PPLCNet_x1_0/latest -o Global.device=gpu  !
Run successfully with command - PPLCNet_x1_0 - python3.7 tools/export_model.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.pretrained_model=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null/PPLCNet_x1_0/latest -o Global.save_inference_dir=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null!
Run successfully with command - PPLCNet_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.use_fp16=False -o Global.inference_model_dir=.././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val -o Global.benchmark=False > .././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/infer_gpu_usetrt_False_precision_False_batchsize_1.log 2>&1 !
Run successfully with command - PPLCNet_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=False -o Global.enable_mkldnn=False -o Global.cpu_num_threads=6 -o Global.inference_model_dir=.././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0_autocast_null -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val -o Global.benchmark=False   > .././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/infer_cpu_usemkldnn_False_threads_6_batchsize_1.log 2>&1 !
Run successfully with command - PPLCNet_x1_0 - python3.7 -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.seed=1234 -o DataLoader.Train.sampler.shuffle=False -o DataLoader.Train.loader.num_workers=0 -o DataLoader.Train.loader.use_shared_memory=False -o AMP.scale_loss=65536 -o AMP.use_dynamic_loss_scaling=True -o AMP.level=O2 -o Optimizer.multi_precision=True -o Global.eval_during_train=False -o Global.device=gpu -o Global.output_dir=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null -o Global.epochs=2     -o DataLoader.Train.sampler.batch_size=8  !
Run successfully with command - PPLCNet_x1_0 - python3.7 tools/eval.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.pretrained_model=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null/PPLCNet_x1_0/latest -o Global.device=gpu  !
Run successfully with command - PPLCNet_x1_0 - python3.7 tools/export_model.py -c ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml -o Global.pretrained_model=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null/PPLCNet_x1_0/latest -o Global.save_inference_dir=./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null!
Run successfully with command - PPLCNet_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=True -o Global.use_tensorrt=False -o Global.use_fp16=False -o Global.inference_model_dir=.././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val -o Global.benchmark=False > .././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/infer_gpu_usetrt_False_precision_False_batchsize_1.log 2>&1 !
Run successfully with command - PPLCNet_x1_0 - python3.7 python/predict_cls.py -c configs/inference_cls.yaml -o Global.use_gpu=False -o Global.enable_mkldnn=False -o Global.cpu_num_threads=6 -o Global.inference_model_dir=.././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/amp_train_gpus_0,1_autocast_null -o Global.batch_size=1 -o Global.infer_imgs=../dataset/ILSVRC2012/val -o Global.benchmark=False   > .././test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/infer_cpu_usemkldnn_False_threads_6_batchsize_1.log 2>&1 !
```

该信息可以在运行log中查看，以`PPLCNet_x1_0`为例，log位置在`./test_tipc/output/PPLCNet_x1_0/lite_train_lite_infer/results_python.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
