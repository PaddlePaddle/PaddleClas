# Linux GPU/CPU C++ 服务化部署测试

Linux GPU/CPU C++ 服务化部署测试的主程序为`test_serving_infer_cpp.sh`，可以测试基于C++的模型服务化部署功能。


## 1. 测试结论汇总

- 推理相关：

|    算法名称     |                   模型名称                   | device_CPU | device_GPU |
| :-------------: | :------------------------------------------: | :--------: | :--------: |
|   MobileNetV3   |            MobileNetV3_large_x1_0            |    支持    |    支持    |
|   MobileNetV3   |          MobileNetV3_large_x1_0_KL           |    支持    |    支持    |
|   MobileNetV3   |         MobileNetV3_large_x1_0_PACT          |    支持    |    支持    |
|    PP-ShiTu     |  PPShiTu_general_rec、PPShiTu_mainbody_det   |    支持    |    支持    |
|    PP-ShiTu     |      GeneralRecognition_PPLCNet_x2_5_KL      |    支持    |    支持    |
|    PP-ShiTu     |     GeneralRecognition_PPLCNet_x2_5_PACT     |    支持    |    支持    |
|     PPHGNet     |                PPHGNet_small                 |    支持    |    支持    |
|     PPHGNet     |               PPHGNet_small_KL               |    支持    |    支持    |
|     PPHGNet     |              PPHGNet_small_PACT              |    支持    |    支持    |
|     PPHGNet     |                 PPHGNet_tiny                 |    支持    |    支持    |
|     PPLCNet     |                PPLCNet_x0_25                 |    支持    |    支持    |
|     PPLCNet     |                PPLCNet_x0_35                 |    支持    |    支持    |
|     PPLCNet     |                 PPLCNet_x0_5                 |    支持    |    支持    |
|     PPLCNet     |                PPLCNet_x0_75                 |    支持    |    支持    |
|     PPLCNet     |                 PPLCNet_x1_0                 |    支持    |    支持    |
|     PPLCNet     |               PPLCNet_x1_0_KL                |    支持    |    支持    |
|     PPLCNet     |              PPLCNet_x1_0_PACT               |    支持    |    支持    |
|     PPLCNet     |                 PPLCNet_x1_5                 |    支持    |    支持    |
|     PPLCNet     |                 PPLCNet_x2_0                 |    支持    |    支持    |
|     PPLCNet     |                 PPLCNet_x2_5                 |    支持    |    支持    |
|    PPLCNetV2    |                PPLCNetV2_base                |    支持    |    支持    |
|    PPLCNetV2    |              PPLCNetV2_base_KL               |    支持    |    支持    |
|     ResNet      |                   ResNet50                   |    支持    |    支持    |
|     ResNet      |                 ResNet50_vd                  |    支持    |    支持    |
|     ResNet      |                ResNet50_vd_KL                |    支持    |    支持    |
|     ResNet      |               ResNet50_vd_PACT               |    支持    |    支持    |
| SwinTransformer |   SwinTransformer_tiny_patch4_window7_224    |    支持    |    支持    |
| SwinTransformer |  SwinTransformer_tiny_patch4_window7_224_KL  |    支持    |    支持    |
| SwinTransformer | SwinTransformer_tiny_patch4_window7_224_PACT |    支持    |    支持    |


## 2. 测试流程

### 2.1 准备数据

分类模型默认使用`./deploy/paddleserving/daisy.jpg`作为测试输入图片，无需下载
识别模型默认使用`drink_dataset_v1.0/test_images/001.jpeg`作为测试输入图片，在**2.2 准备环境**中会下载好。

### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
  ```shell
  # 需要安装2.2及以上版本的Paddle
  # 安装GPU版本的Paddle
  python3.7 -m pip install paddlepaddle-gpu==2.2.0
  # 安装CPU版本的Paddle
  python3.7 -m pip install paddlepaddle==2.2.0
  ```

- 安装依赖
  ```shell
  python3.7 -m pip install -r requirements.txt
  ```

- 安装TensorRT
  编译 serving-server 的脚本内会设置 `TENSORRT_LIBRARY_PATH` 这一环境变量，因此编译前需要安装TensorRT。

  如果使用`registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82`镜像进测试，则已自带TensorRT无需安装，
  否则可以参考 [3.2 安装TensorRT](install.md#32-安装tensorrt) 进行安装，并在修改 [build_server.sh](../../deploy/paddleserving/build_server.sh#L62) 的 `TENSORRT_LIBRARY_PATH` 地址为安装后的路径。

- 安装 PaddleServing 相关组件，包括serving_client、serving-app，自动编译并安装带自定义OP的 serving_server 包，以及自动下载并解压推理模型
  ```bash
  # 安装必要依赖包
  python3.7 -m pip install paddle_serving_client==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
  python3.7 -m pip install paddle-serving-app==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

  # 安装编译自定义OP的serving-server包
  pushd ./deploy/paddleserving
  source build_server.sh python3.7
  popd

  # 测试PP-ShiTu识别模型时需安装faiss包
  python3.7-m pip install faiss-cpu==1.7.1post2 -i https://pypi.tuna.tsinghua.edu.cn/simple

  # 下载模型与数据
  bash test_tipc/prepare.sh test_tipc/configs/PPLCNet/PPLCNet_x1_0_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt serving_infer
  ```

### 2.3 功能测试

测试方法如下所示，希望测试不同的模型文件，只需更换为自己的参数配置文件，即可完成对应模型的测试。

```bash
bash test_tipc/test_serving_infer_cpp.sh ${your_params_file} ${mode}
```

以`PPLCNet_x1_0`的`Linux GPU/CPU C++ 服务化部署测试`为例，命令如下所示。


```bash
bash test_tipc/test_serving_infer_cpp.sh test_tipc/configs/PPLCNet/PPLCNet_x1_0_linux_gpu_normal_normal_serving_cpp_linux_gpu_cpu.txt serving_infer
```

输出结果如下，表示命令运行成功。

```
Run successfully with command - PPLCNet_x1_0 - python3.7 test_cpp_serving_client.py > ../../test_tipc/output/PPLCNet_x1_0/server_infer_cpp_gpu_pipeline_batchsize_1.log 2>&1 !
Run successfully with command - PPLCNet_x1_0 - python3.7 test_cpp_serving_client.py > ../../test_tipc/output/PPLCNet_x1_0/server_infer_cpp_cpu_pipeline_batchsize_1.log 2>&1 !
```

预测结果会自动保存在 `./test_tipc/output/PPLCNet_x1_0/server_infer_gpu_pipeline_http_batchsize_1.log` ，可以看到 PaddleServing 的运行结果：

```
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0612 09:55:16.109890 38303 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:9292"): added 1
I0612 09:55:16.172924 38303 general_model.cpp:490] [client]logid=0,client_cost=60.772ms,server_cost=57.6ms.
prediction: daisy, probability: 0.9099399447441101
0.06275796890258789
```


如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。
