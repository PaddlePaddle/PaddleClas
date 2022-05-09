# Lite_arm_cpp_cpu 预测功能测试

Lite_arm_cpp_cpu 预测功能测试的主程序为`test_lite_arm_cpu_cpp.sh`，可以测试基于 Paddle-Lite 预测库的模型推理功能。

## 1. 测试结论汇总

| 模型类型 |device | batchsize | 精度类型| 线程数 |
|  :----:   |  :----: |   :----:   |  :----:  | :----: |
| 正常模型 | arm_cpu | 1 | FP32 | 1 |

## 2. 测试流程
运行环境配置请参考[文档](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/mobilenetv3_prod/Step6/deploy/lite_infer_cpp_arm_cpu/README.md) 的内容配置 TIPC Lite 的运行环境。

### 2.1 功能测试
先运行 `prepare_lite_arm_cpu_cpp.sh` 准备数据和模型，然后运行 `test_lite_arm_cpu_cpp.sh` 进行测试，最终在 `./output` 目录下生成 `lite_*.log` 后缀的日志文件。

```shell
bash test_tipc/prepare_lite_arm_cpu_cpp.sh test_tipc/config/MobileNetV3/MobileNetV3_large_x1_0_lite_arm_cpu_cpp.txt
```

运行预测指令后，在`./output`文件夹下自动会保存运行日志，包括以下文件：

```shell
test_tipc/output/
|- results.log    # 运行指令状态的日志
|- lite_MobileNetV3_large_x1_0_runtime_device_arm_cpu_precision_FP32_batchsize_1_threads_1.log  # ARM_CPU 上 FP32 状态下，线程数设置为1，测试batch_size=1条件下的预测运行日志
......
```
其中results.log中包含了每条指令的运行状态，如果运行成功会输出：

```
Run successfully with command - adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; /data/local/tmp/arm_cpu/mobilenet_v3 /data/local/tmp/arm_cpu/config.txt /data/local/tmp/arm_cpu/demo.jpg'  > ./output/lite_MobileNetV3_large_x1_0_runtime_device_arm_cpu_precision_FP32_batchsize_1_threads_1.log 2>&1!
......
```
如果运行失败，会输出：
```
Run failed with command - adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; /data/local/tmp/arm_cpu/mobilenet_v3 /data/local/tmp/arm_cpu/config.txt /data/local/tmp/arm_cpu/demo.jpg'  > ./output/lite_MobileNetV3_large_x1_0_runtime_device_arm_cpu_precision_FP32_batchsize_1_threads_1.log 2>&1!
......
```
可以很方便的根据results.log中的内容判定哪一个指令运行错误。

## 3. 更多教程

本文档为功能测试用，更详细的 Lite 预测使用教程请参考：[PaddleLite 推理部署](../../docs/zh_CN/inference_deployment/paddle_lite_deploy.md)  。
