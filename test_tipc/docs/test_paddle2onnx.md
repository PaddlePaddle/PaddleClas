# Paddle2onnx预测功能测试

PaddleServing预测功能测试的主程序为`test_paddle2onnx.sh`，可以测试Paddle2ONNX的模型转化功能，并验证正确性。

## 1. 测试结论汇总

基于训练是否使用量化，进行本测试的模型可以分为`正常模型`和`量化模型`，这两类模型对应的Paddle2ONNX预测功能汇总如下：

| 模型类型 |device |
|  ----   |  ---- |
| 正常模型 | GPU |
| 正常模型 | CPU |


## 2. 测试流程

以下内容以`ResNet50`模型的paddle2onnx测试为例

### 2.1 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_paddle2onnx.sh`进行测试，最终在`test_tipc/output/ResNet50`目录下生成`paddle2onnx_infer_*.log`后缀的日志文件
下方展示以PPHGNet_small为例的测试命令与结果。

```shell
bash test_tipc/prepare.sh ./test_tipc/config/ResNet/ResNet50_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt paddle2onnx_infer

# 用法:
bash test_tipc/test_paddle2onnx.sh ./test_tipc/config/ResNet/ResNet50_linux_gpu_normal_normal_paddle2onnx_python_linux_cpu.txt
```

#### 运行结果

各测试的运行情况会打印在 `./test_tipc/output/ResNet50/results_paddle2onnx.log` 中：
运行成功时会输出：

```
Run successfully with command -  paddle2onnx --model_dir=./deploy/models/ResNet50_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=./deploy/models/ResNet50_infer/inference.onnx --opset_version=10 --enable_onnx_checker=True!
Run successfully with command - cd deploy && python3.7 ./python/predict_cls.py -o Global.inference_model_dir=./models/ResNet50_infer -o Global.use_onnx=True -o Global.use_gpu=False -c=configs/inference_cls.yaml > ../test_tipc/output/ResNet50/paddle2onnx_infer_cpu.log 2>&1 && cd ../!

```

运行失败时会输出：

```
Run failed with command -  paddle2onnx --model_dir=./deploy/models/ResNet50_infer/ --model_filename=inference.pdmodel --params_filename=inference.pdiparams --save_file=./deploy/models/ResNet50_infer/inference.onnx --opset_version=10 --enable_onnx_checker=True!
Run failed with command - cd deploy && python3.7 ./python/predict_cls.py -o Global.inference_model_dir=./models/ResNet50_infer -o Global.use_onnx=True -o Global.use_gpu=False -c=configs/inference_cls.yaml > ../test_tipc/output/ResNet50/paddle2onnx_infer_cpu.log 2>&1 && cd ../!
...
```


## 3. 更多教程

本文档为功能测试用，更详细的Paddle2onnx预测使用教程请参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
