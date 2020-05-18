# Paddle-Lite

## 一、简介

[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 是飞桨推出的一套功能完善、易用性强且性能卓越的轻量化推理引擎。
轻量化体现在使用较少比特数用于表示神经网络的权重和激活，能够大大降低模型的体积，解决终端设备存储空间有限的问题，推理性能也整体优于其他框架。
[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 使用 Paddle-Lite 进行了[移动端模型的性能评估](../models/Mobile.md)，本部分以`ImageNet1k`数据集的`MobileNetV1`模型为例，介绍怎样使用`Paddle-Lite`，在移动端(基于骁龙855的安卓开发平台)对进行模型速度评估。


## 二、评估步骤

### 2.1 导出inference模型

* 首先需要将训练过程中保存的模型存储为用于预测部署的固化模型，可以使用`tools/export_model.py`导出inference模型，具体使用方法如下。

```shell
python tools/export_model.py -m MobileNetV1 -p pretrained/MobileNetV1_pretrained/ -o inference/MobileNetV1
```

最终在`inference/MobileNetV1`文件夹下会保存得到`model`与`parmas`文件。


### 2.2 benchmark二进制文件下载

* 使用adb(Android Debug Bridge)工具可以连接Android手机与PC端，并进行开发调试等。安装好adb，并确保PC端和手机连接成功后，使用以下命令可以查看手机的ARM版本，并基于此选择合适的预编译库。

```shell
adb shell getprop ro.product.cpu.abi
```

* 下载benchmark_bin文件

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v8
```

如果查看的ARM版本为v7，则需要下载v7版本的benchmark_bin文件，下载命令如下。

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v7
```

### 2.3 模型速度benchmark

PC端和手机连接成功后，使用下面的命令开始模型评估。

```
sh tools/lite/benchmark.sh ./benchmark_bin_v8 ./inference result_armv8.txt true
```

其中`./benchmark_bin_v8`为benchmark二进制文件路径，`./inference`为所有需要评测的模型的路径，`result_armv8.txt`为保存的结果文件，最后的参数`true`表示在评估之后会首先进行模型优化。最终在当前文件夹下会输出`result_armv8.txt`的评估结果文件，具体信息如下。

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1                           min = 30.89100    max = 30.73600    average = 30.79750

Threads=2 Warmup=10 Repeats=30
MobileNetV1                           min = 18.26600    max = 18.14000    average = 18.21637

Threads=4 Warmup=10 Repeats=30
MobileNetV1                           min = 10.03200    max = 9.94300     average = 9.97627
```

这里给出了不同线程数下的模型预测速度，单位为FPS，以线程数为1为例，MobileNetV1在骁龙855上的平均速度为`30.79750FPS`。


### 2.4 模型优化与速度评估


* 在2.3节中提到了在模型评估之前对其进行优化，在这里也可以首先对模型进行优化，再直接加载优化后的模型进行速度评估。

* Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel优选等等方法。为了使优化过程更加方便易用，Paddle-Lite提供了opt 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。可以在[Paddle-Lite模型优化工具页面](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)下载。在这里以`MacOS`开发环境为例，下载[opt_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt_mac)模型优化工具，并使用下面的命令对模型进行优化。



```shell
model_file="../MobileNetV1/model"
param_file="../MobileNetV1/params"
opt_models_dir="./opt_models"
mkdir ${opt_models_dir}
./opt_mac --model_file=${model_file} \
    --param_file=${param_file} \
    --valid_targets=arm \
    --optimize_out_type=naive_buffer \
    --prefer_int8_kernel=false \
    --optimize_out=${opt_models_dir}/MobileNetV1
```

其中`model_file`与`param_file`分别是导出的inference模型结构文件与参数文件地址，转换成功后，会在`opt_models`文件夹下生成`MobileNetV1.nb`文件。




使用benchmark_bin文件加载优化后的模型进行评估，具体的命令如下。

```shell
bash benchmark.sh ./benchmark_bin_v8 ./opt_models result_armv8.txt
```

最终`result_armv8.txt`中结果如下。

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1_lite              min = 30.89500    max = 30.78500    average = 30.84173

Threads=2 Warmup=10 Repeats=30
MobileNetV1_lite              min = 18.25300    max = 18.11000    average = 18.18017

Threads=4 Warmup=10 Repeats=30
MobileNetV1_lite              min = 10.00600    max = 9.90000     average = 9.96177
```


以线程数为1为例，MobileNetV1在骁龙855上的平均速度为`30.84173FPS`。


更加具体的参数解释与Paddle-Lite使用方法可以参考 [Paddle-Lite 文档](https://paddle-lite.readthedocs.io/zh/latest/)。
