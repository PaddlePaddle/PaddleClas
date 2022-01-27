# 手机端 benchmark
---
## 目录

* [1. 简介](#1)
* [2. 评估步骤](#2)
   * [2.1 导出 inference 模型](#2.1)
   * [2.2 benchmark 二进制文件下载](#2.2)
   * [2.3 模型速度 benchmark](#2.3)
   * [2.4 模型优化与速度评估](#2.4)

<a name='1'></a>

## 1. 简介

[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 是飞桨推出的一套功能完善、易用性强且性能卓越的轻量化推理引擎。
轻量化体现在使用较少比特数用于表示神经网络的权重和激活，能够大大降低模型的体积，解决终端设备存储空间有限的问题，推理性能也整体优于其他框架。
[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 使用 Paddle-Lite 进行了[移动端模型的性能评估](../models/Mobile.md)，本部分以 `ImageNet1k` 数据集的 `MobileNetV1` 模型为例，介绍怎样使用 `Paddle-Lite`，在移动端(基于骁龙855的安卓开发平台)对进行模型速度评估。

<a name='2'></a>

## 2. 评估步骤

<a name='2.1'></a>

### 2.1 导出 inference 模型

* 首先需要将训练过程中保存的模型存储为用于预测部署的固化模型，可以使用 `tools/export_model.py` 导出 inference 模型，具体使用方法如下。

```shell
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/MobileNetV1/MobileNetV1.yaml \
    -o Arch.pretrained=./pretrained/MobileNetV1_pretrained/ \
    -o Global.save_inference_dir=./inference/MobileNetV1/
```

在上述命令中，通过参数 `Arch.pretrained` 指定训练过程中保存的模型参数文件，也可以指定参数 `Arch.pretrained=True` 加载 PaddleClas 提供的基于 ImageNet1k 的预训练模型参数，最终在 `inference/MobileNetV1` 文件夹下会保存得到 `inference.pdmodel` 与 `inference.pdiparmas` 文件。

<a name='2.2'></a>

### 2.2 benchmark 二进制文件下载

* 使用 adb(Android Debug Bridge)工具可以连接 Android 手机与 PC 端，并进行开发调试等。安装好 adb，并确保 PC 端和手机连接成功后，使用以下命令可以查看手机的 ARM 版本，并基于此选择合适的预编译库。

```shell
adb shell getprop ro.product.cpu.abi
```

* 下载 benchmark_bin 文件

请根据所用 Android 手机的 ARM 版本选择，ARM 版本为 v8，则使用以下命令下载：

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v8
```

如果查看的 ARM 版本为 v7，则需要下载 v7 版本的 benchmark_bin 文件，下载命令如下：

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v7
```

<a name='2.3'></a>

### 2.3 模型速度 benchmark

PC 端和手机连接成功后，使用下面的命令开始模型评估。

```
sh deploy/lite/benchmark/benchmark.sh ./benchmark_bin_v8 ./inference result_armv8.txt true
```

其中 `./benchmark_bin_v8` 为 benchmark 二进制文件路径，`./inference` 为所有需要评测的模型的路径，`result_armv8.txt` 为保存的结果文件，最后的参数 `true` 表示在评估之后会首先进行模型优化。最终在当前文件夹下会输出 `result_armv8.txt` 的评估结果文件，具体信息如下。

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1                           min = 30.89100    max = 30.73600    average = 30.79750

Threads=2 Warmup=10 Repeats=30
MobileNetV1                           min = 18.26600    max = 18.14000    average = 18.21637

Threads=4 Warmup=10 Repeats=30
MobileNetV1                           min = 10.03200    max = 9.94300     average = 9.97627
```

这里给出了不同线程数下的模型预测速度，单位为 FPS，以线程数为 1 为例，MobileNetV1 在骁龙855上的平均速度为 `30.79750FPS`。

<a name='2.4'></a>

### 2.4 模型优化与速度评估


* 在 2.3 节中提到了在模型评估之前对其进行优化，在这里也可以首先对模型进行优化，再直接加载优化后的模型进行速度评估。

* Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel 优选等等方法。为了使优化过程更加方便易用，Paddle-Lite 提供了 opt 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型。可以在[Paddle-Lite 模型优化工具页面](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html)下载。在这里以 `macOS` 开发环境为例，下载[opt_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt_mac)模型优化工具，并使用下面的命令对模型进行优化。



```shell
model_file="../MobileNetV1/inference.pdmodel"
param_file="../MobileNetV1/inference.pdiparams"
opt_models_dir="./opt_models"
mkdir ${opt_models_dir}
./opt_mac --model_file=${model_file} \
    --param_file=${param_file} \
    --valid_targets=arm \
    --optimize_out_type=naive_buffer \
    --prefer_int8_kernel=false \
    --optimize_out=${opt_models_dir}/MobileNetV1
```

其中 `model_file` 与 `param_file` 分别是导出的 inference 模型结构文件与参数文件地址，转换成功后，会在 `opt_models` 文件夹下生成 `MobileNetV1.nb` 文件。

使用 benchmark_bin 文件加载优化后的模型进行评估，具体的命令如下。

```shell
bash benchmark.sh ./benchmark_bin_v8 ./opt_models result_armv8.txt
```

最终 `result_armv8.txt` 中结果如下：

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1_lite              min = 30.89500    max = 30.78500    average = 30.84173

Threads=2 Warmup=10 Repeats=30
MobileNetV1_lite              min = 18.25300    max = 18.11000    average = 18.18017

Threads=4 Warmup=10 Repeats=30
MobileNetV1_lite              min = 10.00600    max = 9.90000     average = 9.96177
```

以线程数为 1 为例，MobileNetV1 在骁龙855上的平均速度为 `30.84173 ms`。

更加具体的参数解释与 Paddle-Lite 使用方法可以参考 [Paddle-Lite 文档](https://paddle-lite.readthedocs.io/zh/latest/)。
