# Benchmark on Mobile

---

## Catalogue

* [1. Introduction](#1)
* [2. Evaluation Steps](#2)
   * [2.1 Export the Inference Model](#2.1)
   * [2.2 Download Benchmark Binary File](#2.2)
   * [2.3 Inference benchmark](#2.3)
   * [2.4 Model Optimization and Speed Evaluation](#2.4)

<a name='1'></a>
## 1. Introduction

[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) is a set of lightweight inference engine which is fully functional, easy to use and then performs well. Lightweighting is reflected in the use of fewer bits to represent the weight and activation of the neural network, which can greatly reduce the size of the model, solve the problem of limited storage space of the mobile device, and the inference speed is better than other frameworks on the whole.

In [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), we uses Paddle-Lite to [evaluate the performance on the mobile device](../models/Mobile_en.md), in this section we uses the `MobileNetV1` model trained on the `ImageNet1k` dataset as an example to introduce how to use `Paddle-Lite` to evaluate the model speed on the mobile terminal (evaluated on SD855)

<a name='2'></a>
## 2. Evaluation Steps

<a name='2.1'></a>
### 2.1 Export the Inference Model

* First you should transform the saved model during training to the special model which can be used to inference, the special model can be exported by `tools/export_model.py`, the specific way of transform is as follows.

```shell
python tools/export_model.py -m MobileNetV1 -p pretrained/MobileNetV1_pretrained/ -o inference/MobileNetV1
```

Finally the `model` and `parmas` can be saved in `inference/MobileNetV1`.

<a name='2.2'></a>
### 2.2 Download Benchmark Binary File

* Use the adb (Android Debug Bridge) tool to connect the Android phone and the PC, then develop and debug. After installing adb and ensuring that the PC and the phone are successfully connected, use the following command to view the ARM version of the phone and select the pre-compiled library based on ARM version.

```shell
adb shell getprop ro.product.cpu.abi
```

* Download Benchmark_bin File

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v8
```

If the ARM version is v7, the v7 benchmark_bin file should be downloaded, the command is as follow.

```shell
wget -c https://paddle-inference-dist.bj.bcebos.com/PaddleLite/benchmark_0/benchmark_bin_v7
```

<a name='2.3'></a>
### 2.3 Inference benchmark

After the PC and mobile phone are successfully connected, use the following command to start the model evaluation.

```
sh deploy/lite/benchmark/benchmark.sh ./benchmark_bin_v8 ./inference result_armv8.txt true
```

Where `./benchmark_bin_v8` is the path of the benchmark binary file, `./inference` is the path of all the models that need to be evaluated, `result_armv8.txt` is the result file, and the final parameter `true` means that the model will be optimized before evaluation. Eventually, the evaluation result file of `result_armv8.txt` will be saved in the current folder. The specific performances are as follows.

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1                           min = 30.89100    max = 30.73600    average = 30.79750

Threads=2 Warmup=10 Repeats=30
MobileNetV1                           min = 18.26600    max = 18.14000    average = 18.21637

Threads=4 Warmup=10 Repeats=30
MobileNetV1                           min = 10.03200    max = 9.94300     average = 9.97627
```

Here is the model inference speed under different number of threads, the unit is FPS, taking model on one threads as an example, the average speed of MobileNetV1 on SD855 is `30.79750FPS`.

<a name='2.4'></a>
### 2.4 Model Optimization and Speed Evaluation

* In II.III section, we mention that the model will be optimized before evaluation, here you can  first optimize the model, and then directly load the optimized model for speed evaluation

* Paddle-Lite
In Paddle-Lite, we provides multiple strategies to automatically optimize the original training model, which contain Quantify, Subgraph fusion, Hybrid scheduling, Kernel optimization and so on. In order to make the optimization more convenient and easy to use, we provide opt tools to automatically complete the optimization steps and output a lightweight, optimal  and executable model in Paddle-Lite, which can be downloaded on [Paddle-Lite Model Optimization Page](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html). Here we take `MacOS` as our development environment, download[opt_mac](https://paddlelite-data.bj.bcebos.com/model_optimize_tool/opt_mac) model optimization tools and use the following commands to optimize the model.


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

Where the `model_file` and `param_file` are exported model file and the file address respectively, after transforming successfully, the `MobileNetV1.nb` will be saved in `opt_models`



Use the benchmark_bin file to load the optimized model for evaluation. The commands are as follows.

```shell
bash benchmark.sh ./benchmark_bin_v8 ./opt_models result_armv8.txt
```

Finally the result is saved in `result_armv8.txt` and shown as follow.

```
PaddleLite Benchmark
Threads=1 Warmup=10 Repeats=30
MobileNetV1_lite              min = 30.89500    max = 30.78500    average = 30.84173

Threads=2 Warmup=10 Repeats=30
MobileNetV1_lite              min = 18.25300    max = 18.11000    average = 18.18017

Threads=4 Warmup=10 Repeats=30
MobileNetV1_lite              min = 10.00600    max = 9.90000     average = 9.96177
```


Taking the model on one threads as an example, the average speed of MobileNetV1 on SD855 is `30.84173FPS`.

More specific parameter explanation and Paddle-Lite usage can refer to [Paddle-Lite docs](https://paddle-lite.readthedocs.io/zh/latest/)。
