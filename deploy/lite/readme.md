# 端侧部署

本教程将介绍基于[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 在移动端部署PaddleClas分类模型的详细步骤。

Paddle Lite是飞桨轻量化推理引擎，为手机、IOT端提供高效推理能力，并广泛整合跨平台硬件，为端侧部署及应用落地问题提供轻量化的部署方案。如果希望直接测试速度，可以参考[Paddle-Lite移动端benchmark测试教程](../../docs/zh_CN/extension/paddle_mobile_inference.md)。


## 1. 准备环境

### 运行准备
- 电脑（编译Paddle Lite）
- 安卓手机（armv7或armv8）

### 1.1 准备交叉编译环境
交叉编译环境用于编译 Paddle Lite 和 PaddleClas 的C++ demo。
支持多种开发环境，不同开发环境的编译流程请参考对应文档。

1. [Docker](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#docker)
2. [Linux](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#linux)
3. [MAC OS](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#mac-os)

### 1.2 准备预测库

预测库有两种获取方式：
1. [建议]直接下载，预测库下载链接如下：
      |平台|预测库下载链接|
      |-|-|
      |Android|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.with_extra.with_cv.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz)|
      |iOS|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)|

      **注**：
      1. 如果是从 Paddle-Lite [官方文档](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#android-toolchain-gcc)下载的预测库，
      注意选择`with_extra=ON，with_cv=ON`的下载链接。
      2. 如果使用量化的模型部署在端侧，建议使用Paddle-Lite develop分支编译预测库。

2. 编译Paddle-Lite得到预测库，Paddle-Lite的编译方式如下：
```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
# 如果使用编译方式，建议使用develop分支编译预测库
git checkout develop
./lite/tools/build_android.sh  --arch=armv8  --with_cv=ON --with_extra=ON
```

**注意**：编译Paddle-Lite获得预测库时，需要打开`--with_cv=ON --with_extra=ON`两个选项，`--arch`表示`arm`版本，这里指定为armv8，更多编译命令介绍请参考[链接](https://paddle-lite.readthedocs.io/zh/latest/user_guides/Compile/Android.html#id2)。

直接下载预测库并解压后，可以得到`inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/`文件夹，通过编译Paddle-Lite得到的预测库位于`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/`文件夹下。
预测库的文件目录如下：

```
inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/
|-- cxx                                        C++ 预测库和头文件
|   |-- include                                C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                           C++预测库
|       |-- libpaddle_api_light_bundled.a             C++静态库
|       `-- libpaddle_light_api_shared.so             C++动态库
|-- java                                     Java预测库
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                                     C++和Java示例代码
|   |-- cxx                                  C++  预测库demo
|   `-- java                                 Java 预测库demo
```

## 2 开始运行

### 2.1 模型优化

Paddle-Lite 提供了多种策略来自动优化原始的模型，其中包括量化、子图融合、混合调度、Kernel优选等方法，使用Paddle-Lite的`opt`工具可以自动对inference模型进行优化，目前支持两种优化方式，优化后的模型更轻量，模型运行速度更快。

**注意**：如果已经准备好了 `.nb` 结尾的模型文件，可以跳过此步骤。

#### 2.1.1 [建议]pip安装paddlelite并进行转换

Python下安装 `paddlelite`，目前最高支持`Python3.7`。
**注意**：`paddlelite`whl包版本必须和预测库版本对应。

```shell
pip install paddlelite==2.10
```

之后使用`paddle_lite_opt`工具可以进行inference模型的转换。`paddle_lite_opt`的部分参数如下

|选项|说明|
|-|-|
|--model_dir|待优化的PaddlePaddle模型（非combined形式）的路径|
|--model_file|待优化的PaddlePaddle模型（combined形式）的网络结构文件路径|
|--param_file|待优化的PaddlePaddle模型（combined形式）的权重文件路径|
|--optimize_out_type|输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf|
|--optimize_out|优化模型的输出路径|
|--valid_targets|指定模型可执行的backend，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm|
|--record_tailoring_info|当使用 根据模型裁剪库文件 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false|

`--model_file`表示inference模型的model文件地址，`--param_file`表示inference模型的param文件地址；`optimize_out`用于指定输出文件的名称（不需要添加`.nb`的后缀）。直接在命令行中运行`paddle_lite_opt`，也可以查看所有参数及其说明。


#### 2.1.2 源码编译Paddle-Lite生成opt工具

模型优化需要Paddle-Lite的`opt`可执行文件，可以通过编译Paddle-Lite源码获得，编译步骤如下：
```shell
# 如果准备环境时已经clone了Paddle-Lite，则不用重新clone Paddle-Lite
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout develop
# 启动编译
./lite/tools/build.sh build_optimize_tool
```

编译完成后，`opt`文件位于`build.opt/lite/api/`下，可通过如下方式查看`opt`的运行选项和使用方式；
```shell
cd build.opt/lite/api/
./opt
```

`opt`的使用方式与参数与上面的`paddle_lite_opt`完全一致。

<a name="2.1.3"></a>

#### 2.1.3 转换示例

下面以PaddleClas的 `MobileNetV3_large_x1_0` 模型为例，介绍使用`paddle_lite_opt`完成预训练模型到inference模型，再到Paddle-Lite优化模型的转换。

```shell
# 进入PaddleClas根目录
cd PaddleClas_root_path

# 下载并解压inference模型
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar

# 将inference模型转化为Paddle-Lite优化模型
paddle_lite_opt --model_file=./MobileNetV3_large_x1_0_infer/inference.pdmodel --param_file=./MobileNetV3_large_x1_0_infer/inference.pdiparams --optimize_out=./MobileNetV3_large_x1_0
```

最终在当前文件夹下生成`MobileNetV3_large_x1_0.nb`的文件。

**注意**：`--optimize_out` 参数为优化后模型的保存路径，无需加后缀`.nb`；`--model_file` 参数为模型结构信息文件的路径，`--param_file` 参数为模型权重信息文件的路径，请注意文件名。

<a name="2.1.4"></a>

#### 2.1.4 执行编译，得到可执行文件clas_system

```shell
# 克隆 Autolog 代码库，以便获取自动化日志
cd PaddleClas_root_path
cd deploy/lite/
git clone https://github.com/LDOUBLEV/AutoLog.git
```

```shell
# 编译
make -j
```

执行 `make` 命令后，会在当前目录生成 `clas_system` 可执行文件，该文件用于 Lite 预测。

<a name="2.2与手机联调"></a>
### 2.2 与手机联调

首先需要进行一些准备工作。
1. 准备一台arm8的安卓手机，如果编译的预测库和opt文件是armv7，则需要arm7的手机，并修改Makefile中`ARM_ABI = arm7`。
2. 电脑上安装ADB工具，用于调试。 ADB安装方式如下：

    3.1. MAC电脑安装ADB:

    ```shell
    brew cask install android-platform-tools
    ```
    3.2. Linux安装ADB
    ```shell
    sudo apt update
    sudo apt install -y wget adb
    ```
    3.3. Window安装ADB

    win上安装需要去谷歌的安卓平台下载ADB软件包进行安装：[链接](https://developer.android.com/studio)

3. 手机连接电脑后，开启手机`USB调试`选项，选择`文件传输`模式，在电脑终端中输入：

```shell
adb devices
```
如果有device输出，则表示安装成功，如下所示：
```
List of devices attached
744be294    device
```


4. 将优化后的模型、预测库文件、测试图像和类别映射文件push到手机上。

```shell
adb shell mkdir -p /data/local/tmp/arm_cpu/
adb push clas_system /data/local/tmp/arm_cpu/
adb shell chmod +x /data/local/tmp/arm_cpu//clas_system
adb push inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/cxx/lib/libpaddle_light_api_shared.so /data/local/tmp/arm_cpu/
adb push MobileNetV3_large_x1_0.nb /data/local/tmp/arm_cpu/
adb push config.txt /data/local/tmp/arm_cpu/
adb push ../../ppcls/utils/imagenet1k_label_list.txt /data/local/tmp/arm_cpu/
adb push imgs/tabby_cat.jpg /data/local/tmp/arm_cpu/
```

#### 注意：
* 上述文件中，`imagenet1k_label_list.txt` 是ImageNet1k数据集的类别映射文件，如果使用自定义的类别，需要更换该类别映射文件。

*  `config.txt` 包含了分类器的超参数，如下：

```shell
clas_model_file ./MobileNetV3_large_x1_0.nb # 模型文件地址
label_path ./imagenet1k_label_list.txt 			# 类别映射文本文件
resize_short_size 256 # resize之后的短边边长
crop_size 224 				# 裁剪后用于预测的边长
visualize 0 # 是否进行可视化，如果选择的话，会在当前文件夹下生成名为clas_result.png的图像文件
num_threads 1 # 线程数，默认是1。
precision FP32 # 精度类型，可以选择 FP32 或者 INT8，默认是 FP32。
runtime_device arm_cpu # 设备类型，默认是 arm_cpu
enable_benchmark 0 # 是否开启benchmark， 默认是 0
tipc_benchmark 0 # 是否开启tipc_benchmark，默认是 0
```

5. 执行预测命令

执行以下命令，可完成在手机上的预测。

```shell
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; /data/local/tmp/arm_cpu/clas_system /data/local/tmp/arm_cpu/config.txt /data/local/tmp/arm_cpu/tabby_cat.jpg'
```

运行效果如下：

<div align="center">
    <img src="./imgs/lite_demo_result.png" width="600">
</div>


## FAQ
Q1：如果想更换模型怎么办，需要重新按照流程走一遍吗？  
A1：如果已经走通了上述步骤，更换模型只需要替换 `.nb` 模型文件即可，同时要注意修改下配置文件中的 `.nb` 文件路径以及类别映射文件（如有必要）。

Q2：换一个图测试怎么做？  
A2：替换 debug 下的测试图像为你想要测试的图像，使用 ADB 再次 push 到手机上即可。

