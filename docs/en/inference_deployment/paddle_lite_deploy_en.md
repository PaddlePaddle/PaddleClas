# Tutorial of PaddleClas Mobile Deployment

This tutorial will introduce how to use [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) to deploy PaddleClas models on mobile phones.

Paddle-Lite is a lightweight inference engine for PaddlePaddle. It provides efficient inference capabilities for mobile phones and IoTs,  and extensively integrates cross-platform hardware to provide lightweight deployment solutions for mobile-side deployment issues.

If you only want to test speed, please refer to [The tutorial of Paddle-Lite mobile-side benchmark test](../others/paddle_mobile_inference_en.md).

---

## Catalogue

- [1. Preparation](#1)
    - [1.1 Build Paddle-Lite library](#1.1)
    - [1.2 Download inference library for Android or iOS](#1.2)
- [2. Start running](#2)
    - [2.1 Inference Model Optimization](#2.1)
        - [2.1.1 [RECOMMEND] Use pip to install Paddle-Lite and optimize model](#2.1.1)
        - [2.1.2 Compile Paddle-Lite to generate opt tool](#2.1.2)
        - [2.1.3 Demo of get the optimized model](#2.1.3)
        - [2.1.4 Compile to get the executable file clas_system](#2.1.4)
    - [2.2 Run optimized model on Phone](#2.2)
- [3. FAQ](#3)

<a name="1"></a>
## 1. Preparation

PaddeLite currently supports the following platforms:
- Computer (for compiling Paddle-Lite)
- Mobile phone (arm7 or arm8)

<a name="1.1"></a>
### 1.1 Prepare cross-compilation environment

The cross-compilation environment is used to compile the C++ demos of Paddle-Lite and PaddleClas.

For the detailed compilation directions of different development environments, please refer to the corresponding [document](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html).

<a name="1.2"></a>
## 1.2 Download inference library for Android or iOS

|Platform|Inference Library Download Link|
|-|-|
|Android|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv7.clang.c++_static.with_extra.with_cv.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv.tar.gz) |
|iOS|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)|

**NOTE**:

1. If you download the inference library from [Paddle-Lite official document](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#android-toolchain-gcc), please choose `with_extra=ON` , `with_cv=ON` .

2. It is recommended to build inference library using [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) develop branch if you want to deploy the [quantitative](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/quantization/README_en.md) model to mobile phones. Please refer to the [link](https://paddle-lite.readthedocs.io/) for more detailed information about compiling.


The structure of the inference library is as follows:

```
inference_lite_lib.android.armv8.clang.c++_static.with_extra.with_cv/
|-- cxx                                                    C++ inference library and header files
|   |-- include                                            C++ header files
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                                                                  C++ inference library
|       |-- libpaddle_api_light_bundled.a           C++ static library
|       `-- libpaddle_light_api_shared.so           C++ dynamic library
|-- java                                                     Java inference library
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                                                     C++ and java demos
|   |-- cxx                                                                                  C++ demos
|   `-- java                                                                              Java demos
```

<a name="2"></a>
## 2. Start running

<a name="2.1"></a>
## 2.1 Inference Model Optimization

Paddle-Lite provides a variety of strategies to automatically optimize the original training model, including quantization, sub-graph fusion, hybrid scheduling, Kernel optimization and so on. In order to make the optimization process more convenient and easy to use, Paddle-Lite provides `opt` tool to automatically complete the optimization steps and output a lightweight, optimal executable model.

**NOTE**: If you have already got the `.nb` file, you can skip this step.

<a name="2.1.1"></a>
### 2.1.1 [RECOMMEND] Use `pip` to install Paddle-Lite and optimize model

* Use pip to install Paddle-Lite. The following command uses `pip3.7` .

```shell
pip install paddlelite==2.8
```
**Note**：The version of `paddlelite`'s wheel must match that of inference lib.

* Use `paddle_lite_opt` to optimize inference model, the parameters of `paddle_lite_opt` are as follows:

| Parameters              | Explanation                                                  |
| ----------------------- | ------------------------------------------------------------ |
| --model_dir             | Path to the PaddlePaddle model (no-combined) file to be optimized. |
| --model_file            | Path to the net structure file of PaddlePaddle model (combined) to be optimized. |
| --param_file            | Path to the net weight files of PaddlePaddle model (combined) to be optimized. |
| --optimize_out_type     | Type of output model, `protobuf` by default. Supports `protobuf` and `naive_buffer` . Compared with `protobuf`, you can use`naive_buffer` to get a more lightweight serialization/deserialization model. If you need to predict on the mobile-side, please set it to `naive_buffer`. |
| --optimize_out          | Path to output model, not needed to add `.nb` suffix.        |
| --valid_targets         | The executable backend of the model, `arm` by default. Supports one or some of `x86` , `arm` , `opencl` , `npu` , `xpu`. If set more than one, please separate the options by space, and the `opt` tool will choose the best way automatically. If need to support Huawei NPU (DaVinci core carried by Kirin 810/990 SoC), please set it to `npu arm` . |
| --record_tailoring_info | Whether to enable `Cut the Library Files According To the Model` , `false` by default. If need to record kernel and OP infos of optimized model, please set it to `true`. |

In addition, you can run `paddle_lite_opt` to get more detailed information about how to use.

<a name="2.1.2"></a>
### 2.1.2 Compile Paddle-Lite to generate `opt` tool

Optimizing model requires Paddle-Lite's `opt` executable file, which can be obtained by compiling the Paddle-Lite. The steps are as follows:

```shell
# get the Paddle-Lite source code, if have gotten , please skip
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout develop
# compile
./lite/tools/build.sh build_optimize_tool
```

After the compilation is complete, the `opt` file is located under `build.opt/lite/api/`.

`opt` tool is used in the same way as `paddle_lite_opt` , please refer to [2.1.1](#2.1.1).

<a name="2.1.3"></a>
### 2.1.3 Demo of get the optimized model

Taking the `MobileNetV3_large_x1_0` model of PaddleClas as an example, we will introduce how to use `paddle_lite_opt` to complete the conversion from the pre-trained model to the inference model, and then to the Paddle-Lite optimized model.

```shell
# enter PaddleClas root directory
cd PaddleClas_root_path

# download and uncompress the inference model
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar


# convert inference model to Paddle-Lite optimized model
paddle_lite_opt --model_file=./MobileNetV3_large_x1_0_infer/inference.pdmodel --param_file=./MobileNetV3_large_x1_0_infer/inference.pdiparams --optimize_out=./MobileNetV3_large_x1_0
```

When the above code command is completed, there will be ``MobileNetV3_large_x1_0.nb` in the current directory, which is the converted model file.
<a name="2.1.4"></a>

#### 2.1.4 Compile to get the executable file clas_system

```shell
# Clone the Autolog repository to get automation logs
cd PaddleClas_root_path
cd deploy/lite/
git clone https://github.com/LDOUBLEV/AutoLog.git
```

```shell
# Compile
make -j
```

After executing the `make` command, the `clas_system` executable file is generated in the current directory, which is used for Lite prediction.

<a name="2.2"></a>
## 2.2 Run optimized model on Phone

1. Prepare an Android phone with `arm8`. If the compiled inference library and `opt` file are `armv7`, you need an `arm7` phone and modify `ARM_ABI = arm7` in the Makefile.

2. Install the ADB tool on the computer.

    * Install ADB for MAC

      Recommend use homebrew to install.

      ```shell
      brew cask install android-platform-tools
      ```
    * Install ADB for Linux

      ```shell
      sudo apt update
      sudo apt install -y wget adb
      ```
    * Install ADB for windows
      If install ADB fo Windows, you need to download from Google's Android platform: [Download Link](https://developer.android.com/studio).

3. First, make sure the phone is connected to the computer, turn on the `USB debugging` option of the phone, and select the `file transfer` mode. Verify whether ADB is installed successfully as follows:

    ```shell
    $ adb devices

    List of devices attached
    744be294    device
    ```

    If there is `device` output like the above, it means the installation was successful.

4. Push the optimized model, prediction library file, test image and class map file to the phone.

```shell
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

You should put the model that optimized by `paddle_lite_opt` under the `demo/cxx/clas/debug/` directory. In this example, use `MobileNetV3_large_x1_0.nb` model file generated in [2.1.3](#2.1.3).

**NOTE**:

* `Imagenet1k_label_list.txt` is the category mapping file of the `ImageNet1k` dataset. If use a custom category, you need to replace the category mapping file.
* `config.txt`  contains the hyperparameters, as follows:

```shell
clas_model_file ./MobileNetV3_large_x1_0.nb # path of model file
label_path ./imagenet1k_label_list.txt      # path of category mapping file
resize_short_size 256                       # the short side length after resize
crop_size 224                               # side length used for inference after cropping
visualize 0                                 # whether to visualize. If you set it to 1, an image file named 'clas_result.png' will be generated in the current directory.
num_threads 1 # The number of threads, the default is 1
precision FP32 # Precision type, you can choose FP32 or INT8, the default is FP32
runtime_device arm_cpu # Device type, the default is arm_cpu
enable_benchmark 0 # Whether to enable benchmark, the default is 0
tipc_benchmark 0 # Whether to enable tipc_benchmark, the default is 0
```

5. Run Model on Phone

Execute the following command to complete the prediction on the mobile phone.

```shell
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/arm_cpu/; /data/local/tmp/arm_cpu/clas_system /data/local/tmp/arm_cpu/config.txt /data/local/tmp/arm_cpu/tabby_cat.jpg'
```

The result is as follows:

![](../../images/inference_deployment/lite_demo_result.png)

<a name="3"></a>
## 3. FAQ

Q1：If I want to change the model, do I need to go through the all process again?  
A1：If you have completed the above steps, you only need to replace the `.nb` model file after replacing the model. At the same time, you may need to modify the path of `.nb` file in the config file and change the category mapping file to be compatible the model .

Q2：How to change the test picture?  
A2：Replace the test image under debug folder with the image you want to test，and then repush to the Phone again.
