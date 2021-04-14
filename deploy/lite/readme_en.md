
# Tutorial of PaddleClas Mobile Deployment

This tutorial will introduce how to use [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) to deploy PaddleClas models on mobile phones.

Paddle-Lite is a lightweight inference engine for PaddlePaddle. It provides efficient inference capabilities for mobile phones and IoTs,  and extensively integrates cross-platform hardware to provide lightweight deployment solutions for mobile-side deployment issues.

If you only want to test speed, please refer to [The tutorial of Paddle-Lite mobile-side benchmark test](../../docs/zh_CN/extension/paddle_mobile_inference.md).

## 1. Preparation

- Computer (for compiling Paddle-Lite)
- Mobile phone (arm7 or arm8)

## 2. Build Paddle-Lite library

The cross-compilation environment is used to compile the C++ demos of Paddle-Lite and PaddleClas.

For the detailed compilation directions of different development environments, please refer to the corresponding documents.

1. [Docker](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#docker)
2. [Linux](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#linux)
3. [macOS](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#mac-os)

## 3. Download inference library for Android or iOS

|Platform|Inference Library Download Link|
|-|-|
|Android|[arm7](https://paddlelite-data.bj.bcebos.com/Release/2.8-rc/Android/gcc/inference_lite_lib.android.armv7.gcc.c++_static.with_extra.with_cv.tar.gz) / [arm8](https://paddlelite-data.bj.bcebos.com/Release/2.8-rc/Android/gcc/inference_lite_lib.android.armv8.gcc.c++_static.with_extra.with_cv.tar.gz)|
|iOS|[arm7](https://paddlelite-data.bj.bcebos.com/Release/2.8-rc/iOS/inference_lite_lib.ios.armv7.with_cv.with_extra.tiny_publish.tar.gz) / [arm8](https://paddlelite-data.bj.bcebos.com/Release/2.8-rc/iOS/inference_lite_lib.ios.armv8.with_cv.with_extra.tiny_publish.tar.gz)|

**NOTE**:

1. If you download the inference library from [Paddle-Lite official document](https://paddle-lite.readthedocs.io/zh/latest/quick_start/release_lib.html#android-toolchain-gcc), please choose `with_extra=ON` , `with_cv=ON` .

2. It is recommended to build inference library using [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) develop branch if you want to deploy the [quantitative](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/quantization/README_en.md) model to mobile phones. Please refer to the [link](https://paddle-lite.readthedocs.io/zh/latest/user_guides/Compile/Android.html#id2) for more detailed information about compiling.


The structure of the inference library is as follows:

```
inference_lite_lib.android.armv8/
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



## 4. Inference Model Optimization

Paddle-Lite provides a variety of strategies to automatically optimize the original training model, including quantization, sub-graph fusion, hybrid scheduling, Kernel optimization and so on. In order to make the optimization process more convenient and easy to use, Paddle-Lite provides `opt` tool to automatically complete the optimization steps and output a lightweight, optimal executable model.

**NOTE**: If you have already got the `.nb` file, you can skip this step.

<a name="4.1"></a>

### 4.1 [RECOMMEND] Use `pip` to install Paddle-Lite and optimize model

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

### 4.2 Compile Paddle-Lite to generate `opt` tool

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

`opt` tool is used in the same way as `paddle_lite_opt` , please refer to [4.1](#4.1).

<a name="4.3"></a>

### 4.3 Demo of get the optimized model

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

## 5. Run optimized model on Phone

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

    First, make sure the phone is connected to the computer, turn on the `USB debugging` option of the phone, and select the `file transfer` mode. Verify whether ADB is installed successfully as follows:

    ```shell
    $ adb devices

    List of devices attached
    744be294    device
    ```

    If there is `device` output like the above, it means the installation was successful.

4. Prepare optimized model, inference library files, test image and dictionary file used.

```shell
cd PaddleClas_root_path
cd deploy/lite/

# prepare.sh will put the inference library files, the test image and the dictionary files in demo/cxx/clas
sh prepare.sh /{lite inference library path}/inference_lite_lib.android.armv8

# enter the working directory of lite demo
cd /{lite inference library path}/inference_lite_lib.android.armv8/
cd demo/cxx/clas/

# copy the C++ inference dynamic library file （ie. .so) to the debug folder
cp ../../../cxx/lib/libpaddle_light_api_shared.so ./debug/
```

The `prepare.sh` take `PaddleClas/deploy/lite/imgs/tabby_cat.jpg` as the test image, and copy it to the `demo/cxx/clas/debug/` directory.

You should put the model that optimized by `paddle_lite_opt` under the `demo/cxx/clas/debug/` directory. In this example, use `MobileNetV3_large_x1_0.nb` model file generated in [2.1.3](#4.3).

The structure of the clas demo is as follows after the above command is completed:

```
demo/cxx/clas/
|-- debug/
|   |--MobileNetV3_large_x1_0.nb                    class model
|   |--tabby_cat.jpg                              test image
|   |--imagenet1k_label_list.txt                    dictionary file
|   |--libpaddle_light_api_shared.so              C++ .so file
|   |--config.txt                                 config file
|-- config.txt                                    config file
|-- image_classfication.cpp                       source code
|-- Makefile                                      compile file
```

**NOTE**:

* `Imagenet1k_label_list.txt` is the category mapping file of the `ImageNet1k` dataset. If use a custom category, you need to replace the category mapping file.
* `config.txt`  contains the hyperparameters, as follows:

```shell
clas_model_file ./MobileNetV3_large_x1_0.nb # path of model file
label_path ./imagenet1k_label_list.txt      # path of category mapping file
resize_short_size 256                       # the short side length after resize
crop_size 224                               # side length used for inference after cropping

visualize 0                                 # whether to visualize. If you set it to 1, an image file named 'clas_result.png' will be generated in the current directory.
```

5. Run Model on Phone

```shell
# run compile to get the executable file 'clas_system'
make -j

# move the compiled executable file to the debug folder
mv clas_system ./debug/

# push the debug folder to Phone
adb push debug /data/local/tmp/

adb shell
cd /data/local/tmp/debug
export LD_LIBRARY_PATH=/data/local/tmp/debug:$LD_LIBRARY_PATH

# the usage of clas_system is as follows:
# ./clas_system "path of config file" "path of test image"
./clas_system ./config.txt ./tabby_cat.jpg
```

**NOTE**: If you make changes to the code, you need to recompile and repush the `debug ` folder to the phone.

The result is as follows:

<div align="center">
    <img src="./imgs/lite_demo_result.png" width="600">
</div>



## FAQ

Q1：If I want to change the model, do I need to go through the all process again?  
A1：If you have completed the above steps, you only need to replace the `.nb` model file after replacing the model. At the same time, you may need to modify the path of `.nb` file in the config file and change the category mapping file to be compatible the model .

Q2：How to change the test picture?  
A2：Replace the test image under debug folder with the image you want to test，and then repush to the Phone again.
