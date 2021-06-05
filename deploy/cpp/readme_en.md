# Server-side C++ inference


In this tutorial, we will introduce the detailed steps of deploying PaddleClas models on the server side.


## 1. Prepare the environment

### Environment

- Linux, docker is recommended.
- Windows, compilation based on `Visual Studio 2019 Community` is supported. In addition, you can refer to [How to use PaddleDetection to make a complete project](https://zhuanlan.zhihu.com/p/145446681) to compile by generating the `sln solution`.
- This document mainly introduces the compilation and inference of PaddleClas C++ in Linux environment.
- If you need to use the Inference Library in Windows environment, please refer to [The compilation tutorial in Windows](./docs/windows_vs2019_build.md) for detailed information.


### 1.1 Compile opencv

* First of all, you need to download the source code compiled package in the Linux environment from the opencv official website. Taking opencv3.4.7 as an example, the download and uncompress command are as follows.

```
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xf 3.4.7.tar.gz
```

Finally, you can see the folder of `opencv-3.4.7/` in the current directory.

* Compile opencv, the opencv source path (`root_path`) and installation path (`install_path`) should be set by yourself. Among them, `root_path` is the downloaded opencv source code path, and `install_path` is the installation path of opencv. In this case, the opencv source is `./opencv-3.4.7`.

```shell
cd ./opencv-3.4.7
export root_path=$PWD
export install_path=${root_path}/opencv3
```

* After entering the opencv source code path, you can compile it in the following way.


```shell
rm -rf build
mkdir build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON

make -j
make install
```

* After `make install` is completed, the opencv header file and library file will be generated in this folder for later PaddleClas source code compilation.

Take opencv3.4.7 for example, the final file structure under the opencv installation path is as follows. **NOTICE**:The following file structure may be different for different Versions of Opencv.

```
opencv3/
|-- bin
|-- include
|-- lib64
|-- share
```

### 1.2 Compile or download the Paddle Inference Library

* There are 2 ways to obtain the Paddle Inference Library, described in detail below.


#### 1.2.1 Compile from the source code
* If you want to get the latest Paddle Inference Library features, you can download the latest code from Paddle GitHub repository and compile the inference library from the source code.
* You can refer to [Paddle Inference Library](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/05_inference_deployment/inference/build_and_install_lib_en.html#build-from-source-code) to get the Paddle source code from github, and then compile To generate the latest inference library. The method of using git to access the code is as follows.


```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

* After entering the Paddle directory, the compilation method is as follows.

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

For more compilation parameter options, please refer to the official website of the Paddle C++ inference library:[https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/05_inference_deployment/inference/build_and_install_lib_en.html#build-from-source-code](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/05_inference_deployment/inference/build_and_install_lib_en.html#build-from-source-code).


* After the compilation process, you can see the following files in the folder of `build/paddle_inference_install_dir/`.

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

Among them, `paddle` is the Paddle library required for C++ prediction later, and `version.txt` contains the version information of the current inference library.



#### 1.2.2 Direct download and installation

* Different cuda versions of the Linux inference library (based on GCC 4.8.2) are provided on the
[Paddle Inference Library official website](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/guides/05_inference_deployment/inference/build_and_install_lib_en.html). You can view and select the appropriate version of the inference library on the official website.

* Please select the `develop` version.

* After downloading, use the following method to uncompress.

```
tar -xf paddle_inference.tgz
```

Finally you can see the following files in the folder of `paddle_inference/`.


## 2. Compile and run the demo

### 2.1 Export the inference model

* You can refer to [Model inference](../../tools/export_model.py)，export the inference model. After the model is exported, assuming it is placed in the `inference` directory, the directory structure is as follows.

```
inference/
|--cls_infer.pdmodel
|--cls_infer.pdiparams
```

**NOTICE**: Among them, `cls_infer.pdmodel` file stores the model structure information and the `cls_infer.pdiparams` file stores the model parameter information.The paths of the two files need to correspond to the parameters of `cls_model_path` and `cls_params_path` in the configuration file `tools/config.txt`.

### 2.2 Compile PaddleClas C++ inference demo


* The compilation commands are as follows. The addresses of Paddle C++ inference library, opencv and other Dependencies need to be replaced with the actual addresses on your own machines.

```shell
sh tools/build.sh
```

Specifically, the content in `tools/build.sh` is as follows.

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir
TENSORRT_DIR=your_tensorrt_lib_dir

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DDEMO_NAME=clas_system \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \

make -j
```

In the above parameters of command:

* `OPENCV_DIR` is the opencv installation path;

* `LIB_DIR` is the download (`paddle_inference` folder) or the generated Paddle Inference Library path (`build/paddle_inference_install_dir` folder);

* `CUDA_LIB_DIR` is the cuda library file path, in docker; it is `/usr/local/cuda/lib64`;

* `CUDNN_LIB_DIR` is the cudnn library file path, in docker it is `/usr/lib/x86_64-linux-gnu/`.

* `TENSORRT_DIR` is the tensorrt library file path，in dokcer it is `/usr/local/TensorRT6-cuda10.0-cudnn7/`，TensorRT is just enabled for GPU.

After the compilation is completed, an executable file named `clas_system` will be generated in the `build` folder.


### Run the demo
* First, please modify the `tools/config.txt` and `tools/run.sh`.

* Some key words in `tools/config.txt` is as follows.
  * use_gpu: Whether to use GPU.
  * gpu_id: GPU id.
  * gpu_mem：GPU memory.
  * cpu_math_library_num_threads：Number of thread for math library acceleration.
  * use_mkldnn：Whether to use mkldnn.
  * use_tensorrt: Whether to use tensorRT.
  * use_fp16：Whether to use Float16 (half precision), it is just enabled when use_tensorrt is set as 1.
  * cls_model_path: Model path of inference model.
  * cls_params_path: Params path of inference model.
  * resize_short_size：Short side length of the image after resize.
  * crop_size：Image size after center crop.


* Then execute the following command to complete the classification of an image.

```shell
sh tools/run.sh
```

* The detection results will be shown on the screen, which is as follows.

<div align="center">
    <img src="./docs/imgs/cpp_infer_result.png" width="600">
</div>

* In the above results,`class id` represents the id corresponding to the category with the highest confidence, and `score` represents the probability that the image belongs to that category.
