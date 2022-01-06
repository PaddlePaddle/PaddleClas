# 服务器端C++预测

本教程将介绍在服务器端部署PP-ShiTU的详细步骤。

## 目录

- [1.准备环境](#1)
  - [1.1 升级cmake](#1.1)
  - [1.2 编译opencv库](#1.2)
  - [1.3 下载或者编译Paddle预测库](#1.3)
    - [1.3.1 预测库源码编译](#1.3.1)
    - [1.3.2 直接下载安装](#1.3.2)
  - [1.4 安装faiss库](#1.4)
- [2.代码编译](#2)
- [3.运行demo](#3)
- [4.使用自己模型](#4)

<a name="1"></a>

## 1. 准备环境

### 运行准备
- Linux环境，推荐使用ubuntu docker。

<a name="1.1"></a>

### 1.1 升级cmake

由于依赖库编译需要较高版本的cmake，因此，第一步首先将cmake升级。

- 下载最新版本cmake

```shell
# 当前版本最新为3.22.0，根据实际情况自行下载，建议最新版本
wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz
tar xf cmake-3.22.0.tar.gz
```

最终可以在当前目录下看到`cmake-3.22.0/`的文件夹。

- 编译cmake，首先设置came源码路径(`root_path`)以及安装路径(`install_path`)，`root_path`为下载的came源码路径，`install_path`为came的安装路径。在本例中，源码路径即为当前目录下的`cmake-3.22.0/`。

```shell
cd ./cmake-3.22.0
export root_path=$PWD
export install_path=${root_path}/cmake
```

- 然后在cmake源码路径下，按照下面的方式进行编译

```shell
./bootstrap --prefix=${install_path}
make -j
make install
```

- 设置环境变量

```shell
export PATH=${install_path}/bin:$PATH
#检查是否正常使用
cmake --version
```

此时，cmake就可以使用了

<a name="1.2"></a>

### 1.2 编译opencv库

* 首先需要从opencv官网上下载在Linux环境下源码编译的包，以3.4.7版本为例，下载及解压缩命令如下：

```
wget https://github.com/opencv/opencv/archive/3.4.7.tar.gz
tar -xvf 3.4.7.tar.gz
```

最终可以在当前目录下看到`opencv-3.4.7/`的文件夹。

* 编译opencv，首先设置opencv源码路径(`root_path`)以及安装路径(`install_path`)，`root_path`为下载的opencv源码路径，`install_path`为opencv的安装路径。在本例中，源码路径即为当前目录下的`opencv-3.4.7/`。

```shell
cd ./opencv-3.4.7
export root_path=$PWD
export install_path=${root_path}/opencv3
```

* 然后在opencv源码路径下，按照下面的方式进行编译。

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

* `make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的PaddleClas代码编译。

以opencv3.4.7版本为例，最终在安装路径下的文件结构如下所示。**注意**：不同的opencv版本，下述的文件结构可能不同。

```
opencv3/
|-- bin
|-- include
|-- lib64
|-- share
```

<a name="1.3"></a>

### 1.3 下载或者编译Paddle预测库

* 有2种方式获取Paddle预测库，下面进行详细介绍。

<a name="1.3.1"></a>

#### 1.3.1 预测库源码编译

* 如果希望获取最新预测库特性，可以从Paddle github上克隆最新代码，源码编译预测库。
* 可以参考[Paddle预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)的说明，从github上获取Paddle代码，然后进行编译，生成最新的预测库。使用git获取代码方法如下。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

* 进入Paddle目录后，使用如下方法编译。

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

更多编译参数选项可以参考[Paddle C++预测库官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#id16)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是之后进行C++预测时所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

<a name="1.3.2"></a>

#### 1.3.2 直接下载安装

* [Paddle预测库官网](https://paddle-inference.readthedocs.io/en/latest/user_guides/download_lib.html)上提供了不同cuda版本的Linux预测库，可以在官网查看并选择合适的预测库版本，注意必须选择`develop`版本。

  以`https://paddle-inference-lib.bj.bcebos.com/2.1.1-gpu-cuda10.2-cudnn8.1-mkl-gcc8.2/paddle_inference.tgz`的`develop`版本为例，使用下述命令下载并解压：


```shell
wget https://paddle-inference-lib.bj.bcebos.com/2.1.1-gpu-cuda10.2-cudnn8.1-mkl-gcc8.2/paddle_inference.tgz

tar -xvf paddle_inference.tgz
```


最终会在当前的文件夹中生成`paddle_inference/`的子文件夹。

<a name="1.4"></a>

### 1.4 安装faiss库

```shell
 # 下载 faiss
 git clone https://github.com/facebookresearch/faiss.git
 cd faiss
 export faiss_install_path=$PWD/faiss_install
 cmake -B build . -DFAISS_ENABLE_PYTHON=OFF  -DCMAKE_INSTALL_PREFIX=${faiss_install_path}
 make -C build -j faiss
 make -C build install
```

在安装`faiss`前，请安装`openblas`，`ubuntu`系统中安装命令如下：

```shell
apt-get install libopenblas-dev
```

注意本教程以安装faiss cpu版本为例，安装时请参考[faiss](https://github.com/facebookresearch/faiss)官网文档，根据需求自行安装。

<a name="2"></a>

## 2. 代码编译

编译命令如下，其中Paddle C++预测库、opencv等其他依赖库的地址需要换成自己机器上的实际地址。同时，编译过程中需要下载编译`yaml-cpp`等C++库，请保持联网环境。


```shell
sh tools/build.sh
```

具体地，`tools/build.sh`中内容如下，请根据具体路径修改。

```shell
OPENCV_DIR=${opencv_install_dir}
LIB_DIR=${paddle_inference_dir}
CUDA_LIB_DIR=/usr/local/cuda/lib64
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
FAISS_DIR=${faiss_install_dir}
FAISS_WITH_MKL=OFF

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DUSE_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DFAISS_DIR=${FAISS_DIR} \
    -DFAISS_WITH_MKL=${FAISS_WITH_MKL}

make -j
cd ..
```

上述命令中，

* `OPENCV_DIR`为opencv编译安装的地址（本例中为`opencv-3.4.7/opencv3`文件夹的路径）；
* `LIB_DIR`为下载的Paddle预测库（`paddle_inference`文件夹），或编译生成的Paddle预测库（`build/paddle_inference_install_dir`文件夹）的路径；
* `CUDA_LIB_DIR`为cuda库文件地址，在docker中为`/usr/local/cuda/lib64`；
* `CUDNN_LIB_DIR`为cudnn库文件地址，在docker中为`/usr/lib/x86_64-linux-gnu/`。
* `TENSORRT_DIR`是tensorrt库文件地址，在dokcer中为`/usr/local/TensorRT6-cuda10.0-cudnn7/`，TensorRT需要结合GPU使用。
* `FAISS_DIR`是faiss的安装地址
* `FAISS_WITH_MKL`是指在编译faiss的过程中，是否使用了mkldnn，本文档中编译faiss，没有使用，而使用了openblas，故设置为`OFF`，若使用了mkldnn，则为`ON`.


在执行上述命令，编译完成之后，会在当前路径下生成`build`文件夹，其中生成一个名为`pp_shitu`的可执行文件。

<a name="3"></a>

## 3. 运行demo

- 请参考[识别快速开始文档](../../docs/zh_CN/quick_start/quick_start_recognition.md)，下载好相应的 轻量级通用主体检测模型、轻量级通用识别模型及瓶装饮料测试数据并解压。

  ```shell
  mkdir models
  cd models
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
  tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
  cd ..

  mkdir data
  cd data
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar
  tar -xf drink_dataset_v1.0.tar
  cd ..
  ```

- 将相应的yaml文件拷到当前文件夹下

  ```shell
  cp ../configs/inference_drink.yaml .
  ```

- 将`inference_drink.yaml`中的相对路径，改成基于本目录的路径或者绝对路径。涉及到的参数有

  - Global.infer_imgs ：此参数可以是具体的图像地址，也可以是图像集所在的目录
  - Global.det_inference_model_dir ： 检测模型存储目录
  - Global.rec_inference_model_dir ： 识别模型存储目录
  - IndexProcess.index_dir ： 检索库的存储目录，在示例中，检索库在下载的demo数据中。

- 字典转换

  由于python的检索库的字典，使用`pickle`进行的序列化存储，导致C++不方便读取，因此进行转换

  ```shell
  python tools/transform_id_map.py -c inference_drink.yaml
  ```

  转换成功后，在`IndexProcess.index_dir`目录下生成`id_map.txt`，方便c++ 读取。

- 执行程序

  ```shell
  ./build/pp_shitu -c inference_drink.yaml
  # or
  ./build/pp_shitu -config inference_drink.yaml
  ```

  若对图像集进行检索，则可能得到，如下结果。注意，此结果只做展示，具体以实际运行结果为准。

  同时，需注意的是，由于opencv 版本问题，会导致图像在预处理的过程中，resize产生细微差别，导致python 和c++结果，轻微不同，如bbox相差几个像素，检索结果小数点后3位diff等。但不会改变最终检索label。

  ![](../../docs/images/quick_start/shitu_c++_result.png)

<a name="4"></a>

## 4.  使用自己模型

使用自己训练的模型，可以参考[模型导出](../../docs/zh_CN/inference_deployment/export_model.md)，导出`inference model`，用于模型预测。

同时注意修改`yaml`文件中具体参数。
