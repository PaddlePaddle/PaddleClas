# 安装说明

---
本章将介绍如何安装PaddleClas及其依赖项。


## 1. 安装PaddlePaddle

运行PaddleClas需要`PaddlePaddle 2.1`或更高版本。可以参考下面的步骤安装PaddlePaddle。

### 1.1 环境要求

- python 3.x
- cuda >= 10.1 (如果使用paddlepaddle-gpu)
- cudnn >= 7.6.4 (如果使用paddlepaddle-gpu)
- nccl >= 2.1.2 (如果使用分布式训练/评估)
- gcc >= 8.2

建议使用我们提供的docker运行PaddleClas，有关docker、nvidia-docker使用请参考[链接](https://www.runoob.com/docker/docker-tutorial.html)。

在cuda10.1时，建议显卡驱动版本大于等于418.39；在使用cuda10.2时，建议显卡驱动版本大于440.33，更多cuda版本与要求的显卡驱动版本可以参考[链接](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)。


如果不使用docker，可以直接跳过1.2部分内容，从1.3部分开始执行。


### 1.2 （建议）准备docker环境。第一次使用这个镜像，会自动下载该镜像，请耐心等待。

```
# 切换到工作目录下
cd /home/Projects
# 首次运行需创建一个docker容器，再次运行时不需要运行当前命令
# 创建一个名字为ppcls的docker容器，并将当前目录映射到容器的/paddle目录下

如果您希望在CPU环境下使用docker，使用docker而不是nvidia-docker创建docker，设置docker容器共享内存shm-size为8G，建议设置8G以上
sudo docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0 /bin/bash

如果希望使用GPU版本的容器，请运行以下命令创建容器。
sudo nvidia-docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash
```


您也可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

```
# ctrl+P+Q可退出docker 容器，重新进入docker 容器使用如下命令
sudo docker exec -it ppcls /bin/bash
```

### 1.3 通过pip安装PaddlePaddle

运行下面的命令，通过pip安装最新GPU版本PaddlePaddle

```bash
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

如果希望在CPU环境中使用PaddlePaddle，可以运行下面的命令安装PaddlePaddle。

```bash
pip3 install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
```

**注意：**
* 如果先安装了CPU版本的paddlepaddle，之后想切换到GPU版本，那么需要首先卸载CPU版本的paddle，再安装GPU版本的paddle，否则容易导致使用的paddle版本混乱。
* 您也可以从源码编译安装PaddlePaddle，请参照[PaddlePaddle 安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。


### 1.4 验证是否安装成功

使用以下命令可以验证PaddlePaddle是否安装成功。

```python
import paddle
paddle.utils.run_check()
```

查看PaddlePaddle版本的命令如下：

```bash
python3 -c "import paddle; print(paddle.__version__)"
```

注意：
- 从源码编译的PaddlePaddle版本号为0.0.0，请确保使用了PaddlePaddle 2.0及之后的源码编译。
- PaddleClas基于PaddlePaddle高性能的分布式训练能力，若您从源码编译，请确保打开编译选项，**WITH_DISTRIBUTE=ON**。具体编译选项参考[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3)。
- 在docker中运行时，为保证docker容器有足够的共享内存用于Paddle的数据读取加速，在创建docker容器时，请设置参数`--shm_size=8g`，条件允许的话可以设置为更大的值。
-


## 2. 安装PaddleClas

### 2.1 克隆PaddleClas模型库

```bash
git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.2
```

如果从github上网速太慢，可以从gitee下载，下载命令如下：

```bash
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
```

### 2.2 安装Python依赖库

Python依赖库在`requirements.txt`中给出，可通过如下命令安装：

```bash
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
