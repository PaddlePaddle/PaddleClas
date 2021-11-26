# 安装 PaddlePaddle

---
## 目录

- [1. 环境要求](#1)
- [2.（建议）使用 Docker 环境](#2)
- [3. 通过 pip 安装 PaddlePaddle](#3)
- [4. 验证安装](#4)

目前，**PaddleClas** 要求 **PaddlePaddle** 版本 `>=2.0`。建议使用我们提供的 Docker 运行 PaddleClas，有关 Docker、nvidia-docker 的相关使用教程可以参考[链接](https://www.runoob.com/Docker/Docker-tutorial.html)。如果不使用 Docker，可以直接跳过 [2.（建议）使用 Docker 环境](#2) 部分内容，从 [3. 通过 pip 安装 PaddlePaddle](#3) 部分开始。

<a name='1'></a>

## 1. 环境要求

**版本要求**：
- python 3.x
- CUDA >= 10.1（如果使用 `paddlepaddle-gpu`）
- cuDNN >= 7.6.4（如果使用 `paddlepaddle-gpu`）
- nccl >= 2.1.2（如果使用分布式训练/评估）
- gcc >= 8.2

**建议**：
* 当 CUDA 版本为 10.1 时，显卡驱动版本 `>= 418.39`；
* 当 CUDA 版本为 10.2 时，显卡驱动版本 `>= 440.33`；
* 更多 CUDA 版本与要求的显卡驱动版本可以参考[链接](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)。

<a name="2"></a>

## 2.（建议）使用 Docker 环境

* 切换到工作目录下

```shell
cd /home/Projects
```

* 创建 docker 容器

下述命令会创建一个名为 ppcls 的 Docker 容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

```shell
# 对于 GPU 用户
sudo nvidia-docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash

# 对于 CPU 用户
sudo docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.1.0 /bin/bash
```

**注意**：
* 首次使用该镜像时，下述命令会自动下载该镜像文件，下载需要一定的时间，请耐心等待；
* 上述命令会创建一个名为 ppcls 的 Docker 容器，之后再次使用该容器时无需再次运行该命令；
* 参数 `--shm-size=8G` 将设置容器的共享内存为 8 G，如机器环境允许，建议将该参数设置较大，如 `64G`；
* 您也可以访问 [DockerHub](https://hub.Docker.com/r/paddlepaddle/paddle/tags/) 获取与您机器适配的镜像；
* 退出/进入 docker 容器：
    * 在进入 Docker 容器后，可使用组合键 `Ctrl + P + Q` 退出当前容器，同时不关闭该容器；
    * 如需再次进入容器，可使用下述命令：

    ```shell
    sudo Docker exec -it ppcls /bin/bash
    ```

<a name="3"></a>

## 3. 通过 pip 安装 PaddlePaddle

可运行下面的命令，通过 pip 安装最新版本 PaddlePaddle：

```bash
# 对于 CPU 用户
pip3 install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple

# 对于 GPU 用户
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

**注意：**
* 如果先安装了 CPU 版本的 PaddlePaddle，之后想切换到 GPU 版本，那么需要使用 pip 先卸载 CPU 版本的 PaddlePaddle，再安装 GPU 版本的 PaddlePaddle，否则容易导致 PaddlePaddle 冲突。
* 您也可以从源码编译安装 PaddlePaddle，请参照 [PaddlePaddle 安装文档](http://www.paddlepaddle.org.cn/install/quick) 中的说明进行操作。

<a name='4'></a>
## 4. 验证安装

使用以下命令可以验证 PaddlePaddle 是否安装成功。

```python
import paddle
paddle.utils.run_check()
```

查看 PaddlePaddle 版本的命令如下：

```bash
python -c "import paddle; print(paddle.__version__)"
```

**注意**：
- 从源码编译的 PaddlePaddle 版本号为 `0.0.0`，请确保使用 PaddlePaddle 2.0 及之后的源码进行编译；
- PaddleClas 基于 PaddlePaddle 高性能的分布式训练能力，若您从源码编译，请确保打开编译选项 `WITH_DISTRIBUTE=ON`。具体编译选项参考 [编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyixuanxiangbiao)；
- 在 Docker 中运行时，为保证 Docker 容器有足够的共享内存用于 Paddle 的数据读取加速，在创建 Docker 容器时，请设置参数 `--shm_size=8g`，条件允许的话可以设置为更大的值。
