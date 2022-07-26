# 环境准备

---
## 目录
- [1. 手动配置PaddlePaddle, PaddleClas环境](#1)
  - [1.1 安装 PaddlePaddle](#1.1)
    - [1.1.1 使用Paddle官方镜像](#1.1.1)
    - [1.1.2 在现有环境中安装paddle](#1.1.2)
    - [1.1.3 安装验证](#1.1.3)
  - [1.2 克隆 PaddleClas](#1.2)
  - [1.3 安装 Python 依赖库](#1.3)
- [2. 快速创建PaddlePaddle, PaddleClas环境](#2)


我们提供了两种配置PaddlePaddle、PaddleClas环境的方法，第一种需要基于 docker 手动配置，您可以根据提供的命令更灵活的配置您的环境，详情请见[1. 手动配置PaddlePaddle, PaddleClas环境](#1)。第二种方式是我们将 PaddlePaddle、PaddleClas 相关的环境已配置到一个 docker 镜像中，您可以直接拉取使用，详情请见[2. 快速创建PaddlePaddle, PaddleClas环境](#2)。

<a name='1'></a>
## 1. 手动配置PaddlePaddle, PaddleClas环境

<a name='1.1'></a>
### 1.1 安装PaddlePaddle
目前，**PaddleClas** 要求 **PaddlePaddle** 版本 `>=2.3`。
建议使用Paddle官方提供的 Docker 镜像运行 PaddleClas，有关 Docker、nvidia-docker 的相关使用教程可以参考[链接](https://www.runoob.com/Docker/Docker-tutorial.html)。

<a name='1.1.1'></a>

#### 1.1.1 使用Paddle官方镜像

* 切换到工作目录下，例如工作目录为`/home/Projects`，则运行命令: 

```shell
cd /home/Projects
```

* 创建 docker 容器

下述命令会创建一个名为 ppcls 的 Docker 容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

```shell
# 对于 GPU 用户
sudo nvidia-docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it registry.baidubce.com/paddlepaddle/paddle:2.3.0-gpu-cuda10.2-cudnn7 /bin/bash

# 对于 CPU 用户
sudo docker run --name ppcls -v $PWD:/paddle --shm-size=8G --network=host -it paddlepaddle/paddle:2.3.0-gpu-cuda10.2-cudnn7 /bin/bash
```

**注意**：
* 首次使用该镜像时，下述命令会自动下载该镜像文件，下载需要一定的时间，请耐心等待；
* 上述命令会创建一个名为 ppcls 的 Docker 容器，之后再次使用该容器时无需再次运行该命令；
* 参数 `--shm-size=8G` 将设置容器的共享内存为 8 G，如机器环境允许，建议将该参数设置较大，如 `64G`；
* 您也可以访问 [DockerHub](https://hub.Docker.com/r/paddlepaddle/paddle/tags/) ，手动选择需要的镜像；
* 退出/进入 docker 容器：
    * 在进入 Docker 容器后，可使用组合键 `Ctrl + P + Q` 退出当前容器，同时不关闭该容器；
    * 如需再次进入容器，可使用下述命令：

    ```shell
    sudo Docker exec -it ppcls /bin/bash
    ```
<a name='1.1.2'></a>
#### 1.1.2 在现有环境中安装paddle
您也可以用pip或conda直接安装paddle，详情请参考官方文档中的[快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/docker/linux-docker.html)部分。

<a name='1.1.3'></a>
#### 1.1.3 安装验证
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
- 从源码编译的 PaddlePaddle 版本号为 `0.0.0`，请确保使用 PaddlePaddle 2.3 及之后的源码进行编译；
- PaddleClas 基于 PaddlePaddle 高性能的分布式训练能力，若您从源码编译，请确保打开编译选项 `WITH_DISTRIBUTE=ON`。具体编译选项参考 [编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyixuanxiangbiao)；
- 在 Docker 中运行时，为保证 Docker 容器有足够的共享内存用于 Paddle 的数据读取加速，在创建 Docker 容器时，请设置参数 `--shm-size=8g`，条件允许的话可以设置为更大的值。


<a name='1.2'></a>

### 1.2 克隆 PaddleClas

从 GitHub 下载：

```shell
git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.4
```

如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：

```shell
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.4
```
<a name='1.3'></a>

### 1.3 安装 Python 依赖库

PaddleClas 的 Python 依赖库在 `requirements.txt` 中给出，可通过如下命令安装：

```shell
pip install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

<a name='2'></a>
## 2. 快速创建PaddlePaddle, PaddleClas环境

我们也提供了包含最新 PaddleClas 代码的 docker 镜像，并预先安装好了所有的环境和库依赖，您只需要**拉取并运行docker镜像**，无需其他任何额外操作，即可开始享用 PaddleClas 的所有功能。

在[Docker Hub](https://hub.docker.com/repository/docker/paddlecloud/paddleclas)中获取这些镜像及相应的使用指南，包括CPU、GPU、ROCm 版本。

如果您对自动化制作docker镜像感兴趣，或有自定义需求，请访问[PaddlePaddle/PaddleCloud](https://github.com/PaddlePaddle/PaddleCloud/tree/main/tekton)做进一步了解。

**备注**：当前的镜像中的 PaddleClas 代码默认使用最新的 release/2.4 分支。


