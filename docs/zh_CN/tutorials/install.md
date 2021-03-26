# 安装说明

---

## 一、简介

本章将介绍如何安装PaddleClas及其依赖项。


## 二、安装PaddlePaddle

运行PaddleClas需要`PaddlePaddle 2.0`或更高版本。请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

### 通过pip安装PaddlePaddle

如果已经安装好了cuda、cudnn、nccl或者安装好了docker、nvidia-docker运行环境，可以pip安装最新GPU版本PaddlePaddle

```bash
pip3 install paddlepaddle-gpu --upgrade -i https://mirror.baidu.com/pypi/simple
```

如果希望在CPU环境中使用PaddlePaddle，可以运行下面的命令安装PaddlePaddle。

```bash
pip3 install paddlepaddle --upgrade -i https://mirror.baidu.com/pypi/simple
>>>>>>> upstream/develop
```

### 源码编译PaddlePaddle

您也可以从源码编译安装PaddlePaddle，请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。


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

**运行环境需求:**

- Python3
- CUDA >= 9.0
- cuDNN >= 7.6.4
- nccl >= 2.1.2


## 三、安装PaddleClas

**克隆PaddleClas模型库：**

```bash
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
```

如果从github上网速太慢，可以从gitee下载，下载命令如下：

```bash
git clone https://gitee.com/paddlepaddle/PaddleClas.git -b develop
```



**安装Python依赖库：**

Python依赖库在`requirements.txt`中给出，可通过如下命令安装：

```bash
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

visualdl可能出现安装失败，请尝试

```bash
pip3 install --upgrade visualdl -i https://mirror.baidu.com/pypi/simple
```

此外，visualdl目前只支持在python3下运行，因此如果希望使用visualdl，需要使用python3。
