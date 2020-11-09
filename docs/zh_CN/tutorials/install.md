# 安装说明

---

## 一、简介

本章将介绍如何安装PaddleClas及其依赖项。


## 二、安装PaddlePaddle

运行PaddleClas需要PaddlePaddle 2.0rc或更高版本。请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

如果已经安装好了cuda、cudnn、nccl或者安装好了docker、nvidia-docker运行环境，可以pip安装最新GPU版本PaddlePaddle

```bash
pip install paddlepaddle-gpu --upgrade
```

也可以从源码编译安装PaddlePaddle，请参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

使用以下命令可以验证PaddlePaddle是否安装成功。

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```

查看PaddlePaddle版本的命令如下：

```bash
python -c "import paddle; print(paddle.__version__)"
```

注意：
- 从源码编译的PaddlePaddle版本号为0.0.0，请确保使用了Fluid v1.7之后的源码编译。
- PaddleClas基于PaddlePaddle高性能的分布式训练能力，若您从源码编译，请确保打开编译选项，**WITH_DISTRIBUTE=ON**。具体编译选项参考[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3)

**运行环境需求:**

- Python3 (当前只支持Linux系统)
- CUDA >= 9.0
- cuDNN >= 5.0
- nccl >= 2.1.2


## 三、安装PaddleClas

**克隆PaddleClas模型库：**

```
cd path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleClas/blob/master/requirements.txt)中给出，可通过如下命令安装：

```
pip install --upgrade -r requirements.txt
```

visualdl可能出现安装失败，请尝试

```
pip3 install --upgrade visualdl==2.0.0b3 -i https://mirror.baidu.com/pypi/simple

```

此外，visualdl目前只支持在python3下运行，因此如果希望使用visualdl，需要使用python3。
