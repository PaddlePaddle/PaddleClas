# 安装说明

---

## 1.简介

本章将介绍如何安装PaddleClas及其依赖项.
有关模型库的基本信息请参考[README](https://github.com/PaddlePaddle/PaddleClas/blob/master/README.md)


## 2.安装PaddlePaddle

运行PaddleClas需要PaddlePaddle Fluid v1.7或更高版本。

pip安装最新GPU版本PaddlePaddle

```bash
pip install paddlepaddle-gpu --upgrade
```

或是从源码安装PaddlePaddle，具体参照[安装文档](http://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

使用以下命令验证

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```

查看PaddlePaddle版本

```bash
python -c "import paddle; print(paddle.__version__)"
```

注意：
- 从源码编译的PaddlePaddle版本号为0.0.0，请确保使用了Fluid v1.7之后的源码编译。
- PaddleClas基于PaddlePaddle高性能的分布式训练能力，若您从源码编译，请确保打开编译选项，**WITH_DISTRIBUTE=ON**。具体编译选项参考[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3)

**运行环境需求:**

- Python2（官方已不提供更新维护）或Python3 (windows系统仅支持Python3)
- CUDA >= 9.0
- cuDNN >= 5.0
- nccl >= 2.1.2


## 3.安装PaddleClas

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


## 4.数据集和预训练模型

PaddleClas加载PaddleClas/dataset/中数据进行训练，请参照[数据文档](./data.md)进行准备。
PaddleClas提供丰富的预训练模型，请参照[数据文档](./data.md)进行准备。


## 5.开始使用

请参照[开始使用](./getting_started.md)文档
