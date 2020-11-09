# Installation

---

## Introducation

This document introduces how to install PaddleClas and its requirements.

## Install PaddlePaddle

Python 3.5, CUDA 9.0, CUDNN7.0 nccl2.1.2 and later version are required at first, For now, PaddleClas only support training on the GPU device. Please follow the instructions in the [Installation](http://www.paddlepaddle.org.cn/install/quick) if the PaddlePaddle on the device is lower than PaddlePaddle-2.0rc.

Install PaddlePaddle

```bash
pip install paddlepaddle-gpu --upgrade
```

or compile from source code, please refer to [Installation](http://www.paddlepaddle.org.cn/install/quick).

Verify Installation

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```

Check PaddlePaddle version：

```bash
python -c "import paddle; print(paddle.__version__)"
```

Note:
- Make sure the compiled version is later than v1.7
- Indicate **WITH_DISTRIBUTE=ON** when compiling, Please refer to [Instruction](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#id3) for more details.


## Install PaddleClas

**Clone PaddleClas: **

```
cd path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```

**Install requirements**


```
pip install --upgrade -r requirements.txt
```

If the install process of visualdl failed, you can try the following commands.

```
pip3 install --upgrade visualdl==2.0.0b3 -i https://mirror.baidu.com/pypi/simple

```

What's more, visualdl is just supported in python3, so python3 is needed if you want to use visualdl.
