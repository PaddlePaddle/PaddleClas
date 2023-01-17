# 基于PaddleClas的DALI开发实践

- [1. 简介](#1-简介)
- [2. 环境准备](#2-环境准备)
  - [2.1 安装DALI](#21-安装dali)
- [3. 基本概念介绍](#3-基本概念介绍)
  - [3.1 Operator](#31-operator)
  - [3.2 Device](#32-device)
  - [3.3 DataNode](#33-datanode)
  - [3.4 Pipeline](#34-pipeline)
- [4. 开发实践](#4-开发实践)
  - [4.1 开发与接入流程](#41-开发与接入流程)
  - [4.2 RandomFlip](#42-randomflip)
    - [4.2.1 继承DALI已有类](#421-继承dali已有类)
    - [4.2.2 重载 \_\_init\_\_ 方法](#422-重载-__init__-方法)
    - [4.2.3 重载 \_\_call\_\_ 方法](#423-重载-__call__-方法)
  - [4.3 RandomRotation](#43-randomrotation)
    - [4.3.1 继承DALI已有类](#431-继承dali已有类)
    - [4.3.2 重载 \_\_init\_\_ 方法](#432-重载-__init__-方法)
    - [4.3.3 重载 \_\_call\_\_ 方法](#433-重载-__call__-方法)
- [5. FAQ](#5-faq)

## 1. 简介
NVIDIA **Da**ta Loading **Li**brary (DALI) 是由 NVIDIA 开发的一套高性能数据预处理开源代码库，其提供了许多优化后的预处理算子，能很大程度上减少数据预处理耗时，非常适合在深度学习任务中使用。具体地，DALI 通过将大部分的数据预处理转移到 GPU 来解决 CPU 瓶颈问题。此外，DALI 编写了配套的高效执行引擎，最大限度地提高输入管道的吞吐量。

实际上 DALI 提供了不同粒度的图像、音频处理算子与随机数算子，这一特点基本上满足了大部分用户的需求，即用户只需在python侧进行开发，而不需要接触更为底层更复杂的C++代码。

本文档作为DALI的开发入门实践教程，在 PaddleClas 的文件结构与代码逻辑基础上，来介绍如何利用已有的DALI算子，根据自己的需求进行python侧的二次开发，以减少初学者的学习成本，提升模型训练效率。

## 2. 环境准备

### 2.1 安装DALI
进入DALI的官方教程 **[DALI-installtion](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html#)**

首先运行 `nvcc -V` 查看运行环境中的CUDA版本
```log
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```
可以看到具体版本号是 `release 10.2, V10.2.89`，因此接下来需安装 CUDA10.2 的DALI包

```shell
# for CUDA10.2
python3.7 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
```
其余版本的CUDA请将上述命令末尾的 `cuda102` 改成对应的CUDA版本即可，如 CUDA11.0就改成 `cuda110`。DALI的具体支持设备可查看 **[DALI-support_matrix](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/support_matrix.html)**

## 3. 基本概念介绍

### 3.1 Operator
DALI 预处理过程的基本单位是 Operator(算子)，PaddleClas 的 `operators.py` 设计逻辑与之类似，是一种较为通用的设计方式。DALI 提供了多种算子供用户根据具体需求使用，如 `nvidia.dali.ops.decoders.Image`（图像解码算子）， `nvidia.dali.ops.Flip`（水平、垂直翻转算子），以及稍复杂的融合算子 `nvidia.dali.ops.decoders.ImageRandomCrop`（图像解码+随机裁剪的融合算子）。同时 DALI 也提供了一些随机数算子以在图像增强中加入随机性，如 `nvidia.dali.ops.random.CoinFlip`（伯努利分布随机数算子），`nvidia.dali.ops.random.Uniform`（均匀分布随机数算子）。

详细的算子库结构可以查看 **[DALI-operators](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#modules)**

### 3.2 Device
DALI 可以选择将数据预处理放到GPU上进行，因此绝大部分算子自身具有 `device` 这一参数，以在不同的设备上运行。
而 DALI 将具体情况分为了三种：
1. `cpu` - 接受在CPU上的输入，且输出在CPU上。
2. `mixed` - 接受在CPU上的输入，但输出在GPU上。
3. `gpu` - 接受在GPU上的输入，且输出在GPU上。

因此可以指定每个算子的处理时的设备，加快并行效率，减少阻塞耗时。

### 3.3 DataNode
与常见的深度学习框架中静态图的设计思路（如 tensorflow）相似，DALI 的 Operator 输入和输出一般是一个或多个在CPU/GPU上的数据，被称为 **DataNode**，这些 DataNode 在多个 Operator 中被有顺序地处理、传递，直到成为最后一个 Operator 的输出，然后才被用户获取并输入到网络模型中去。

### 3.4 Pipeline
从用户读取、解析给定的图片路径文件（如`.txt`格式文件）开始，到解码出图片，再到使用一个或多个Operator对图片进行预处理，最后返回处理完毕的图像（一般为Tensor格式）。这一整个过程称之为 **Pipeline**，当准备好需要的 Operator(s) 之后，就需要开始编写这一部分的代码，将 数据读取、预处理Operator(s) 组装成一个 Pipeline。如果将 Pipeline 当作是一个计算图，那么 Operator 和 DataNode 都是图中的结点，如下图所示。

![DALI-pipeline](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/_images/two_readers.svg)

## 4. 开发实践
本章节希望通过一个简单的例子和一个稍复杂的例子，介绍如何基于 DALI 提供的算子，在python侧进行二次开发，以满足用户实际需要。

### 4.1 开发与接入流程
1. 在 `ppcls/data/preprocess/ops/dali_operators.py` 中开发python侧DALI算子的代码。
2. 在 `ppcls/data/preprocess/ops/dali.py` 开头处 import 导入开发好的算子类，并在 `convert_cfg_to_dali` 函数内参照其它算子配置转换逻辑，为添加的算子也加入对应的配置转换逻辑。
3. （可选）如果开发的算子属于 fused operator，则还需在 `ppcls/data/preprocess/ops/dali.py` 的 `build_dali_transforms` 函数内，参照已有融合算子逻辑，添加新算子对应的融合逻辑。
4. （可选）如果开发的是 External Source 类的 sampler 算子，可参照已有的 `ExternalSource_RandomIdentity` 代码进行开发，并在添加对应调用逻辑。实际上 External Source 类可视作对原有的Dataset和Sampler代码进行合并。

### 4.2 RandomFlip
以 PaddleClas 已有的 [RandFlipImage](../../../../ppcls/data/preprocess/ops/operators.py#L499) 算子为例，我们希望在使用DALI训练时，将其转换为对应的 DALI 算子，且同样具备 **按指定的 `prob` 概率进行 指定的水平 or 垂直翻转**

#### 4.2.1 继承DALI已有类
DALI 已经提供了简单的翻转算子 [`nvidia.dali.ops.Flip`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#nvidia.dali.ops.Flip)，其通过 `horizontal` 与 `vertical` 参数来分别控制是否对图像进行水平、垂直翻转。但是其缺少随机性，无法直接按照一定概率进行翻转或不翻转，因此我们需要继承这个翻转类，并重载其 `__init__` 方法和 `__call__` 方法。继承代码如下所示：

```python
import nvidia.dali.ops as ops

class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        ...

    def __call__(self, data, **kwargs):
        ...
```

#### 4.2.2 重载 \_\_init\_\_ 方法
我们需要在构造算子时加入随机参数来控制是否翻转，因此仿照普通 `RandFlipImage`算子的逻辑，在继承类的初始化方法中加入参数 `prob`，同理再加入 `flip_code` 用于控制水平、垂直翻转。

由于每一次执行我们都需要生成一个随机数（此处用0或1表示），代表是否在翻转轴上进行翻转，因此我们实例化一个 `ops.random.CoinFlip` 来作为随机数生成器（实例化对象为下方代码中的 `self.rng`），同理我们也需要记录翻转轴参数 `flip_code`，以供 `__call__` 方法中调用。

修改后代码如下所示：
```python
class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", prob=0.5, flip_code=1, **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        self.flip_code = flip_code
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        ...
```

#### 4.2.3 重载 \_\_call\_\_ 方法
有了 `self.rng` 和 `self.flip_code`，我们就能在每次调用的 `__call__` 方法内部，加入随机性并控制方向。首先调用 `self.rng()` 的 `__call__` 方法，生成一个0或1的随机整数，0代表不进行翻转，1代表进行翻转；然后根据 `self.flip_code` ，将这个随机整数作为父类 `__call__` 方法的 `horizontal` 或 `vertical` 参数，调用父类的 `__call__` 方法完成翻转。这样就完成了一个简单的自定义DALI RandomFlip 算子的编写。完整代码如下所示：
```python
class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", prob=0.5, flip_code=1, **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        self.flip_code = flip_code
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_flip = self.rng()
        if self.flip_code == 1:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=0, **kwargs)
        elif self.flip_code == 1:
            return super(RandFlipImage, self).__call__(
                data, horizontal=0, vertical=do_flip, **kwargs)
        else:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=do_flip, **kwargs)
```

### 4.3 RandomRotation
以 PaddleClas 已有的 [RandomRotation](../../../../ppcls/data/preprocess/ops/operators.py#L684) 算子为例，我们希望在使用DALI训练时，将其转换为对应的 DALI 算子，且同样具备 **按指定的参数与角度进行随机旋转**

#### 4.3.1 继承DALI已有类
DALI 已经提供了简单的翻转算子 [`nvidia.dali.ops.Rotate`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#nvidia.dali.ops.Rotate)，其通过 `angle`、`fill_value`、`interp_type` 等参数控制旋转的角度、填充值以及插值方式。但是其缺少一定的随机性，此我们需要继承这个旋转类，并重载其 `__init__` 方法和 `__call__` 方法。继承代码如下所示：

```python
import nvidia.dali.ops as ops

class RandomRotation(ops.Rotate):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandomRotation, self).__init__(*kargs, device=device, **kwargs)
        ...

    def __call__(self, data, **kwargs):
        ...
```

#### 4.3.2 重载 \_\_init\_\_ 方法
我们需要在构造算子时加入随机参数来控制是否翻转，因此仿照普通 `RandomRotation` 算子的逻辑，在继承类的初始化方法中加入参数 `prob`，同理再加入 `angle` 用于控制旋转角度。

由于每一次执行我们都需要生成一个随机数（此处用0或1表示），代表是否进行随机旋转，因此我们实例化一个 `ops.random.CoinFlip` 来作为随机数生成器（实例化对象为下方代码中的 `self.rng`）。除此之外我们还需要实例化一个随机数生成器来作为实际旋转时的角度（实例化对象为下方代码中的 `self.rng_angle`），由于角度是一个均匀分布而不是伯努利分布，因此需要使用 `random.Uniform` 这个类。

修改后代码如下所示：
```python
class RandomRotation(ops.Rotate):
    def __init__(self, *kargs, device="cpu", prob=0.5, angle=0, **kwargs):
        super(RandomRotation, self).__init__(*kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)
        self.rng_angle = ops.random.Uniform(range=(-angle, angle))

    def __call__(self, data, **kwargs):
        ...
```

#### 4.3.3 重载 \_\_call\_\_ 方法
有了以上的一些变量，根据 `operators.py` 里 `RandomRotation` 的逻辑，仿照 [RandomFlip-重载__call__方法](#413-重载-__call__-方法) 的写法进行代码编写，就能得到完整代码，如下所示：
```python
class RandomRotation(ops.Rotate):
    def __init__(self, *kargs, device="cpu", prob=0.5, angle=0, **kwargs):
        super(RandomRotation, self).__init__(*kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)
        discrete_angle = list(range(-angle, angle + 1))
        self.rng_angle = ops.random.Uniform(values=discrete_angle)

    def __call__(self, data, **kwargs):
        do_rotate = self.rng()
        angle = self.rng_angle()
        flip_data = super(RandomRotation, self).__call__(
            data,
            angle=fn.cast(
                do_rotate, dtype=types.FLOAT) * angle,
            keep_size=True,
            fill_value=0,
            **kwargs)
        return flip_data
```

具体每个参数的含义，如`angle`、`keep_size`、`fill_value`等，可以查看DALI对应算子的文档

## 5. FAQ

- **Q**：是否所有算子都能以继承-重载的方式改写成DALI算子？
  **A**：具体视算子本身的执行逻辑而定，如 `RandomErasing` 算子实际上比较难在python侧转换成DALI算子，尽管DALI有一个对应的 [random_erasing Demo](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/erase.html?highlight=erase)，但其实际执行中的随机逻辑与 `RandomErasing` 存在一定差异，无法保证等价转换。可以尝试使用 [python_function](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.python_function.html?highlight=python_function#nvidia.dali.fn.python_function) 来接入python实现的数据增强

- **Q**：使用DALI训练模型的最终精度与不使用DALI不同？
  **A**：由于DALI底层实现是NVIDIA官方编写的代码，而operators.py中调用的是cv2、Pillow库，可能存在无法避免的细微差异，如同样的插值方法，实现存在不同。因此只能尽量从执行逻辑、参数、随机数分布上进行等价转换，而无法做到完全一致。如果出现较大diff，可以检查转换来的DALI算子代码执行逻辑、参数、随机数分布是否存在问题，也可以将读取结果可视化检查。另外需要注意的是如果使用DALI的数据预处理接口进行训练，那么为了获得最佳的精度，也应该用DALI的数据预处理接口进行测试，否则可能会造成精度下降。

- **Q**：如果模型使用比较复杂的Sampler如PKsampler该如何改写呢？
  **A**：从开发成本考虑，目前比较推荐的方法([#issue 4407](https://github.com/NVIDIA/DALI/issues/4407#issuecomment-1298132180))是使用DALI官方提供的 [`External Source Operator`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/external_input.html) 完成自定义Sampler的编写，实际上 [dali.py](../../../../ppcls/data/dataloader/dali.py) 也提供了基于 `External Source Operator` 的 `PKSampler` 的实现 `ExternalSource_RandomIdentity`。
