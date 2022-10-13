# 基于PaddleClas的DALI开发实战

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
DALI 预处理过程的基本单位是 Operator(算子)，PaddleClas 的 `operators.py` 设计逻辑与之类似，是一种较为方便且通用的设计方式。DALI 提供了多种算子供用户根据具体需求使用，如 `nvidia.dali.ops.decoders.Image`（图像解码算子）， `nvidia.dali.ops.Flip`（水平、垂直翻转算子），以及稍复杂的 `nvidia.dali.ops.decoders.ImageRandomCrop`（图像解码+随机裁剪的融合算子）。同时 DALI 也提供了一些随机数算子以支持图像增强中的随机性这一重要特性，如 `nvidia.dali.ops.random.CoinFlip`（二项分布随机数算子），`nvidia.dali.ops.random.Uniform`（均匀分布随机数算子）。

详细的算子库结构可以查看 **[DALI-operators](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#modules)**

### 3.2 Device
DALI 可以选择将部分数据预处理放到GPU上进行，因此绝大部分算子自身具有 `device` 这一参数，以在不同的设备上运行。
而 DALI 将具体情况分为了三种：
1. `cpu` - 接受在CPU上的输入，且输出在CPU上。
2. `mixed` - 接受CPU上的输入，但输出在GPU上。
3. `gpu` - 接受在GPU上的输入，且输出在GPU上。

以此将不同算子的处理流程放置在不同的设备上，加快并行效率，减少阻塞耗时。

### 3.3 DataNode
与常见的深度学习框架中静态图的设计思路（如 tensorflow）相似，DALI 的 Operator 输入和输出一般是一个或多个在CPU/GPU上的数据，被称为 **DataNode**，这些 DataNode 在多个 Operator 中被有顺序地处理流动，直到成为最后一个 Operator 的输出，然后才被用户拿来输入到实际模型中去。

### 3.4 Pipeline
从用户读取、解析给定的图片路径文件（如`.txt`格式文件）开始，到从解析出的路径字符串中解码出图片，再到使用一个或多个Operator对图片进行预处理，最后返回处理完毕的图像（一般为张量格式）。这一整个过程称之为 **Pipeline**，当准备好需要的 Operator(s) 之后，就需要开始编写这一部分的代码，将 数据读取、预处理Operator(s) 组装成一个 Pipeline。如果将 Pipeline 当作是一个计算图，那么 Operator 和 DataNode 都是图中的结点，如下图所示。

![DALI-pipeline](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/_images/two_readers.svg)

## 4. 开发实战
本章节希望通过一个简单的例子和一个稍复杂的例子，介绍如何基于 DALI 提供的算子，在python侧进行二次开发，以满足实际需求。

### 4.1 RandomFlip
以 PaddleClas 已有的 [RandFlipImage](../../../../ppcls/data/preprocess/ops/operators.py#L499) 算子为例，我们希望在开启DALI训练时，将其转换为对应的 DALI 算子，且同样具备 **按指定的 `prob` 概率进行 指定的水平 or 垂直翻转**

#### 4.1.1 继承DALI已有类
DALI 已经提供了简单的翻转算子 [`nvidia.dali.ops.Flip`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#nvidia.dali.ops.Flip)，其通过 `horizontal` 与 `vertical` 参数来分别控制是否对图像进行水平、垂直翻转。但是其缺少随机性，无法直接按照一定概率进行翻转或不反转，因此我们需要继承这个翻转类，并重写其 `__init__` 方法和 `__call__` 方法。继承代码如下所示：

```python
import nvidia.dali.ops as ops

class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        ...

    def __call__(self, data, **kwargs):
        ...
```

#### 4.1.2 重写 \_\_init\_\_ 方法
我们需要在构造算子时加入随机参数来控制是否翻转，因此仿照普通 `RandFlipImage`算子的逻辑，在继承类的初始化方法中加入参数 `prob`，同理再加入 `flip_code` 用于控制水平、垂直翻转。

由于每一次执行我们都需要生成一个随机数（此处用0或1表示），代表是否在翻转轴上进行翻转，因此我们实例化一个 `ops.random.CoinFlip` 来作为随机数生成器（实例化对象为上述代码中的 `self.rng`），同理我们也需要记录翻转轴参数 `flip_code`，以供在后续 `__call__` 方法中调用。

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

#### 4.1.3 重写 \_\_call\_\_ 方法
有了 `self.rng` 和 `self.flip_code`，我们就能在每次调用的 `__call__` 方法内部，加入随机性。首先调用 `self.rng()` 的 `__call__` 方法，生成一个0或1的随机整数，0代表不进行翻转，1代表进行翻转；然后根据 `self.flip_code` ，将这个随机整数作为父类 `__call__` 方法的 `horizontal` 或 `vertical` 参数，调用父类的 `__call__` 方法完成翻转。这样就完成了一个简单的自定义DALI RandomFlip 算子的编写。完整代码如下所示：
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

### 4.2 RandomErasing
以 PaddleClas 已有的 [RandomErasing](../../../../ppcls/data/preprocess/ops/random_erasing.py#L52) 算子为例，我们希望在开启DALI训练时，将其转换为对应的 DALI 算子，且同样具备 **按指定的参数概率进行随机擦除**

#### 4.1.1 继承DALI已有类
DALI 已经提供了简单的翻转算子 [`nvidia.dali.ops.Erase`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops_legacy.html#nvidia.dali.ops.Flip)，其通过 `anchor`、`fill_value`、`shape`等参数控制擦除区域和擦除填充值。但是其缺少一定的随机性，无法直接按照一定分布选取擦除区域，且无法自定义填充值，因此我们需要继承这个擦除类，并重写其 `__init__` 方法和 `__call__` 方法。继承代码如下所示：

```python
import nvidia.dali.ops as ops

class RandomErasing(ops.Erase):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandomErasing, self).__init__(*kargs, device=device, **kwargs)
        ...

    def __call__(self, data, **kwargs):
        ...
```

#### 4.1.2 重写 \_\_init\_\_ 方法
我们需要在构造算子时加入随机参数来控制是否翻转，因此仿照普通 `RandomErasing`算子的逻辑，在继承类的初始化方法中加入参数 `prob`，同理再加入 `flip_code` 用于控制水平、垂直翻转。

由于每一次执行我们都需要生成一个随机数（此处用0或1表示），代表是否进行随机擦除，因此我们实例化一个 `ops.random.CoinFlip` 来作为随机数生成器（实例化对象为上述代码中的 `self.rng`）。在此基础上我们需要根据 `operators.py` 中的代码，构建出如 `target_area`、`aspect_ratio`这样的一些随机变量来控制擦除时的区域。

同时随机擦除时填充擦除区域的值也有多种选择，因此我们还需要另一个填充值生成类，该类的写法与 `random_erasing.py` 中的写法类似。

修改后代码如下所示：
```python
class Pixels(ops.random.Normal):
    def __init__(self, *kargs, device="cpu", mode="const", mean=[0.0, 0.0, 0.0], channel_first=False, h=224, w=224, c=3, **kwargs):
        super(Pixels, self).__init__(*kargs, device=device, **kwargs)
        self._mode = mode
        self._mean = mean
        self.channel_first = channel_first
        self.h = h
        self.w = w
        self.c = c

    def __call__(self, **kwargs):
        if self._mode == "rand":
            return super(Pixels, self).__call__(shape=(3)) if not self.channel_first else super(Pixels, self).__call__(shape=(3))
        elif self._mode == "pixel":
            return super(Pixels, self).__call__(shape=(self.h, self.w, self.c)) if not self.channel_first else super(Pixels, self).__call__(shape=(self.c, self.h, self.w))
        elif self._mode == "const":
            return fn.constant(fdata=self._mean, shape=(self.c)) if not self.channel_first else fn.constant(fdata=self._mean, shape=(self.c))
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(ops.Erase):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0.0, 0.0, 0.0],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const',
                 channel_first=False,
                 img_h=224,
                 img_w=224,
                 **kwargs):
        super(RandomErasing, self).__init__(*kargs, device=device, **kwargs)
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.mean = mean
        self.get_pixels = Pixels(device=device, mode=mode, mean=mean, channel_first=False, h=224, w=224, c=3)
        self.channel_first = channel_first
        self.img_h = img_h
        self.img_w = img_w
        self.area = img_h * img_w

    def __call__(self, data, **kwargs):
        ...
```

#### 4.1.3 重写 \_\_call\_\_ 方法
有了以上的一些变量，按照`operators.py`的逻辑进行代码编写，就能得到完整代码如下所示：
```python
class Pixels(ops.random.Normal):
    def __init__(self, *kargs, device="cpu", mode="const", mean=[0.0, 0.0, 0.0], channel_first=False, h=224, w=224, c=3, **kwargs):
        super(Pixels, self).__init__(*kargs, device=device, **kwargs)
        self._mode = mode
        self._mean = mean
        self.channel_first = channel_first
        self.h = h
        self.w = w
        self.c = c

    def __call__(self, **kwargs):
        if self._mode == "rand":
            return super(Pixels, self).__call__(shape=(3)) if not self.channel_first else super(Pixels, self).__call__(shape=(3))
        elif self._mode == "pixel":
            return super(Pixels, self).__call__(shape=(self.h, self.w, self.c)) if not self.channel_first else super(Pixels, self).__call__(shape=(self.c, self.h, self.w))
        elif self._mode == "const":
            return fn.constant(fdata=self._mean, shape=(self.c)) if not self.channel_first else fn.constant(fdata=self._mean, shape=(self.c))
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(ops.Erase):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0.0, 0.0, 0.0],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const',
                 channel_first=False,
                 img_h=224,
                 img_w=224,
                 **kwargs):
        super(RandomErasing, self).__init__(*kargs, device=device, **kwargs)
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.mean = mean
        self.get_pixels = Pixels(device=device, mode=mode, mean=mean, channel_first=False, h=224, w=224, c=3)
        self.channel_first = channel_first
        self.img_h = img_h
        self.img_w = img_w
        self.area = img_h * img_w

    def __call__(self, data, **kwargs):
        do_aug = fn.random.coin_flip(probability=self.EPSILON)
        keep = do_aug ^ True
        target_area = fn.random.uniform(range=(self.sl, self.sh)) * self.area
        aspect_ratio = fn.random.uniform(range=(self.r1[0], self.r1[1]))
        if self.use_log_aspect:
            aspect_ratio = nvmath.exp(aspect_ratio)
        h = nvmath.floor(nvmath.sqrt(target_area * aspect_ratio))
        w = nvmath.floor(nvmath.sqrt(target_area / aspect_ratio))
        pixels = self.get_pixels()
        range1 = fn.stack((self.img_h-h)/self.img_h-(self.img_h-h)/self.img_h, (self.img_h-h)/self.img_h)
        range2 = fn.stack((self.img_w-w)/self.img_w-(self.img_w-w)/self.img_w, (self.img_w-w)/self.img_w)
        # shapes
        x1 = fn.random.uniform(range=range1)
        y1 = fn.random.uniform(range=range2)
        anchor = fn.stack(x1, y1)
        shape = fn.stack(h, w)
        aug_data = super(RandomErasing, self).__call__(
            data,
            anchor=anchor,
            normalized_anchor=True,
            shape=shape,
            fill_value=pixels)
        return aug_data * do_aug + data * keep
```



