# TheseusLayer 使用说明

基于 TheseusLayer 构建的网络模型，支持网络截断、返回网络中间层输出和修改网络中间层的功能。

---

## 目录

- [1. 前言](#1)
- [2. 网络层描述符说明](#2)
- [3. 功能介绍](#3)
    - [3.1 网络截断（stop_after）](#3.1)
    - [3.2 返回网络中间层输出（update_res）](#3.2)
    - [3.3 修改网络中间层（upgrade_sublayer）](#3.3)

<a name="1"></a>

## 1. 前言

`TheseusLayer` 是继承了 `nn.Layer` 的子类，使用 `TheseusLayer` 作为父类构建的网络模型，可以通过 `TheseusLayer` 的 `stop_after()`、`update_res()` 和 `upgrade_sublayer()` 实现网络截断、返回中间层输出以及修改网络中间层的功能。目前 PaddleClas 中 `ppcls.arch.backbone.legendary_models` 下的所有模型均支持上述操作。

如需基于 `TheseusLayer` 构建新的网络结构，只需继承 `TheseusLayer` 即可：

```python
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer

class net(TheseusLayer):
    def __init__():
        super().__init__()

    def forward(x):
        pass
```

<a name="2"></a>

## 2. 网络层描述符说明

使用 `TheseusLayer` 提供的方法对模型进行操作/修改时，需要通过参数指定网络中间层，因此 `TheseusLayer` 规定了用于描述网络中间层的网络层描述符。

网络层描述符的使用需要符合以下规则：
* 为 Python 字符串（str）类型；
* 使用网络层对象的变量名指定该网络层；
* 以 `.` 作为网络层级的分隔符；
* 对于 `nn.Sequential` 类型或是 `nn.LayerList` 类型的层，使用 `["index"]` 指定其子层。

以 `MobileNetV1` 网络为例，其模型结构定义在 [MobileNetV1](../../../ppcls/arch/backbone/legendary_models/mobilenet_v1.py)，为方便说明，可参考下方网络结构及不同网络层所对应的网络层描述符。可以清晰看出，对于 `MobileNetV1` 网络的任一子层，均可按层级结构逐层指定，不同层级结构间使用 `.` 进行分隔即可。

```shell
# 网络层对象的变量名（该对象所属类）....................(该网络层对应的网络层描述符)

MobileNetV1
├── conv (ConvBNLayer)............................("conv")
│   ├── conv (nn.Conv2D)..........................("conv.conv")
│   ├── bn (nn.BatchNorm).........................("conv.bn")
│   └── relu (nn.ReLU)............................("conv.relu")
│
├── blocks (nn.Sequential)........................("blocks")
│   ├── blocks0 (DepthwiseSeparable)..............("blocks[0]")
│   │   ├── depthwise_conv (ConvBNLayer)..........("blocks[0].depthwise_conv")
│   │   │   ├── conv (nn.Conv2D)..................("blocks[0].depthwise_conv.conv")
│   │   │   ├── bn (nn.BatchNorm).................("blocks[0].depthwise_conv.bn")
│   │   │   └── relu (nn.ReLU)....................("blocks[0].depthwise_conv.relu")
│   │   └── pointwise_conv (ConvBNLayer)..........("blocks[0].pointwise_conv")
│   │       ├── conv (nn.Conv2D)..................("blocks[0].pointwise_conv.conv")
│   │       ├── bn (nn.BatchNorm).................("blocks[0].pointwise_conv.bn")
│   │       └── relu (nn.ReLU)....................("blocks[0].pointwise_conv.relu")
│   .
│   .
│   .
│   └── blocks12 (DepthwiseSeparable).............("blocks[12]")
│       ├── depthwise_conv (ConvBNLayer)..........("blocks[0].depthwise_conv")
│       │   ├── conv (nn.Conv2D)..................("blocks[0].depthwise_conv.conv")
│       │   ├── bn (nn.BatchNorm).................("blocks[0].depthwise_conv.bn")
│       │   └── relu (nn.ReLU)....................("blocks[0].depthwise_conv.relu")
│       └── pointwise_conv (ConvBNLayer)..........("blocks[0].pointwise_conv")
│           ├── conv (nn.Conv2D)..................("blocks[0].pointwise_conv.conv")
│           ├── bn (nn.BatchNorm).................("blocks[0].pointwise_conv.bn")
│           └── relu (nn.ReLU)....................("blocks[0].pointwise_conv.relu")
│
├── avg_pool (nn.AdaptiveAvgPool2D)...............("avg_pool")
│
├── flatten (nn.Flatten)..........................("flatten")
│
└── fc (nn.Linear)................................("fc")
```

因此，对于 `MobileNetV1` 网络：
* 网络层描述符 `flatten`，其指定了网络 `MobileNetV1` 的 `flatten` 这一层。
* 网络层描述符 `blocks[5]`，其指定了网络 `MobileNetV1` 的 `blocks` 层中的第 `6` 个 `DepthwiseSeparable` 对象这一层；
* 网络层描述符 `blocks[0].depthwise_conv.conv`，其指定了网络 `MobileNetV1` 的 `blocks` 层中的第 `1` 个 `DepthwiseSeparable` 对象中的 `depthwise_conv` 中的 `conv` 这一层；

<a name="3"></a>

## 3. 方法说明

PaddleClas 提供的 backbone 网络均基于图像分类数据集训练得到，因此网络的尾部带有用于分类的全连接层，而在特定任务场景下，需要去掉分类的全连接层。在部分下游任务中，例如目标检测场景，需要获取到网络中间层的输出结果，也可能需要对网络的中间层进行修改，因此 `TheseusLayer` 提供了 3 个接口函数用于实现不同的修改功能。

<a name="3.1"></a>

### 3.1 网络截断（stop_after）

```python
def stop_after(self, stop_layer_name: str) -> bool:
    """stop forward and backward after 'stop_layer_name'.

    Args:
        stop_layer_name (str): The name of layer that stop forward and backward after this layer.

    Returns:
        bool: 'True' if successful, 'False' otherwise.
    """
```

该方法可通过参数 `stop_layer_name` 指定网络中的特定子层，并停止该层之后的所有层的前后向传输，在逻辑上，该层之后不再有其他网络层。

* 参数：
    * `stop_layer_name`： `str` 类型的对象，用于指定网络子层的网络层描述符。关于网络层描述符的具体规则，请查看[网络层描述符说明](#2)。
* 返回值：
    * 当该方法成功执行时，其返回值为 `True`，否则为 `False`。

以 `MobileNetV1` 网络为例，参数 `stop_layer_name` 为 `"blocks[0].depthwise_conv.conv"`，具体效果可以参考下方代码案例进行尝试。

```python
# cd <root-path-to-PaddleClas> or pip install paddleclas to import paddleclas
import paddleclas

net = paddleclas.MobileNetV1()
print("========== the origin mobilenetv1 net arch ==========")
print(net)

res = net.stop_after(stop_layer_name="blocks[0].depthwise_conv.conv")
print("The result returned by stop_after(): ", res)
# The result returned by stop_after(): True

print("\n\n========== the truncated mobilenetv1 net arch ==========")
print(net)
```

<a name="3.2"></a>

### 3.2 返回网络中间层输出（update_res）

```python
def update_res(
        self,
        return_patterns: Union[str, List[str]]) -> Dict[str, nn.Layer]:
    """update the result(s) to be returned.

    Args:
        return_patterns (Union[str, List[str]]): The name of layer to return output.

    Returns:
        Dict[str, nn.Layer]: The pattern(str) and corresponding layer(nn.Layer) that have been set successfully.
    """
```

该方法可通过参数 `return_patterns` 指定一层（str 对象）或多层（list 对象）网络的中间子层，并在网络前向时，将指定层的输出结果与网络的最终结果一同返回。

* 参数：
    * `return_patterns`：作为网络层描述符的 `str` 对象，或是 `str` 对象所组成的 `list` 对象，其元素为用于指定网络子层的网络层描述符。关于网络层描述符的具体规则，请查看[网络层描述符说明](#2)。
* 返回值：
    * 该方法的返回值为 `list` 对象，元素为设置成功的子层的网络层描述符。

以 `MobileNetV1` 网络为例，当 `return_patterns` 为 `["blocks[0]", "blocks[2]", "blocks[4]", "blocks[10]"]`，在网络前向推理时，网络的输出结果将包含以上 4 层的输出和网络最终的输出，具体效果可以参考下方代码案例进行尝试。

```python
import numpy as np
import paddle

# cd <root-path-to-PaddleClas> or pip install paddleclas to import paddleclas
import paddleclas

np_input = np.zeros((1, 3, 224, 224))
pd_input  = paddle.to_tensor(np_input, dtype="float32")

net = paddleclas.MobileNetV1(pretrained=True)

output = net(pd_input)
print("The output's type of origin net: ", type(output))
# The output's type of origin net: <class 'paddle.Tensor'>

res = net.update_res(return_patterns=["blocks[0]", "blocks[2]", "blocks[4]", "blocks[10]"])
print("The result returned by update_res(): ", res)
# The result returned by update_res():  ['blocks[0]', 'blocks[2]', 'blocks[4]', 'blocks[10]']

output = net(pd_input)
print("The output's keys of processed net: ", output.keys())
# The output's keys of net:  dict_keys(['output', 'blocks[0]', 'blocks[2]', 'blocks[4]', 'blocks[10]'])
# 网络前向输出 output 为 dict 类型对象，其中，output["output"] 为网络最终输出，output["blocks[0]"] 等为网络中间层输出结果
```

除了通过调用方法 `update_res()` 的方式之外，也同样可以在实例化网络对象时，通过指定参数 `return_patterns` 实现相同效果：

```python
net = paddleclas.MobileNetV1(pretrained=True, return_patterns=["blocks[0]", "blocks[2]", "blocks[4]", "blocks[10]"])
```

并且在实例化网络对象时，还可以通过参数 `return_stages` 指定网络不同 `stage` 的输出，如下方代码所示：

```python
# 当 `return_stages` 为 `True`，会将网络所有 stage 的前向输出一并返回，如下所示：
net = paddleclas.MobileNetV1(pretrained=True, return_stages=True)

# 当 `return_stages` 为 list 对象，可以指定需要返回输出结果的 stage 的序号，如下所示：
net = paddleclas.MobileNetV1(pretrained=True, return_stages=[0, 1, 2, 3])
```

<a name="3.3"></a>

### 3.3 修改网络中间层（upgrade_sublayer）

```python
def upgrade_sublayer(self,
                        layer_name_pattern: Union[str, List[str]],
                        handle_func: Callable[[nn.Layer, str], nn.Layer]
                        ) -> Dict[str, nn.Layer]:
    """use 'handle_func' to modify the sub-layer(s) specified by 'layer_name_pattern'.

    Args:
        layer_name_pattern (Union[str, List[str]]): The name of layer to be modified by 'handle_func'.
        handle_func (Callable[[nn.Layer, str], nn.Layer]): The function to modify target layer specified by 'layer_name_pattern'. The formal params are the layer(nn.Layer) and pattern(str) that is (a member of) layer_name_pattern (when layer_name_pattern is List type). And the return is the layer processed.

    Returns:
        Dict[str, nn.Layer]: The key is the pattern and corresponding value is the result returned by 'handle_func()'.
    """
```

该方法可通过参数 `layer_name_pattern` 指定一层（str 对象）或多层（list 对象）网络中间子层，并使用参数 `handle_func` 所指定的函数对指定的子层进行修改。

* 参数：
    * `layer_name_pattern`：作为网络层描述符的 `str` 对象，或是 `str` 对象所组成的 `list` 对象，其元素为用于指定网络子层的网络层描述符。关于网络层描述符的具体规则，请查看[网络层描述符说明](#2)。
    * `handle_func`：有 2 个形参的可调用对象，第 1 个形参为 `nn.Layer` 类型，第 2 个形参为 `str` 类型，该可调用对象返回值必须为 `nn.Layer` 类型对象或是有 `forward` 方法的对象。
* 返回值：
    * 该方法的返回值为 `list` 对象，元素为修改成功的网络子层的网络层描述符。

`upgrade_sublayer` 方法会根据 `layer_name_pattern` 查找对应的网络子层，并将查找到的子层和其 `pattern` 传入可调用对象 `handle_func`，并使用 `handle_func` 的返回值替换该层。

以 `MobileNetV1` 网络为例，将网络最后的 2 个 block 中的深度可分离卷积（depthwise_conv）改为 `5*5` 大小的卷积核，同时将 padding 改为 `2`，如下方代码所示：

```python
from paddle import nn

# cd <root-path-to-PaddleClas> or pip install paddleclas to import paddleclas
import paddleclas

# 该函数必须有两个形参
# 第一个形参用于接受指定的网络中间子层
# 第二个形参用于接受指定网络子层的网络层描述符
def rep_func(layer: nn.Layer, pattern: str):
    new_layer = nn.Conv2D(
        # layer 为 blocks[11].depthwise_conv.conv 或
        # blocks[12].depthwise_conv.conv 所对应的网络中间子层
        # 因此，新的网络层（new_layer）与被替换掉的网络层具有相同的
        # in_channels 属性和 out_channels 属性
        in_channels=layer._in_channels,
        out_channels=layer._out_channels,
        kernel_size=5,
        padding=2
    )
    # 该函数返回值为新的网络层
    # upgrade_sublayer() 方法将使用该返回值替换对应的网络中间子层
    return new_layer

net = paddleclas.MobileNetV1(pretrained=True)
print("========== the origin mobilenetv1 net arch ==========")
print(net)

res = net.upgrade_sublayer(layer_name_pattern=["blocks[11].depthwise_conv.conv", "blocks[12].depthwise_conv.conv"], handle_func=rep_func)
print("The result returned by upgrade_sublayer() is", res)
# The result returned by upgrade_sublayer() is ['blocks[11].depthwise_conv.conv', 'blocks[12].depthwise_conv.conv']

print("\n\n========== the upgraded mobilenetv1 net arch ==========")
print(net)
```
