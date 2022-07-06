# PaddleClas FAQ Summary - 2022 Season 1

## Before You Read

- We collect some frequently asked questions in issues and user groups since PaddleClas is open-sourced and provide brief answers, aiming to give some reference for the majority to save you from twists and turns.
- There are many talents in the field of image classification, recognition and retrieval with quickly updated models and papers, and the answers here mainly rely on our limited project practice, so it is not possible to cover all facets. We sincerely hope that the man of insight will help to supplement and correct the content, thanks a lot.

## Catalogue

- [1. Theory](#1-theory)
- [2. Practice](#2-actual-combat)
  - [2.1 Common problems of training and evaluation](#21-common-problems-of-training-and-evaluation)
    - [Q2.1.1 How to freeze the parameters of some layers during training?](#q211-how-to-freeze-the-parameters-of-some-layers-during-training)

<a name="1"></a>
## 1. Theory

<a name="2"></a>
## 2. Practice

<a name="2.1"></a>
### 2.1 Common problems of training and evaluation

#### Q2.1.1 How to freeze the parameters of some layers during training?
**A**: There are currently three methods available
1. Manually modify the model code, use `paddle.ParamAttr(learning_rate=0.0)`, and set the learning rate of the frozen layer to 0.0. For details, see [paddle.ParamAttr documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/ParamAttr_en.html#paramattr). The following code can set the learning rate of the weight parameter of the self.conv layer to 0.0.
   ```python
   self.conv = Conv2D(
        in_channels=num_channels,
        out_channels=num_filters,
        kernel_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        weight_attr=ParamAttr(learning_rate=0.0), # <-- set here
        bias_attr=False,
        data_format=data_format)
   ```

2. Manually set stop_gradient=True for the frozen layer, please refer to [this link](https://github.com/RainFrost1/PaddleClas/blob/24e968b8d9f7d9e2309e713cbf2afe8fda9deacd/ppcls/engine/train/train_idml.py#L40-L66). When using this method, after the gradient is returned to the layer which set strop_gradient=True, the gradient backward is stopped, that is, the weight of the previous layer will be fixed.

3. After loss.backward() and before optimizer.step(), use the clear_gradients() method in nn.Layer. For the layer to be fixed, call this method without affecting the loss return. The following code can clear the gradient of a layer or the gradient of a parameter of a layer
    ```python
    import paddle
    linear = paddle.nn.Linear(3, 4)
    x = paddle.randn([4, 3])
    y = linear(x)
    loss = y.sum().backward()

    print(linear.weight.grad)
    print(linear.bias.grad)
    linear.clear_gradients() # clear the gradients of the entire layer
    # linear.weight.clear_grad() # Only clear the gradient of the weight parameter of the Linear layer
    print(linear.weight.grad)
    print(linear.bias.grad)
    ```
