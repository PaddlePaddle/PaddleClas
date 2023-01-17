# 使用静态图模式训练

飞桨框架支持动态图模式和静态图模式，通常情况下使用动态图训练，即可满足大部分场景需求。飞桨经过多个版本的持续优化，动态图模型训练的性能已经可以和静态图媲美。如果在某些场景下确实需要使用静态图模式训练，我们推荐优先使用动转静训练功能，即仍然采用更易用的动态图方式构建模型，再转为静态图模式进行训练。同时，考虑到对静态图训练的兼容，PaddleClas 同样提供了静态图训练功能。

## 一、动转静训练

仅需在训练配置文件中设置参数 `Global.to_static` 为 `True`，同时通过参数 `Global.image_shape` 输入数据 shape 即可，如 `[3, 224, 224]`。

## 二、静态图训练

在 PaddleClas 中，静态图训练与动态图类似，同样使用配置文件的方式指定训练参数，训练入口脚本为 `ppcls/static/train.py`，以 ResNet50 模型为例，训练启动命令如下：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    ppcls/static/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50_amp_O1.yaml
```
