# Fused Vision Transformer 高性能推理使用

PaddleClas 中已经添加高性能推理模型相关实现，支持：

| Model                                                                                           | FP16 | Wint8 | Wint4 | PTQ |
|-------------------------------------------------------------------------------------------------|------|-------|-------|-----|
| [Fused Vision Transformer](../../../ppcls/arch/backbone/model_zoo/fused_vision_transformer.py)  | ✅    | ✅    | ✅    | ❌  |

[TOC]

## 安装自定义算子库

PaddleClas 针对于 Fused Vision Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能。

```shell
cd ./PaddleClas/csrc
pip install -r requirements.txt
python setup_cuda.py install
```

## Demo

* 支持以下`fused_vit`类型
  * `Fused_ViT_small_patch16_224`
  * `Fused_ViT_base_patch16_224`
  * `Fused_ViT_base_patch16_384`
  * `Fused_ViT_base_patch32_384`
  * `Fused_ViT_large_patch16_224`
  * `Fused_ViT_large_patch16_384`
  * `Fused_ViT_large_patch32_384`
* 预训练权重来自Vision Transformer对应权重

### FP16

* `fused_vit`通过`paddle.set_default_dtype`来设置`weight`的数据类型

```python
import paddle
 
from paddleclas import (
    Fused_ViT_large_patch16_224,
)

if __name__ == '__main__':
    dtype = "float16"
    N, C, H, W = (1, 3, 224, 224)
    images = paddle.randn([N, C, H, W]).cast(dtype)
    paddle.set_default_dtype(dtype)

    # ----- Fused Model -----
    fused_model = Fused_ViT_large_patch16_224(pretrained=True, class_num=1000)
    fused_output = fused_model(images)
    print(fused_output)
```

### Weight Only Int8/Int4 推理

> 当前 weight_only_int8/4 仅支持A100，V100 上的 weight only int8/4 存在精度问题。

* 参数介绍：
  * `use_weight_only`：使用 weight only 推理，默认为 False
  * `quant_type`：weight only 类型，默认为`weight_only_int8`，可选`weight_only_int4`

```python
import paddle
 
from paddleclas import (
    Fused_ViT_large_patch16_224,
)

if __name__ == '__main__':
    dtype = "float16"
    N, C, H, W = (1, 3, 224, 224)
    images = paddle.randn([N, C, H, W]).cast(dtype)
    paddle.set_default_dtype(dtype)

    # ----- 8 bits Quanted Model -----
    quanted_model_8 = Fused_ViT_large_patch16_224(pretrained=True, class_num=1000, use_weight_only=True)
    quanted_output_8 = quanted_model_8(images)
    print(quanted_output_8)

    # ----- 4 bits Quanted Model -----
    quanted_model_4 = Fused_ViT_large_patch16_224(pretrained=True, class_num=1000, use_weight_only=True, quant_type="weight_only_int4")
    quanted_output_4 = quanted_model_4(images)
    print(quanted_output_4)
```
