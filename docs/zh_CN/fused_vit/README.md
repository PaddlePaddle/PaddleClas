# Fused Vision Transformer 高性能推理使用

PaddleClas 中已经添加高性能推理模型相关实现，支持：

| Model                                                                                           | FP16 | Wint8 | Wint4 | PTQ |
|-------------------------------------------------------------------------------------------------|------|-------|-------|-----|
| [Fused Vision Transformer](../../../ppcls/arch/backbone/model_zoo/fused_vision_transformer.py)  | ✅    | ✅    | ✅    | ❌  |

* 支持以下`fused_vit`类型
  * `Fused_ViT_small_patch16_224`
  * `Fused_ViT_base_patch16_224`
  * `Fused_ViT_base_patch16_384`
  * `Fused_ViT_base_patch32_384`
  * `Fused_ViT_large_patch16_224`
  * `Fused_ViT_large_patch16_384`
  * `Fused_ViT_large_patch32_384`
* 预训练权重来自Vision Transformer对应权重

## 安装自定义算子库

PaddleClas 针对于 Fused Vision Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能。

```shell
cd ./PaddleClas/csrc
pip install -r requirements.txt
python setup_cuda.py install
```

## 静态图推理

* 模型导出

```python
from paddleclas import (
    Fused_ViT_large_patch16_224,
    Fused_ViT_large_patch32_384
)
import paddle

if __name__ == "__main__":
    dtype = "float16"
    paddle.set_default_dtype(dtype)
    path = "/your/path/fused_384_fp16/static_model"
    model = Fused_ViT_large_patch32_384(pretrained=True, class_num=1000)
    model.eval()
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + [3, 384, 384],
                dtype=dtype
            )
        ]
    )
    paddle.jit.save(model, path)
```

* 模型推理

```python
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from paddle.inference import Config
from paddleclas_ops import (
    qkv_transpose_split,
    transpose_remove_padding
)
import paddle
import numpy as np

from paddleclas import (
    Fused_ViT_large_patch32_384,
)

def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i])
    
    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

def static_infer(model_file, params_file, images):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)
    
    predictor = create_predictor(config)

    output = run(predictor, [images])
    
    return output

def main_fp16():
    dtype = "float16"
    N, C, H, W = (1, 3, 384, 384)
    images = np.random.rand(N, C, H, W).astype(dtype)

    # fp32 static infer
    model_file = "/your/path/fused_384_fp16/static_model.pdmodel"
    params_file = "/your/path/fused_384_fp16/static_model.pdiparams"
    static_fp16_output = static_infer(model_file, params_file, images)

if __name__ == "__main__":
    main_fp16()
```

## 动态图推理

### FP16

* `fused_vit`通过`paddle.set_default_dtype`来设置`weight`的数据类型

```python
import paddle
 
from paddleclas import (
    Fused_ViT_large_patch32_384,
)

if __name__ == '__main__':
    dtype = "float16"
    N, C, H, W = (1, 3, 384, 384)
    images = paddle.randn([N, C, H, W]).cast(dtype)
    paddle.set_default_dtype(dtype)

    # ----- Fused Model -----
    fused_model = Fused_ViT_large_patch32_384(pretrained=True, class_num=1000)
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
    Fused_ViT_large_patch32_384,
)

if __name__ == '__main__':
    dtype = "float16"
    N, C, H, W = (1, 3, 384, 384)
    images = paddle.randn([N, C, H, W]).cast(dtype)
    paddle.set_default_dtype(dtype)

    # ----- 8 bits Quanted Model -----
    quanted_model_8 = Fused_ViT_large_patch32_384(pretrained=True, class_num=1000, use_weight_only=True)
    quanted_output_8 = quanted_model_8(images)
    print(quanted_output_8)

    # ----- 4 bits Quanted Model -----
    quanted_model_4 = Fused_ViT_large_patch32_384(pretrained=True, class_num=1000, use_weight_only=True, quant_type="weight_only_int4")
    quanted_output_4 = quanted_model_4(images)
    print(quanted_output_4)
```

## 性能数据
* 环境
  * GPU: V100(16GB)
  * paddle cuda: 11.8
  * warmup_time=10, test_time=100
  
### ViT_large_patch16_224 (N, C, H, W) = (1, 3, 224, 224)
* 相较于origin vit fp32
  * fused vit fp32提升136.86%
  * fused vit static fp32提升132.88%
  * fused vit fp16提升175.90%
  * fused vit static fp16提升209.76%
  * fused vit wint8提升220.39%

| 次数 | origin vit fp32 | fused vit fp32 | fused vit  static fp32 | fused vit fp16 | fused vit  static fp16 | fused vit wint8 |
| ---- | --------------- | -------------- | ---------------------- | -------------- | ---------------------- | --------------- |
| 1    | 21.96368694     | 15.52280903    | 16.04069233            | 12.71003485    | 10.15134335            | 9.884309769     |
| 2    | 21.41675949     | 15.57497263    | 15.88841915            | 12.28075504    | 10.69375753            | 9.766004086     |
| 3    | 21.50935411     | 15.41593552    | 15.86268902            | 12.11336613    | 10.0124383             | 9.629456997     |
| 4    | 21.6180253      | 16.7787528     | 16.74314499            | 12.00630426    | 10.32423496            | 9.885079861     |
| 5    | 21.67198658     | 15.75221062    | 16.87407255            | 12.39049435    | 10.3924942             | 9.920876026     |
| 平均 | 21.63596249     | 15.80893612    | 16.28180361            | 12.30019093    | 10.31485367            | 9.817145348     |


### ViT_large_patch32_384 (N, C, H, W) = (1, 3, 384, 384)
* 相较于origin vit fp32
  * fused vit fp32提升163.87%
  * fused vit static fp32提升161.75%
  * fused vit fp16提升177.96%
  * fused vit static fp16提升216.21%
  * fused vit wint8提升244.83%

| 次数 | origin vit fp32 | fused vit fp32 | fused vit  static fp32 | fused vit fp16 | fused vit  static fp16 | fused vit wint8 |
| ---- | --------------- | -------------- | ---------------------- | -------------- | ---------------------- | --------------- |
| 1    | 21.51850939     | 13.04458618    | 12.54903555            | 12.19017506    | 9.848031998            | 8.791148663     |
| 2    | 20.26674986     | 13.22592258    | 13.61229658            | 12.60421276    | 10.83174944            | 8.864238262     |
| 3    | 21.70393944     | 13.23971033    | 13.60147953            | 11.62441969    | 8.626818657            | 8.980474472     |
| 4    | 24.82402325     | 13.95256519    | 14.1339612             | 12.76894808    | 10.11808157            | 9.212892056     |
| 5    | 20.70281267     | 13.06502581    | 13.49942923            | 12.07243919    | 10.99670172            | 8.678123951     |
| 平均 | 21.80320692     | 13.30556202    | 13.47924042            | 12.25203896    | 10.08427668            | 8.905375481     |