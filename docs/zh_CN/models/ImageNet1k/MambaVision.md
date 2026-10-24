# MambaVision

## 1. 模型介绍

MambaVision 是 NVIDIA 提出的分层视觉骨干网络，在统一架构中结合卷积、自注意力与 Mamba 状态空间混合器。浅层卷积用于提取局部视觉特征，深层 Mamba 和 Attention Block 用于建模长程依赖。

- 论文：[MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)
- 官方实现：[NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)
- PaddlePaddle 权重：[AI Studio MambaVision 模型空间](https://aistudio.baidu.com/modelsdetail/51615)

本实现提供 11 个 MambaVision 分类模型入口及对应的 ImageNet-1K 训练、评估和推理配置。模型名称中的 `21K` 表示上游预训练数据来源，当前发布的权重均使用 1,000 类分类头。

## 2. 模型与权重

| PaddleClas 名称 | 配置 | 输入尺寸 | 参数量(M) | Paddle Top-1 / Top-5 | 权重文件 |
| --- | --- | ---: | ---: | ---: | --- |
| `MambaVision_T` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_T.yaml) | 224 | 31.8 | 82.176% / 96.172% | `mambavision_tiny_1k.pdparams` |
| `MambaVision_T2` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_T2.yaml) | 224 | 35.1 | 82.636% / 96.272% | `mambavision_tiny2_1k.pdparams` |
| `MambaVision_S` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_S.yaml) | 224 | 50.1 | 83.232% / 96.502% | `mambavision_small_1k.pdparams` |
| `MambaVision_B` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_B.yaml) | 224 | 97.7 | 84.204% / 96.848% | `mambavision_base_1k.pdparams` |
| `MambaVision_B_21K` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_B_21K.yaml) | 224 | 97.7 | 84.876% / 97.478% | `mambavision_base_21k.pdparams` |
| `MambaVision_L` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L.yaml) | 224 | 227.9 | 84.954% / 97.078% | `mambavision_large_1k.pdparams` |
| `MambaVision_L_21K` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L_21K.yaml) | 224 | 227.9 | 86.140% / 97.968% | `mambavision_large_21k.pdparams` |
| `MambaVision_L2` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L2.yaml) | 224 | 241.5 | 85.282% / 97.160% | `mambavision_large2_1k.pdparams` |
| `MambaVision_L2_512_21K` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L2_512_21K.yaml) | 512 | 241.5 | 87.114% / 98.256% | `mambavision_L2_21k_240m_512.pdparams` |
| `MambaVision_L3_256_21K` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L3_256_21K.yaml) | 256 | 739.6 | 87.294% / 98.318% | `mambavision_L3_21k_740m_256.pdparams` |
| `MambaVision_L3_512_21K` | [配置](../../../../ppcls/configs/ImageNet/MambaVision/MambaVision_L3_512_21K.yaml) | 512 | 739.6 | 87.822% / 98.452% | `mambavision_L3_21k_740m_512.pdparams` |

以上 Paddle 指标使用转换后的 PaddlePaddle 权重在 ImageNet-1K 50,000 张验证图像上评估。转换前后的逐图 Top-1 预测一致率均为 100%，logits 最大绝对差均小于 `1e-4`。这些结果用于验证权重转换和前向计算的一致性，不代表使用 PaddlePaddle 从头训练复现论文精度。

全部权重可从 [AI Studio MambaVision 模型空间](https://aistudio.baidu.com/modelsdetail/51615) 下载。请根据表中的 PaddleClas 名称选择对应的权重文件。

当前不提供 `pretrained=True` 自动下载。下载权重后，请将本地 `.pdparams` 文件路径传给模型工厂的 `pretrained` 参数，或通过配置项 `Global.pretrained_model` 加载。

## 3. 使用方法

以下命令以 `MambaVision_T` 为例。使用其他规格时，需要同时替换配置文件和权重文件。

### 3.1 训练

```bash
python tools/train.py \
  -c ppcls/configs/ImageNet/MambaVision/MambaVision_T.yaml \
  -o Global.device=gpu
```

### 3.2 评估

```bash
python tools/eval.py \
  -c ppcls/configs/ImageNet/MambaVision/MambaVision_T.yaml \
  -o Global.pretrained_model=/path/to/mambavision_tiny_1k.pdparams \
  -o Global.device=gpu
```

### 3.3 推理

```bash
python tools/infer.py \
  -c ppcls/configs/ImageNet/MambaVision/MambaVision_T.yaml \
  -o Global.pretrained_model=/path/to/mambavision_tiny_1k.pdparams \
  -o Infer.infer_imgs=/path/to/image.jpg \
  -o Global.device=gpu
```

### 3.4 Python API

```python
import paddle

from ppcls.arch.backbone import MambaVision_T

model = MambaVision_T(pretrained="/path/to/mambavision_tiny_1k.pdparams")
model.eval()

x = paddle.randn([1, 3, 224, 224])
with paddle.no_grad():
    logits = model(x)
print(logits.shape)  # [1, 1000]
```

## 4. 许可证

MambaVision 源代码遵循 [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/main/LICENSE)，预训练权重遵循 [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。使用代码和权重前请确认符合相应许可证条款。
