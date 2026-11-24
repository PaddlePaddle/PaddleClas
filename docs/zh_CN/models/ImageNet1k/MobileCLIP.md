# MobileCLIP

[MobileCLIP](https://arxiv.org/abs/2311.17049) 是 Apple 提出的轻量级图文模型。本实现支持第一代 S0、S1、S2、B 和 B-LT，基于[官方代码](https://github.com/apple-aiml-research/ml-mobileclip)的重参数化结构。

| 型号 | PaddleClas 模型名 | 图像分辨率 | 特征维度 |
| --- | --- | --- | --- |
| S0 | MobileCLIP_S0 | 256 | 512 |
| S1 | MobileCLIP_S1 | 256 | 512 |
| S2 | MobileCLIP_S2 | 256 | 512 |
| B | MobileCLIP_B | 224 | 512 |
| B-LT | MobileCLIP_B_LT | 224 | 512 |

分类模型使用 PaddleClas 的训练、评估和导出入口，`class_num=0` 返回图像特征，`class_num>0` 添加分类头。完整图文模型通过 `MobileCLIP` 类提供。

## 权重转换

下载[官方权重](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/get_pretrained_models.sh)，准备官方 PyTorch 环境和 PaddleClas 环境。转换分两步执行，可使用不同的 Python 环境。

在 PyTorch 环境导出融合权重和参考输出：

```bash
python tools/convert_mobileclip.py --stage export \
  --torch-repo ../ml-mobileclip \
  --checkpoint checkpoints/mobileclip_s0.pt \
  --model mobileclip_s0 --output-dir output/mobileclip
```

在 Paddle 环境转换并验证：

```bash
OMP_NUM_THREADS=4 python tools/convert_mobileclip.py --stage convert \
  --model mobileclip_s0 --output-dir output/mobileclip
```

其余型号替换为 `mobileclip_s1`、`mobileclip_s2`、`mobileclip_b` 或 `mobileclip_blt`。输出目录中包含完整图文权重 `<model>.pdparams`、用于分类骨干初始化的 `<model>_image.pdparams`，以及对齐结果 `alignment.json`。

脚本严格检查参数名称和形状，仅转置 Linear 权重；图文投影参数保留原布局。使用相同的 NumPy 随机输入，比较原始特征、归一化特征、温度参数和双向图文 Logits，最大绝对误差必须小于 `1e-4`。

## 训练与导出

按 PaddleClas 的 ImageNet 数据格式准备 `dataset/ILSVRC2012`，使用对应配置训练：

```bash
python tools/train.py -c ppcls/configs/MobileCLIP/MobileCLIP_S0.yaml \
  -o Arch.pretrained=output/mobileclip/mobileclip_s0/mobileclip_s0_image.pdparams
```

`Arch.pretrained` 通过公共权重加载工具初始化图像编码器，分类头从随机参数开始训练。恢复分类训练使用标准 `Global.checkpoints`。评估和导出使用训练得到的分类权重：

```bash
python tools/eval.py -c ppcls/configs/MobileCLIP/MobileCLIP_S0.yaml \
  -o Arch.pretrained=False -o Global.pretrained_model=output/latest

python tools/export_model.py -c ppcls/configs/MobileCLIP/MobileCLIP_S0.yaml \
  -o Arch.pretrained=False -o Global.pretrained_model=output/latest
```

TIPC 配置位于 `test_tipc/configs/MobileCLIP`，使用仓库现有 TIPC 流程执行。

## 图文推理

图像输入为 RGB、NCHW、float32，像素范围 `[0, 1]`。预处理采用双线性等比缩放和中心裁剪，不使用 CLIP 的 mean/std。文本输入为官方 OpenCLIP tokenizer 产生的 int64 token IDs，默认长度为 77。

```python
import paddle
from ppcls.arch.backbone.model_zoo.mobileclip import MobileCLIP

model = MobileCLIP("mobileclip_s0")
model.set_state_dict(paddle.load(
    "output/mobileclip/mobileclip_s0/mobileclip_s0.pdparams"))
model.eval()
with paddle.no_grad():
    image_features, text_features, scale = model(images, tokens)
    logits = scale * image_features @ text_features.transpose([1, 0])
```

## 验证结果

CPU float32 环境：Paddle `3.4.0.dev20260228`、PyTorch `2.5.1`、timm `0.9.16`。官方代码版本为 `48faa0fea4b08d74188b3841771aca6ff2c92852`。

五套权重均通过三个随机种子、batch=2 的前向对齐，以及 CPU 导出推理检查。分类训练使用两张固定 ImageNet 样本、1000 类分类头和 AdamW，运行六步：

| 型号 | Logits 最大绝对误差 | 初始 / 最终 Loss |
| --- | --- | --- |
| S0 | 1.23978e-5 | 6.88825 / 1.17533 |
| S1 | 1.04904e-5 | 6.90880 / 1.45180 |
| S2 | 1.52588e-5 | 6.85909 / 1.42821 |
| B | 1.23978e-5 | 7.09592 / 1.03836 |
| B-LT | 1.33514e-5 | 7.17074 / 1.44730 |

五个型号均通过 `lite_train_lite_infer` 的 CPU 训练、评估、导出和部署预测，公共脚本同时通过 ResNet18 回归。`build_model`、图像预训练加载、`class_num=0` 特征提取、自定义类别数和分类 checkpoint 回载接口均已验证。

小样本训练仅验证可训练性；完整 ImageNet 精度和 GPU、AMP、多卡 TIPC 尚未验证。

上游许可证和第三方声明见 [MobileCLIP/licenses](MobileCLIP/licenses)。
