# 模型量化

模型量化是 [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 的特色功能之一，支持动态和静态两种量化训练方式，对权重全局量化和 Channel-Wise 量化，同时以兼容 Paddle-Lite 的格式保存模型。
[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 使用该量化工具，量化了78.9%的mobilenet_v3_large_x1_0的蒸馏模型, 量化后SD855上预测速度从19.308ms加速到14.395ms，存储大小从21M减小到10M， top1识别准确率75.9%。
具体的训练方法可以参见 [PaddleSlim 量化训练](https://paddlepaddle.github.io/PaddleSlim/quick_start/quant_aware_tutorial.html)。
