# LCNet series

## Overview

The LCNet series is a network that has excellent performance on Intel-CPU proposed by the Baidu PaddleCV team. The author summarizes some methods that can improve the accuracy of the model on Intel-CPU but hardly increase the inference time. The author combines these methods into a new network, namely LCNet. Compared with other lightweight networks, LCNet can achieve higher accuracy with the same inference time. LCNet has shown strong competitiveness in image classification, object detection, and semantic segmentation.



## Accuracy, FLOPS and Parameters

| Models           | Top1 | Top5 | FLOPs<br>(M) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LCNet_x0_25        |0.5186           | 0.7565           | 18    | 1.5  |
| LCNet_x0_35        |0.5809           | 0.8083           | 29    | 1.6  |
| LCNet_x0_5         |0.6314           | 0.8466           | 47    | 1.9  |
| LCNet_x0_75        |0.6818           | 0.8830           | 99    | 2.4  |
| LCNet_x1_0         |0.7132           | 0.9003           | 161   | 3.0  |
| LCNet_x1_5         |0.7371           | 0.9153           | 342   | 4.5  |
| LCNet_x2_0         |0.7518           | 0.9227           | 590   | 6.5  |
| LCNet_x2_5         |0.7660           | 0.9300           | 906   | 9.0  |
| LCNet_x0_5_ssld    |0.6610           | 0.8646           | 47    | 1.9  |
| LCNet_x1_0_ssld    |0.7439           | 0.9209           | 161   | 3.0  |
| LCNet_x2_5_ssld    |0.8082           | 0.9533           | 906   | 9.0  |



## Inference speed based on Intel(R)-Xeon(R)-Gold-6148-CPU

| Models                 | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|------------------|-----------|-------------------|--------------------------|
| LCNet_x0_25        | 224       | 256               | 1.74                    |
| LCNet_x0_35        | 224       | 256               | 1.92                    |
| LCNet_x0_5         | 224       | 256               | 2.05                    |
| LCNet_x0_75        | 224       | 256               | 2.29                    |
| LCNet_x1_0         | 224       | 256               | 2.46                    |
| LCNet_x1_5         | 224       | 256               | 3.19                    |
| LCNet_x2_0         | 224       | 256               | 4.27                    |
| LCNet_x2_5         | 224       | 256               | 5.39                    |
| LCNet_x0_5_ssld    | 224       | 256               | 2.05                    |
| LCNet_x1_0_ssld    | 224       | 256               | 2.46                    |
| LCNet_x2_5_ssld    | 224       | 256               | 5.39                    |
