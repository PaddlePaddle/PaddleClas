#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn

class DSHSDLoss(nn.Layer):
    """
    # DSHSD(IEEE ACCESS 2019)
    # paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
    # [DSHSD] epoch:70,  bit:48,  dataset:cifar10-1,  MAP:0.809, Best MAP: 0.809
    # [DSHSD] epoch:250, bit:48,  dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
    # [DSHSD] epoch:135, bit:48,  dataset:imagenet,   MAP:0.647, Best MAP: 0.647
    """
    def __init__(self, n_class, bit, alpha, multi_label=False):
        super(DSHSDLoss, self).__init__()
        self.m = 2 * bit     
        self.alpha = alpha
        self.multi_label = multi_label
        self.n_class = n_class
        self.fc = paddle.nn.Linear(bit, n_class, bias_attr=False)

    def forward(self, input, label):        
        feature = input["features"]
        feature = feature.tanh().astype("float32")

        dist = paddle.sum(
                    paddle.square((paddle.unsqueeze(feature, 1) - paddle.unsqueeze(feature, 0))), 
                    axis=2)
        
        # label to ont-hot
        label = paddle.flatten(label)
        label = paddle.nn.functional.one_hot(label,  self.n_class).astype("float32")

        s = (paddle.matmul(label, label, transpose_y=True) == 0).astype("float32")
        Ld = (1 - s) / 2 * dist + s / 2 * (self.m - dist).clip(min=0)
        Ld = Ld.mean()
        
        logits = self.fc(feature)
        if self.multi_label:
            # multiple labels classification loss
            Lc = (logits - label * logits + ((1 + (-logits).exp()).log())).sum(axis=1).mean()
        else:
            # single labels classification loss
            Lc = (-paddle.nn.functional.softmax(logits).log() * label).sum(axis=1).mean()

        return {"dshsdloss": Lc + Ld * self.alpha}

