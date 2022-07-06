# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(axis=1) * b.norm(axis=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose([1, 0]), y_t.transpose([1, 0]))


class DISTLoss(nn.Layer):
    # DISTLoss
    # paper [Knowledge Distillation from A Stronger Teacher](https://arxiv.org/pdf/2205.10536v1.pdf)
    # code reference: https://github.com/hunto/image_classification_sota/blob/d4f15a0494/lib/models/losses/dist_kd.py
    def __init__(self, beta=1.0, gamma=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = F.softmax(z_s, axis=-1)
        y_t = F.softmax(z_t, axis=-1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss
