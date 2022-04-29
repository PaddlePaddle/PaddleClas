# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(axis=1)
    prod = paddle.mm(e, e.t())
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clip(
        min=eps)

    if not squared:
        res = res.sqrt()
    return res


class RKdAngle(nn.Layer):
    # paper : [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG)
    # reference: https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    def __init__(self, target_size=None):
        super().__init__()
        if target_size is not None:
            self.avgpool = paddle.nn.AdaptiveAvgPool2D(target_size)
        else:
            self.avgpool = None

    def forward(self, student, teacher):
        # GAP to reduce memory
        if self.avgpool is not None:
            # NxC1xH1xW1 -> NxC1x1x1
            student = self.avgpool(student)
            # NxC2xH2xW2 -> NxC2x1x1
            teacher = self.avgpool(teacher)

        # reshape for feature map distillation
        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, axis=2)
        t_angle = paddle.bmm(norm_td, norm_td.transpose([0, 2, 1])).reshape(
            [-1, 1])

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, axis=2)
        s_angle = paddle.bmm(norm_sd, norm_sd.transpose([0, 2, 1])).reshape(
            [-1, 1])
        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Layer):
    # paper : [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068?context=cs.LG)
    # reference: https://github.com/lenscloth/RKD/blob/master/metric/loss.py
    def __init__(self, eps=1e-12, target_size=1):
        super().__init__()
        self.eps = eps
        if target_size is not None:
            self.avgpool = paddle.nn.AdaptiveAvgPool2D(target_size)
        else:
            self.avgpool = None

    def forward(self, student, teacher):
        # GAP to reduce memory
        if self.avgpool is not None:
            # NxC1xH1xW1 -> NxC1x1x1
            student = self.avgpool(student)
            # NxC2xH2xW2 -> NxC2x1x1
            teacher = self.avgpool(teacher)

        bs = student.shape[0]
        student = student.reshape([bs, -1])
        teacher = teacher.reshape([bs, -1])

        t_d = pdist(teacher, squared=False)
        mean_td = t_d.mean()
        t_d = t_d / (mean_td + self.eps)

        d = pdist(student, squared=False)
        mean_d = d.mean()
        d = d / (mean_d + self.eps)

        loss = F.smooth_l1_loss(d, t_d, reduction="mean")
        return loss
