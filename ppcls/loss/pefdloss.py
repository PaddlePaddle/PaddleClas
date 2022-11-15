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

from ppcls.utils.initializer import kaiming_normal_, kaiming_uniform_


class Regressor(nn.Layer):
    """Linear regressor"""

    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regressor, self).__init__()
        self.conv = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class PEFDLoss(nn.Layer):
    """Improved Feature Distillation via Projector Ensemble
    Reference: https://arxiv.org/pdf/2210.15274.pdf
    Code reference: https://github.com/chenyd7/PEFD
    """

    def __init__(self,
                 student_channel,
                 teacher_channel,
                 num_projectors=3,
                 mode="flatten"):
        super().__init__()

        if num_projectors <= 0:
            raise ValueError("Number of projectors must be greater than 0.")

        if mode not in ["flatten", "gap"]:
            raise ValueError("Mode must be \"flatten\" or \"gap\".")

        self.mode = mode
        self.projectors = nn.LayerList()

        for _ in range(num_projectors):
            self.projectors.append(Regressor(student_channel, teacher_channel))

    def forward(self, student_feature, teacher_feature):
        if self.mode == "gap":
            student_feature = F.adaptive_avg_pool2d(student_feature, (1, 1))
            teacher_feature = F.adaptive_avg_pool2d(teacher_feature, (1, 1))

        student_feature = student_feature.flatten(1)
        f_t = teacher_feature.flatten(1)

        q = len(self.projectors)
        f_s = 0.0
        for i in range(q):
            f_s += self.projectors[i](student_feature)
        f_s = f_s / q

        # inner product (normalize first and inner product)
        normft = f_t.pow(2).sum(1, keepdim=True).pow(1. / 2)
        outft = f_t / normft
        normfs = f_s.pow(2).sum(1, keepdim=True).pow(1. / 2)
        outfs = f_s / normfs

        cos_theta = (outft * outfs).sum(1, keepdim=True)
        loss = paddle.mean(1 - cos_theta)

        return loss
