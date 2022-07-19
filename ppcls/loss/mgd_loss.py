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
from ppcls.utils.initializer import kaiming_normal_


class MGDLoss(nn.Layer):
    """Paddle version of `Masked Generative Distillation`
    MGDLoss
    Reference: https://arxiv.org/abs/2205.01529
    Code was heavily based on https://github.com/yzd-v/MGD
    """

    def __init__(
            self,
            student_channels,
            teacher_channels,
            alpha_mgd=1.756,
            lambda_mgd=0.15, ):
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2D(
                teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                teacher_channels, teacher_channels, kernel_size=3, padding=1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, pred_s, pred_t):
        """Forward function.
        Args:
            pred_s(Tensor): Bs*C*H*W, student's feature map
            pred_t(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert pred_s.shape[-2:] == pred_t.shape[-2:]

        if self.align is not None:
            pred_s = self.align(pred_s)

        loss = self.get_dis_loss(pred_s, pred_t) * self.alpha_mgd

        return loss

    def get_dis_loss(self, pred_s, pred_t):
        loss_mse = nn.MSELoss(reduction='mean')
        N, C, _, _ = pred_t.shape
        mat = paddle.rand([N, C, 1, 1])
        mat = paddle.where(mat < self.lambda_mgd, 0, 1).astype("float32")
        masked_fea = paddle.multiply(pred_s, mat)
        new_fea = self.generation(masked_fea)
        dis_loss = loss_mse(new_fea, pred_t)
        return dis_loss
