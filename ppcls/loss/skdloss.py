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


class SKDLoss(nn.Layer):
    """
    Spherical Knowledge Distillation
    paper: https://arxiv.org/pdf/2010.07485.pdf
    code reference: https://github.com/forjiuzhou/Spherical-Knowledge-Distillation
    """

    def __init__(self,
                 temperature,
                 multiplier=2.0,
                 alpha=0.9,
                 use_target_as_gt=False):
        super().__init__()
        self.temperature = temperature
        self.multiplier = multiplier
        self.alpha = alpha
        self.use_target_as_gt = use_target_as_gt

    def forward(self, logits_student, logits_teacher, target=None):
        """Compute Spherical Knowledge Distillation loss.
        Args:
            logits_student: student's logits with shape (batch_size, num_classes)
            logits_teacher: teacher's logits with shape (batch_size, num_classes)
        """
        if target is None or self.use_target_as_gt:
            target = logits_teacher.argmax(axis=-1)

        target = F.one_hot(
            target.reshape([-1]), num_classes=logits_student[0].shape[0])

        logits_student = F.layer_norm(
            logits_student,
            logits_student.shape[1:],
            weight=None,
            bias=None,
            epsilon=1e-7) * self.multiplier
        logits_teacher = F.layer_norm(
            logits_teacher,
            logits_teacher.shape[1:],
            weight=None,
            bias=None,
            epsilon=1e-7) * self.multiplier

        kd_loss = -paddle.sum(F.softmax(logits_teacher / self.temperature) *
                              F.log_softmax(logits_student / self.temperature),
                              axis=1)

        kd_loss = paddle.mean(kd_loss) * self.temperature**2

        ce_loss = paddle.mean(-paddle.sum(
            target * F.log_softmax(logits_student), axis=1))

        return kd_loss * self.alpha + ce_loss * (1 - self.alpha)
