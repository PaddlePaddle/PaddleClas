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


class WSLLoss(nn.Layer):
    """
    Weighted Soft Labels Loss
    paper: https://arxiv.org/pdf/2102.00650.pdf
    code reference: https://github.com/bellymonster/Weighted-Soft-Label-Distillation
    """

    def __init__(self, temperature=2.0, use_target_as_gt=False):
        super().__init__()
        self.temperature = temperature
        self.use_target_as_gt = use_target_as_gt

    def forward(self, logits_student, logits_teacher, target=None):
        """Compute weighted soft labels loss.
        Args:
            logits_student: student's logits with shape (batch_size, num_classes)
            logits_teacher: teacher's logits with shape (batch_size, num_classes)
            target: ground truth labels with shape (batch_size)
        """
        if target is None or self.use_target_as_gt:
            target = logits_teacher.argmax(axis=-1)

        target = F.one_hot(
            target.reshape([-1]), num_classes=logits_student[0].shape[0])

        s_input_for_softmax = logits_student / self.temperature
        t_input_for_softmax = logits_teacher / self.temperature

        ce_loss_s = -paddle.sum(target *
                                F.log_softmax(logits_student.detach()),
                                axis=1)
        ce_loss_t = -paddle.sum(target *
                                F.log_softmax(logits_teacher.detach()),
                                axis=1)

        ratio = ce_loss_s / (ce_loss_t + 1e-7)
        ratio = paddle.maximum(ratio, paddle.zeros_like(ratio))

        kd_loss = -paddle.sum(F.softmax(t_input_for_softmax) *
                              F.log_softmax(s_input_for_softmax),
                              axis=1)
        weight = 1 - paddle.exp(-ratio)

        weighted_kd_loss = (self.temperature**2) * paddle.mean(kd_loss *
                                                               weight)

        return weighted_kd_loss
