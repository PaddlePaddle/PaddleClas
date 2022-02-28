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


class KLDivLoss(nn.Layer):
    """
    Distilling the Knowledge in a Neural Network
    """

    def __init__(self, temperature=4):
        super(KLDivLoss, self).__init__()
        self.T = temperature

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, axis=1)
        p_t = F.softmax(y_t / self.T, axis=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return {"loss_kldiv": loss}
