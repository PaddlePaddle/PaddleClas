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

import paddle.nn as nn
import paddle.nn.functional as F


class ATLoss(nn.Layer):
    def __init__(self, p=2):
        super(ATLoss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        loss = sum(
            [self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return {"loss_at": loss}

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).reshape([f.shape[0], -1]))
