# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn

class RAMLoss(nn.Layer):

    def __init__(self, mode="pretrain"):
        super().__init__()
        self.mode = mode

    ## **kwargs 参数没有实际意义，仅作为兼容框架本身所使用的参数
    def forward(self, loss_tag, loss_dis, loss_alignment, **kwargs):
        loss = loss_tag + loss_dis + loss_alignment
        return {"RAMLoss": loss}