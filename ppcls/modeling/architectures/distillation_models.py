#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from .resnet_vd import ResNet50_vd
from .mobilenet_v3 import MobileNetV3_large_x1_0
from .resnext101_wsl import ResNeXt101_32x16d_wsl

__all__ = [
    'ResNet50_vd_distill_MobileNetV3_large_x1_0',
    'ResNeXt101_32x16d_wsl_distill_ResNet50_vd'
]


class ResNet50_vd_distill_MobileNetV3_large_x1_0():
    def net(self, input, class_dim=1000):
        # student
        student = MobileNetV3_large_x1_0()
        out_student = student.net(input, class_dim=class_dim)
        # teacher
        teacher = ResNet50_vd()
        out_teacher = teacher.net(input, class_dim=class_dim)
        out_teacher.stop_gradient = True

        return out_teacher, out_student


class ResNeXt101_32x16d_wsl_distill_ResNet50_vd():
    def net(self, input, class_dim=1000):
        # student
        student = ResNet50_vd()
        out_student = student.net(input, class_dim=class_dim)
        # teacher
        teacher = ResNeXt101_32x16d_wsl()
        out_teacher = teacher.net(input, class_dim=class_dim)
        out_teacher.stop_gradient = True

        return out_teacher, out_student
