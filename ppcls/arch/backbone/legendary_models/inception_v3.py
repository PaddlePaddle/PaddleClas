# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain

__all__ = ["InceptionV3"]


# InceptionV3 config
# key: inception blocks 
# value: conv num in different blocks
NET_CONFIG = {
    'inception_a':[[192, 256, 288], [32, 64, 64]], 
    'inception_b':[288],   
    'inception_c':[[768, 768, 768, 768], [128, 160, 160, 192]],
    'inception_d':[768],   
    'inception_e':[1280,2048]
}


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act="relu"):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)
        self.batch_norm = BatchNorm(
            num_filters,
            act=act)

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.batch_norm(y)
        return y

class InceptionStem(TheseusLayer):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.conv_1a_3x3 = ConvBNLayer(num_channels=3,
                                       num_filters=32,
                                       filter_size=3,
                                       stride=2,
                                       act="relu")
        self.conv_2a_3x3 = ConvBNLayer(num_channels=32,
                                       num_filters=32,
                                       filter_size=3,
                                       stride=1,
                                       act="relu")
        self.conv_2b_3x3 = ConvBNLayer(num_channels=32,
                                       num_filters=64,
                                       filter_size=3,
                                       padding=1,
                                       act="relu")

        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=0)
        self.conv_3b_1x1 = ConvBNLayer(num_channels=64,
                                       num_filters=80,
                                       filter_size=1,
                                       act="relu")        
        self.conv_4a_3x3 = ConvBNLayer(num_channels=80,
                                       num_filters=192,
                                       filter_size=3,
                                       act="relu")
    def forward(self, x):
        y = self.conv_1a_3x3(x)
        y = self.conv_2a_3x3(y)
        y = self.conv_2b_3x3(y)
        y = self.maxpool(y)
        y = self.conv_3b_1x1(y)
        y = self.conv_4a_3x3(y)
        y = self.maxpool(y)
        return y

                         
class InceptionA(TheseusLayer):
    def __init__(self, num_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels,
                                     num_filters=64,
                                     filter_size=1,
                                     act="relu") 
        self.branch5x5_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=48, 
                                       filter_size=1, 
                                       act="relu")
        self.branch5x5_2 = ConvBNLayer(num_channels=48, 
                                       num_filters=64, 
                                       filter_size=5, 
                                       padding=2, 
                                       act="relu")

        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(num_channels=64,
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu")
        self.branch3x3dbl_3 = ConvBNLayer(num_channels=96,
                               num_filters=96, 
                               filter_size=3, 
                               padding=1,
                               act="relu")
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels,
                               num_filters=pool_features, 
                               filter_size=1, 
                               act="relu")

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)
        outputs = paddle.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=1)
        return outputs

    
class InceptionB(TheseusLayer):
    def __init__(self, num_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = ConvBNLayer(num_channels=num_channels,
                                     num_filters=384,
                                     filter_size=3,
                                     stride=2,
                                     act="relu") 
        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=64, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(num_channels=64, 
                                       num_filters=96, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu")
        self.branch3x3dbl_3 = ConvBNLayer(num_channels=96, 
                                       num_filters=96, 
                                       filter_size=3,
                                       stride=2,
                                       act="relu")
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)
        
    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.branch_pool(x)

        outputs = paddle.concat([branch3x3, branch3x3dbl, branch_pool], axis=1)

        return outputs

class InceptionC(TheseusLayer):
    def __init__(self, num_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu")


        self.branch7x7_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       stride=1,
                                       act="relu")
        self.branch7x7_2 = ConvBNLayer(num_channels=channels_7x7,
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       stride=1,
                                       padding=(0, 3),
                                       act="relu")
        self.branch7x7_3 = ConvBNLayer(num_channels=channels_7x7,
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       stride=1,
                                       padding=(3, 0),
                                       act="relu")
        
        self.branch7x7dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=channels_7x7, 
                                       filter_size=1, 
                                       act="relu")
        self.branch7x7dbl_2 = ConvBNLayer(num_channels=channels_7x7,  
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu")
        self.branch7x7dbl_3 = ConvBNLayer(num_channels=channels_7x7, 
                                       num_filters=channels_7x7, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu")
        self.branch7x7dbl_4 = ConvBNLayer(num_channels=channels_7x7,  
                                       num_filters=channels_7x7, 
                                       filter_size=(7, 1), 
                                       padding = (3, 0),
                                       act="relu")
        self.branch7x7dbl_5 = ConvBNLayer(num_channels=channels_7x7, 
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding = (0, 3),
                                       act="relu")
       
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels,
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu")
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        outputs = paddle.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=1)
        
        return outputs
    
class InceptionD(TheseusLayer):
    def __init__(self, num_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3_2 = ConvBNLayer(num_channels=192, 
                                       num_filters=320, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu")
        self.branch7x7x3_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu")
        self.branch7x7x3_2 = ConvBNLayer(num_channels=192,
                                       num_filters=192, 
                                       filter_size=(1, 7), 
                                       padding=(0, 3),
                                       act="relu")
        self.branch7x7x3_3 = ConvBNLayer(num_channels=192, 
                                       num_filters=192, 
                                       filter_size=(7, 1), 
                                       padding=(3, 0),
                                       act="relu")
        self.branch7x7x3_4 = ConvBNLayer(num_channels=192,  
                                       num_filters=192, 
                                       filter_size=3, 
                                       stride=2,
                                       act="relu")
        self.branch_pool = MaxPool2D(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = self.branch_pool(x)
        
        outputs = paddle.concat([branch3x3, branch7x7x3, branch_pool], axis=1)
        return outputs
    
class InceptionE(TheseusLayer):
    def __init__(self, num_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=320, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3_1 = ConvBNLayer(num_channels=num_channels,
                                       num_filters=384, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3_2a = ConvBNLayer(num_channels=384, 
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu")
        self.branch3x3_2b = ConvBNLayer(num_channels=384, 
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu")
        
        self.branch3x3dbl_1 = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=448, 
                                       filter_size=1, 
                                       act="relu")
        self.branch3x3dbl_2 = ConvBNLayer(num_channels=448, 
                                       num_filters=384, 
                                       filter_size=3, 
                                       padding=1,
                                       act="relu")
        self.branch3x3dbl_3a = ConvBNLayer(num_channels=384,
                                       num_filters=384, 
                                       filter_size=(1, 3), 
                                       padding=(0, 1),
                                       act="relu")
        self.branch3x3dbl_3b = ConvBNLayer(num_channels=384,
                                       num_filters=384, 
                                       filter_size=(3, 1), 
                                       padding=(1, 0),
                                       act="relu")
        self.branch_pool = AvgPool2D(kernel_size=3, stride=1, padding=1, exclusive=False)
        self.branch_pool_conv = ConvBNLayer(num_channels=num_channels, 
                                       num_filters=192, 
                                       filter_size=1, 
                                       act="relu")
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = paddle.concat(branch3x3, axis=1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = paddle.concat(branch3x3dbl, axis=1)

        branch_pool = self.branch_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        outputs = paddle.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=1)
        return outputs   


class Inception_V3(TheseusLayer):

    def __init__(self, 
                 config, 
                 class_num=1000, 
                 pretrained=False,
                 **kwargs):
        super(Inception_V3, self).__init__()

        self.inception_a_list = config['inception_a']
        self.inception_c_list = config['inception_c']
        self.inception_b_list = config['inception_b']
        self.inception_d_list = config['inception_d']
        self.inception_e_list = config ['inception_e']
        
        self.inception_stem = InceptionStem()

        self.inception_block_list = nn.LayerList()
        for i in range(len(self.inception_a_list[0])):
            inception_a = InceptionA(self.inception_a_list[0][i], 
                                     self.inception_a_list[1][i])
            self.inception_block_list.append(inception_a)

        for i in range(len(self.inception_b_list)):
            inception_b = InceptionB(self.inception_b_list[i])
            self.inception_block_list.append(inception_b)

        for i in range(len(self.inception_c_list[0])):
            inception_c = InceptionC(self.inception_c_list[0][i], 
                                     self.inception_c_list[1][i])
            self.inception_block_list.append(inception_c)

        for i in range(len(self.inception_d_list)):
            inception_d = InceptionD(self.inception_d_list[i])
            self.inception_block_list.append(inception_d)

        for i in range(len(self.inception_e_list)):
            inception_e = InceptionE(self.inception_e_list[i])
            self.inception_block_list.append(inception_e)
 
        self.gap = AdaptiveAvgPool2D(1)
        self.drop = Dropout(p=0.2, mode="downscale_in_infer")
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.out = Linear(
            2048,
            class_num,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv)),
            bias_attr=ParamAttr())

        if pretrained is not None:
            load_dygraph_pretrain(self, pretrained)

    def forward(self, x):
        y = self.inception_stem(x)
        for inception_block in self.inception_block_list:
           y = inception_block(y)
        y = self.gap(y)
        y = paddle.reshape(y, shape=[-1, 2048])
        y = self.drop(y)
        y = self.out(y)
        return y




def InceptionV3(**kwargs):
    """
    InceptionV3
    Args:
        kwargs: 
            class_num: int=1000. Output dim of last fc layer.
            pretrained: 
    Returns:
        model: nn.Layer. Specific `InceptionV3` model 
    """
    model = Inception_V3(NET_CONFIG, **kwargs)
    return model

