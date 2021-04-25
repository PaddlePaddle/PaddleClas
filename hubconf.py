# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


dependencies = ['paddle', 'numpy']

import paddle

from ppcls.modeling.architectures import alexnet as _alexnet
from ppcls.modeling.architectures import vgg as _vgg 
from ppcls.modeling.architectures import resnet as _resnet 
from ppcls.modeling.architectures import squeezenet as _squeezenet
from ppcls.modeling.architectures import densenet as _densenet
from ppcls.modeling.architectures import inception_v3 as _inception_v3
from ppcls.modeling.architectures import inception_v4 as _inception_v4
from ppcls.modeling.architectures import googlenet as _googlenet
from ppcls.modeling.architectures import shufflenet_v2 as _shufflenet_v2
from ppcls.modeling.architectures import mobilenet_v1 as _mobilenet_v1
from ppcls.modeling.architectures import mobilenet_v2 as _mobilenet_v2
from ppcls.modeling.architectures import mobilenet_v3 as _mobilenet_v3
from ppcls.modeling.architectures import resnext as _resnext


def _load_pretrained_urls():
    '''Load pretrained model parameters url from README.md
    '''
    import re
    import os
    from collections import OrderedDict

    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')

    with open(readme_path, 'r') as f:
        lines = f.readlines()
        lines = [lin for lin in lines if lin.strip().startswith('|') and 'Download link' in lin]
    
    urls = OrderedDict()
    for lin in lines:
        try:
            name = re.findall(r'\|(.*?)\|', lin)[0].strip().replace('<br>', '')
            url = re.findall(r'\((.*?)\)', lin)[-1].strip()
            if name in url:
                urls[name] = url
        except:
            pass

    return urls


_checkpoints = _load_pretrained_urls()


def _load_pretrained_parameters(model, name):
    assert name in _checkpoints, 'Not provide {} pretrained model.'.format(name)
    path = paddle.utils.download.get_weights_path_from_url(_checkpoints[name])
    model.set_state_dict(paddle.load(path))
    return model
    

def AlexNet(pretrained=False, **kwargs):
    '''AlexNet
    '''
    model = _alexnet.AlexNet(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'AlexNet')

    return model



def VGG11(pretrained=False, **kwargs):
    '''VGG11
    '''
    model = _vgg.VGG11(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'VGG11')

    return model


def VGG13(pretrained=False, **kwargs):
    '''VGG13
    '''
    model = _vgg.VGG13(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'VGG13')

    return model


def VGG16(pretrained=False, **kwargs):
    '''VGG16
    '''
    model = _vgg.VGG16(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'VGG16')

    return model


def VGG19(pretrained=False, **kwargs):
    '''VGG19
    '''
    model = _vgg.VGG19(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'VGG19')

    return model




def ResNet18(pretrained=False, **kwargs):
    '''ResNet18
    '''
    model = _resnet.ResNet18(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNet18')

    return model


def ResNet34(pretrained=False, **kwargs):
    '''ResNet34
    '''
    model = _resnet.ResNet34(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNet34')

    return model


def ResNet50(pretrained=False, **kwargs):
    '''ResNet50
    '''
    model = _resnet.ResNet50(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNet50')
        
    return model


def ResNet101(pretrained=False, **kwargs):
    '''ResNet101
    '''
    model = _resnet.ResNet101(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNet101')

    return model


def ResNet152(pretrained=False, **kwargs):
    '''ResNet152
    '''
    model = _resnet.ResNet152(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNet152')

    return model



def SqueezeNet1_0(pretrained=False, **kwargs):
    '''SqueezeNet1_0
    '''
    model = _squeezenet.SqueezeNet1_0(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'SqueezeNet1_0')

    return model


def SqueezeNet1_1(pretrained=False, **kwargs):
    '''SqueezeNet1_1
    '''
    model = _squeezenet.SqueezeNet1_1(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'SqueezeNet1_1')

    return model




def DenseNet121(pretrained=False, **kwargs):
    '''DenseNet121
    '''
    model = _densenet.DenseNet121(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'DenseNet121')

    return model


def DenseNet161(pretrained=False, **kwargs):
    '''DenseNet161
    '''
    model = _densenet.DenseNet161(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'DenseNet161')

    return model


def DenseNet169(pretrained=False, **kwargs):
    '''DenseNet169
    '''
    model = _densenet.DenseNet169(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'DenseNet169')

    return model


def DenseNet201(pretrained=False, **kwargs):
    '''DenseNet201
    '''
    model = _densenet.DenseNet201(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'DenseNet201')

    return model


def DenseNet264(pretrained=False, **kwargs):
    '''DenseNet264
    '''
    model = _densenet.DenseNet264(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'DenseNet264')

    return model



def InceptionV3(pretrained=False, **kwargs):
    '''InceptionV3
    '''
    model = _inception_v3.InceptionV3(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'InceptionV3')

    return model


def InceptionV4(pretrained=False, **kwargs):
    '''InceptionV4
    '''
    model = _inception_v4.InceptionV4(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'InceptionV4')

    return model



def GoogLeNet(pretrained=False, **kwargs):
    '''GoogLeNet
    '''
    model = _googlenet.GoogLeNet(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'GoogLeNet')

    return model



def ShuffleNet(pretrained=False, **kwargs):
    '''ShuffleNet
    '''
    model = _shufflenet_v2.ShuffleNet(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ShuffleNet')

    return model



def MobileNetV1(pretrained=False, **kwargs):
    '''MobileNetV1
    '''
    model = _mobilenet_v1.MobileNetV1(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV1')

    return model


def MobileNetV1_x0_25(pretrained=False, **kwargs):
    '''MobileNetV1_x0_25
    '''
    model = _mobilenet_v1.MobileNetV1_x0_25(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV1_x0_25')

    return model


def MobileNetV1_x0_5(pretrained=False, **kwargs):
    '''MobileNetV1_x0_5
    '''
    model = _mobilenet_v1.MobileNetV1_x0_5(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV1_x0_5')

    return model


def MobileNetV1_x0_75(pretrained=False, **kwargs):
    '''MobileNetV1_x0_75
    '''
    model = _mobilenet_v1.MobileNetV1_x0_75(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV1_x0_75')

    return model


def MobileNetV2_x0_25(pretrained=False, **kwargs):
    '''MobileNetV2_x0_25
    '''
    model = _mobilenet_v2.MobileNetV2_x0_25(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV2_x0_25')

    return model


def MobileNetV2_x0_5(pretrained=False, **kwargs):
    '''MobileNetV2_x0_5
    '''
    model = _mobilenet_v2.MobileNetV2_x0_5(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV2_x0_5')

    return model


def MobileNetV2_x0_75(pretrained=False, **kwargs):
    '''MobileNetV2_x0_75
    '''
    model = _mobilenet_v2.MobileNetV2_x0_75(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV2_x0_75')

    return model


def MobileNetV2_x1_5(pretrained=False, **kwargs):
    '''MobileNetV2_x1_5
    '''
    model = _mobilenet_v2.MobileNetV2_x1_5(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV2_x1_5')

    return model


def MobileNetV2_x2_0(pretrained=False, **kwargs):
    '''MobileNetV2_x2_0
    '''
    model = _mobilenet_v2.MobileNetV2_x2_0(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV2_x2_0')

    return model


def MobileNetV3_large_x0_35(pretrained=False, **kwargs):
    '''MobileNetV3_large_x0_35
    '''
    model = _mobilenet_v3.MobileNetV3_large_x0_35(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_large_x0_35')

    return model


def MobileNetV3_large_x0_5(pretrained=False, **kwargs):
    '''MobileNetV3_large_x0_5
    '''
    model = _mobilenet_v3.MobileNetV3_large_x0_5(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_large_x0_5')

    return model


def MobileNetV3_large_x0_75(pretrained=False, **kwargs):
    '''MobileNetV3_large_x0_75
    '''
    model = _mobilenet_v3.MobileNetV3_large_x0_75(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_large_x0_75')

    return model


def MobileNetV3_large_x1_0(pretrained=False, **kwargs):
    '''MobileNetV3_large_x1_0
    '''
    model = _mobilenet_v3.MobileNetV3_large_x1_0(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_large_x1_0')

    return model


def MobileNetV3_large_x1_25(pretrained=False, **kwargs):
    '''MobileNetV3_large_x1_25
    '''
    model = _mobilenet_v3.MobileNetV3_large_x1_25(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_large_x1_25')

    return model


def MobileNetV3_small_x0_35(pretrained=False, **kwargs):
    '''MobileNetV3_small_x0_35
    '''
    model = _mobilenet_v3.MobileNetV3_small_x0_35(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_small_x0_35')

    return model


def MobileNetV3_small_x0_5(pretrained=False, **kwargs):
    '''MobileNetV3_small_x0_5
    '''
    model = _mobilenet_v3.MobileNetV3_small_x0_5(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_small_x0_5')

    return model


def MobileNetV3_small_x0_75(pretrained=False, **kwargs):
    '''MobileNetV3_small_x0_75
    '''
    model = _mobilenet_v3.MobileNetV3_small_x0_75(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_small_x0_75')

    return model


def MobileNetV3_small_x1_0(pretrained=False, **kwargs):
    '''MobileNetV3_small_x1_0
    '''
    model = _mobilenet_v3.MobileNetV3_small_x1_0(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_small_x1_0')

    return model


def MobileNetV3_small_x1_25(pretrained=False, **kwargs):
    '''MobileNetV3_small_x1_25
    '''
    model = _mobilenet_v3.MobileNetV3_small_x1_25(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'MobileNetV3_small_x1_25')

    return model



def ResNeXt101_32x4d(pretrained=False, **kwargs):
    '''ResNeXt101_32x4d
    '''
    model = _resnext.ResNeXt101_32x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt101_32x4d')

    return model


def ResNeXt101_64x4d(pretrained=False, **kwargs):
    '''ResNeXt101_64x4d
    '''
    model = _resnext.ResNeXt101_64x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt101_64x4d')

    return model


def ResNeXt152_32x4d(pretrained=False, **kwargs):
    '''ResNeXt152_32x4d
    '''
    model = _resnext.ResNeXt152_32x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt152_32x4d')

    return model


def ResNeXt152_64x4d(pretrained=False, **kwargs):
    '''ResNeXt152_64x4d
    '''
    model = _resnext.ResNeXt152_64x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt152_64x4d')

    return model


def ResNeXt50_32x4d(pretrained=False, **kwargs):
    '''ResNeXt50_32x4d
    '''
    model = _resnext.ResNeXt50_32x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt50_32x4d')

    return model


def ResNeXt50_64x4d(pretrained=False, **kwargs):
    '''ResNeXt50_64x4d
    '''
    model = _resnext.ResNeXt50_64x4d(**kwargs)
    if pretrained:
        model = _load_pretrained_parameters(model, 'ResNeXt50_64x4d')

    return model