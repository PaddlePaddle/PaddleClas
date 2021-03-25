
dependencies = ['paddle', 'numpy']

import paddle

from ppcls.modeling.architectures.resnet import ResNet18 as _ResNet18
from ppcls.modeling.architectures.resnet import ResNet34 as _ResNet34
from ppcls.modeling.architectures.resnet import ResNet50 as _ResNet50


_checkpoints = {
    'ResNet18': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_pretrained.pdparams'
}



def ResNet18(**kwargs):
    '''ResNet18
    '''
    pretrained = kwargs.pop('pretrained', False)

    model = _ResNet18(**kwargs)
    if pretrained:
        path = paddle.utils.download.get_weights_path_from_url(_checkpoints['ResNet18'])
        model.set_state_dict(paddle.load(path))

    return model



def ResNet34(**kwargs):
    '''ResNet34
    '''
    model = _ResNet34(**kwargs)
    return model



def ResNet50(**kwargs):
    '''ResNet50
    '''
    model = _ResNet50(**kwargs)
    return model