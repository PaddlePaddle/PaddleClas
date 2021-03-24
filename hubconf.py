
dependencies = ['paddle', 'numpy']


from ppcls.modeling.architectures.resnet import ResNet18 as _ResNet18
from ppcls.modeling.architectures.resnet import ResNet34 as _ResNet34
from ppcls.modeling.architectures.resnet import ResNet50 as _ResNet50



def ResNet18(**kwargs):
    '''ResNet18
    '''
    model = _ResNet18(**kwargs)
    return model



def ResNet34(**kwargs):
    '''ResNet34
    '''
    model = _ResNet18(**kwargs)
    return model



def ResNet50(**kwargs):
    '''ResNet50
    '''
    model = _ResNet18(**kwargs)
    return model