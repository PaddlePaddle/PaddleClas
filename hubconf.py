
dependencies = ['paddle', 'numpy']

import paddle

from ppcls.modeling.architectures import resnet as _resnet 


# _checkpoints = {
#     'ResNet18': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_pretrained.pdparams',
#     'ResNet34': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_pretrained.pdparams',
# }

def _load_pretrained_urls():
    '''Load pretrained model parameters url from README.md
    '''
    import re
    from collections import OrderedDict

    with open('./README.md', 'r') as f:
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


def ResNet18(**kwargs):
    '''ResNet18
    '''
    pretrained = kwargs.pop('pretrained', False)

    model = _resnet.ResNet18(**kwargs)
    if pretrained:
        assert 'ResNet18' in _checkpoints, 'Not provide `ResNet18` pretrained model.'
        path = paddle.utils.download.get_weights_path_from_url(_checkpoints['ResNet18'])
        model.set_state_dict(paddle.load(path))

    return model



def ResNet34(**kwargs):
    '''ResNet34
    '''
    pretrained = kwargs.pop('pretrained', False)

    model = _resnet.ResNet34(**kwargs)
    if pretrained:
        assert 'ResNet34' in _checkpoints, 'Not provide `ResNet34` pretrained model.'
        path = paddle.utils.download.get_weights_path_from_url(_checkpoints['ResNet34'])
        model.set_state_dict(paddle.load(path))

    return model


