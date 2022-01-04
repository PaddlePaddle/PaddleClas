import paddle
from paddle.nn import Sigmoid
from ppcls.arch.backbone.legendary_models.vgg import VGG19

__all__ = ["VGG19Sigmoid"]


class SigmoidSuffix(paddle.nn.Layer):
    def __init__(self, origin_layer):
        super().__init__()
        self.origin_layer = origin_layer
        self.sigmoid = Sigmoid()

    def forward(self, input, res_dict=None, **kwargs):
        x = self.origin_layer(input)
        x = self.sigmoid(x)
        return x


def VGG19Sigmoid(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(origin_layer, pattern):
        new_layer = SigmoidSuffix(origin_layer)
        return new_layer

    pattern = "fc2"
    model = VGG19(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    model.upgrade_sublayer(pattern, replace_function)
    return model
