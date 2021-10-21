import paddle
from paddle.nn import Tanh
from ppcls.arch.backbone.legendary_models.lcnet import VGG19
 
__all__ = ["LcNetTanh"]
 
 
class TanhSuffix(paddle.nn.Layer):
    def __init__(self, origin_layer):
        super(TanhSuffix, self).__init__()
        self.origin_layer = origin_layer
        self.tanh = Tanh()
 
    def forward(self, input, res_dict=None, **kwargs):
        x = self.origin_layer(input)
        x = self.tanh(x)
        return x
 
 
def LcNetTanh(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(origin_layer):
        new_layer = LcNetTanh(origin_layer)
        return new_layer
 
    match_re = "linear_2"
    model = VGG19(pretrained=pretrained, use_ssld=use_ssld, **kwargs)
    model.replace_sub(match_re, replace_function, True)
    return model
