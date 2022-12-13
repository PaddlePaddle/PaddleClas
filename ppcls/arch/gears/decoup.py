import paddle
import paddle.nn as nn


class Decoup(nn.Layer):
    def __init__(self, logits_index, features_index, **kwargs):
        super(Decoup, self).__init__()
        self.logits_index = logits_index
        self.features_index = features_index


    def forward(self, out, **kwargs):
        assert isinstance(out, (list, tuple)), 'out must  be list or tuple'
        out = {'logits': out[self.logits_index], 'features':out[self.features_index]}
        return out

