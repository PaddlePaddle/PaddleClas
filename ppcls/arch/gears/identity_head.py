from paddle import nn
import paddle

class IdentityHead(nn.Layer):
    def __init__(self):
        super(IdentityHead, self).__init__()

    def forward(self, x, label=None):
        return {"features": x, "logits": None}
