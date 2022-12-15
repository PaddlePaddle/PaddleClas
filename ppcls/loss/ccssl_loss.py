from ppcls.engine.train.train import forward
from .softsuploss import SoftSupConLoss
import copy
import paddle.nn as nn


class CCSSLCeLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CCSSLCeLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, batch, **kwargs):
        p_targets_u_w = batch['p_targets_u_w']
        logits_s1 = batch['logits_s1']
        mask = batch['mask']
        loss_u = self.celoss(logits_s1, p_targets_u_w) * mask
        loss_u = loss_u.mean()

        return {'CCSSLCeLoss': loss_u}
