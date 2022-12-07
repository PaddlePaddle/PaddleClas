
from .softsuploss import SoftSupConLoss
import copy
import paddle.nn as nn


class CCSSLLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CCSSLLoss, self).__init__()
        ce_cfg = copy.deepcopy(kwargs['CELoss'])
        self.ce_weight = ce_cfg.pop('weight')
        softsupconloss_cfg = copy.deepcopy(kwargs['SoftSupConLoss'])
        self.softsupconloss_weight = softsupconloss_cfg.pop('weight')

        self.softsuploss = SoftSupConLoss(**softsupconloss_cfg)
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feats, batch, **kwargs):
        """
        Args:
            feats: feature of s1 and s2, (n, 2, d)
            batch: dict 
        """
        logits_w = batch['logits_w']
        logits_s1 = batch['logits_s1']
        p_targets_u_w = batch['p_targets_u_w']
        mask = batch['mask']

        max_probs = batch['max_probs']
        # reduction = batch['reduction']
       

        loss_u = self.celoss(logits_s1, p_targets_u_w) * mask
        loss_u = loss_u.mean()

        loss_c = self.softsuploss(feats, max_probs, p_targets_u_w)

        return {'CCSSLLoss': self.ce_weight*loss_u + self.softsupconloss_weight * loss_c}








