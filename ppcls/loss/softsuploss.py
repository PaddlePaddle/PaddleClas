# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

class SoftSupConLoss(nn.Layer):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def __call__(self, feat, batch, max_probs=None, labels=None, mask=None, reduction="mean", select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            feat: hidden vector of shape [batch_size, n_views, ...].
            labels: ground truth of shape [batch_size].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        max_probs = batch['max_probs']
        labels = batch['p_targets_u_w']
        # reduction = batch['reduction']
        batch_size = feat.shape[0]
        if labels is not None:
            labels = labels.reshape((-1, 1))
            mask = paddle.equal(labels, labels.T).astype('float32')
            max_probs = max_probs.reshape((-1, 1))
            score_mask = paddle.matmul(max_probs, max_probs.T)
            mask = paddle.multiply(mask, score_mask)
            
        contrast_count = feat.shape[1]
        contrast_feat = paddle.concat(paddle.unbind(feat, axis=1), axis=0)  # (2n, d)
        if self.contrast_mode == 'all':
            anchor_feat = contrast_feat
            anchor_count = contrast_count
        anchor_dot_contrast = paddle.matmul(anchor_feat, contrast_feat.T) / self.temperature
        logits_max = anchor_dot_contrast.max(axis=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = paddle.concat([mask, mask], axis=0)
        mask = paddle.concat([mask, mask], axis=1)
        
        logits_mask = 1 - paddle.eye(batch_size * contrast_count, dtype=paddle.float64)
        mask = mask * logits_mask
        exp_logits = paddle.exp(logits) * logits_mask
        log_prob = logits - paddle.log(exp_logits.sum(axis=1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / mask.sum(axis=1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.reshape((anchor_count, batch_size))
        if reduction == 'mean':
            loss = loss.mean()

        return {"SoftSupConLoss": loss}