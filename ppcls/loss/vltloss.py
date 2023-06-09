import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from ..arch.backbone.model_zoo.vl_ltr import CVLP_vit16


def cross_entropy(outputs, teacher_outputs):
    logprobs = F.log_softmax(outputs, axis=-1)
    soft_targets = F.softmax(teacher_outputs, axis=-1)
    distill_loss = -paddle.sum(soft_targets * logprobs, axis=-1)
    return paddle.mean(distill_loss)


def labels2idxs(labels: paddle.Tensor):
    #labels = paddle.cast(labels,paddle.int32)
    buff = [
        paddle.cast(
            labels[i] == labels, dtype=paddle.int32)
        for i in range(labels.shape[0])
    ]
    targets = paddle.stack(buff)
    return targets


def kl_div(outputs1, outputs2, T=1.):
    return paddle.log(
        F.kl_div(
            F.log_softmax(
                outputs1 / T, axis=1),
            F.log_softmax(
                outputs2 / T, axis=1),
            reduction='sum', )) * (T * T) / paddle.numel(outputs1)


class SoftTargetCrossEntropy(nn.Layer):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: paddle.Tensor,
                target: paddle.Tensor) -> paddle.Tensor:
        target = paddle.cast(target, dtype=paddle.float32)
        loss = paddle.sum(-target * paddle.nn.functional.log_softmax(
            x, axis=-1),
                          axis=-1)
        return paddle.mean(loss)


class LabelSmoothingCrossEntropy(nn.Layer):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: paddle.Tensor, target: paddle.Tensor):
        logprobs = F.log_softmax(x, axis=-1)
        smooth_loss = -paddle.mean(logprobs, axis=-1)
        #smooth_loss = -logprobs.mean(dim=-1)
        if target.ndim == 1:
            nll_loss = -paddle.gather(
                logprobs, index=paddle.unsqueeze(target), axis=-1)
            nll_loss = paddle.squeeze(nll_loss, axis=1)
        else:
            assert target.ndim == 2
            nll_loss = -paddle.sum(target * logprobs, axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return paddle.mean(loss)


class PretrainSentLoss(nn.Layer):
    def __init__(self,
                 loss_type: str,
                 context_length,
                 pretrained_clip,
                 args=None,
                 distill_type='none',
                 alpha=0.,
                 beta=0.,
                 tau=0.,
                 set_training_mode=False):
        super().__init__()
        self.teacher_model = CVLP_vit16(
            context_length=context_length, pretrained_clip=pretrained_clip)
        self.base_criterion = LabelSmoothingCrossEntropy()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        assert distill_type in ['none', 'feat', 'logits', 'logits_kl']
        self.distill_type: str = distill_type
        if beta > 0:
            assert self.distill_type.startswith("logits")
            self.teacher_model.stop_gradient = True

    def forward(self, x, labels: paddle.Tensor):
        # x \in [inputs, outputs]
        inputs, outputs = x
        labels = paddle.squeeze(labels)
        labels = labels2idxs(labels)
        #check the format for labels
        if isinstance(outputs, paddle.Tensor):
            loss = self.base_criterion(outputs, labels)
            return loss
        # assume that the model outputs a tuple of outputs
        if self.alpha > 0.:
            assert self.distill_type.startswith("feat")
            # assume that the model outputs a tuple of [outputs1, outputs2, distill_loss]
            outputs1, outputs2, distill_loss = outputs
            distill_loss = paddle.mean(distill_loss)
        else:
            # assume that the model outputs a tuple of [outputs1, outputs2]
            outputs1, outputs2 = outputs
            distill_loss = 0.
        if self.loss_type in ["softCE", "smoothCE"]:
            labels = labels / paddle.sum(labels, axis=1, keepdim=True)
        loss1 = self.base_criterion(outputs1, labels)
        loss2 = self.base_criterion(outputs2, labels)
        base_loss = (loss1 + loss2) / 2.0
        loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss
        if self.beta > 0:
            _, (teacher_outputs1,
                teacher_outputs2) = self.teacher_model(inputs)
            teacher_outputs1, teacher_outputs2 = teacher_outputs1.detach(
            ), teacher_outputs2.detach()
            if self.distill_type == 'logits_kl':
                distill_loss1 = kl_div(outputs1, teacher_outputs1, T=self.tau)
                distill_loss2 = kl_div(outputs2, teacher_outputs2, T=self.tau)
                distill_loss = (distill_loss1 + distill_loss2) / 2.0
            else:
                assert self.distill_type == "logits"
                distill_loss1 = cross_entropy(outputs1, teacher_outputs1)
                distill_loss2 = cross_entropy(outputs2, teacher_outputs2)
                distill_loss = (distill_loss1 + distill_loss2) / 2.0
            loss = (1 - self.beta) * loss + self.beta * distill_loss
        return {"pretrain_loss": loss}


class LGRTwoBrachLoss(nn.Layer):
    def __init__(self,
                 teacher_model: nn.Layer=None,
                 distillation_type: str='none',
                 alpha: float=0.0,
                 tau: float=0.0):
        super().__init__()
        self.loss = VLT_DistillationLoss(teacher_model, distillation_type,
                                         alpha, tau)

    def forward(self, x, labels):
        output1, output2 = x[0], x[1]
        loss1 = self.loss(output1, labels)
        loss2 = self.loss(output2, labels)
        loss = loss1 + loss2
        return {"two_branch_loss": loss}


class LGRTwoBrachCELoss(nn.Layer):
    def __init__(self,
                 teacher_model: nn.Layer=None,
                 distillation_type: str='none',
                 alpha: float=0.0,
                 tau: float=0.0):
        super().__init__()
        self.loss = F.cross_entropy

    def forward(self, x, labels):
        output1, output2 = x[0], x[1]
        loss1 = self.loss(output1, labels)
        loss2 = self.loss(output2, labels)
        loss = loss1.mean() + loss2.mean()
        return {"two_branch_loss": loss}


class VLT_DistillationLoss(nn.Layer):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self,
                 teacher_model: nn.Layer=None,
                 distillation_type: str='none',
                 alpha: float=0.0,
                 tau: float=0.0):
        super().__init__()
        self.base_criterion = SoftTargetCrossEntropy()
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, x, labels):
        outputs = x
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, paddle.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        return base_loss
