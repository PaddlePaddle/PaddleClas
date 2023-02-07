from collections import defaultdict
import copy
import paddle
from paddle import nn
from paddle.nn import functional as F
from ..legendary_models.resnet import ResNet50, MODEL_URLS, _load_pretrained

__all__ = [
    "ResNet50_last_stage_stride1", "ResNet50_adaptive_max_pool2d",
    'ResNet50_metabin'
]


def ResNet50_last_stage_stride1(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(conv, pattern):
        new_conv = nn.Conv2D(
            in_channels=conv._in_channels,
            out_channels=conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=1,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=conv._bias_attr)
        return new_conv

    pattern = ["blocks[13].conv1.conv", "blocks[13].short.conv"]
    model = ResNet50(pretrained=False, use_ssld=use_ssld, **kwargs)
    model.upgrade_sublayer(pattern, replace_function)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model


def ResNet50_adaptive_max_pool2d(pretrained=False, use_ssld=False, **kwargs):
    def replace_function(pool, pattern):
        new_pool = nn.AdaptiveMaxPool2D(output_size=1)
        return new_pool

    pattern = ["avg_pool"]
    model = ResNet50(pretrained=False, use_ssld=use_ssld, **kwargs)
    model.upgrade_sublayer(pattern, replace_function)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model


class BINGate(nn.Layer):
    def __init__(self, num_features):
        super().__init__()
        self.gate = self.create_parameter(
            shape=[num_features],
            default_initializer=nn.initializer.Constant(1.0))
        self.add_parameter("gate", self.gate)

    def forward(self, opt={}):
        flag_update = 'lr_gate' in opt and \
            opt.get('enable_inside_update', False)
        if flag_update and self.gate.grad is not None:  # update gate
            lr = opt['lr_gate'] * self.gate.optimize_attr.get('learning_rate',
                                                              1.0)
            gate = self.gate - lr * self.gate.grad
            gate.clip_(min=0, max=1)
        else:
            gate = self.gate
        return gate

    def clip_gate(self):
        self.gate.set_value(self.gate.clip(0, 1))


class MetaBN(nn.BatchNorm2D):
    def forward(self, inputs, opt={}):
        mode = opt.get("bn_mode", "general") if self.training else "eval"
        if mode == "general":  # update, but not apply running_mean/var
            result = F.batch_norm(inputs, self._mean, self._variance,
                                  self.weight, self.bias, self.training,
                                  self._momentum, self._epsilon)
        elif mode == "hold":  # not update, not apply running_mean/var
            result = F.batch_norm(
                inputs,
                paddle.mean(
                    inputs, axis=(0, 2, 3)),
                paddle.var(inputs, axis=(0, 2, 3)),
                self.weight,
                self.bias,
                self.training,
                self._momentum,
                self._epsilon)
        elif mode == "eval":  # fix and apply running_mean/var,
            if self._mean is None:
                result = F.batch_norm(
                    inputs,
                    paddle.mean(
                        inputs, axis=(0, 2, 3)),
                    paddle.var(inputs, axis=(0, 2, 3)),
                    self.weight,
                    self.bias,
                    True,
                    self._momentum,
                    self._epsilon)
            else:
                result = F.batch_norm(inputs, self._mean, self._variance,
                                      self.weight, self.bias, False,
                                      self._momentum, self._epsilon)
        return result


class MetaBIN(nn.Layer):
    """
    MetaBIN (Meta Batch-Instance Normalization)
    reference: https://arxiv.org/abs/2011.14670
    """

    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = MetaBN(
            num_features=num_features, use_global_stats=True)
        self.instance_norm = nn.InstanceNorm2D(num_features=num_features)
        self.gate = BINGate(num_features=num_features)
        self.opt = defaultdict()

    def forward(self, inputs):
        out_bn = self.batch_norm(inputs, self.opt)
        out_in = self.instance_norm(inputs)
        gate = self.gate(self.opt)
        gate = gate.unsqueeze([0, -1, -1])
        out = out_bn * gate + out_in * (1 - gate)
        return out

    def reset_opt(self):
        self.opt = defaultdict()

    def setup_opt(self, opt):
        """
        enable_inside_update: enable inside updating for `gate` in MetaBIN
        lr_gate: learning rate of `gate` during meta-train phase
        bn_mode: control the running stats & updating of BN
        """
        self.check_opt(opt)
        self.opt = copy.deepcopy(opt)

    @classmethod
    def check_opt(cls, opt):
        assert isinstance(opt, dict), \
            TypeError('Got the wrong type of `opt`. Please use `dict` type.')

        if opt.get('enable_inside_update', False) and 'lr_gate' not in opt:
            raise RuntimeError('Missing `lr_gate` in opt.')

        assert isinstance(opt.get('lr_gate', 1.0), float), \
            TypeError('Got the wrong type of `lr_gate`. Please use `float` type.')
        assert isinstance(opt.get('enable_inside_update', True), bool), \
            TypeError('Got the wrong type of `enable_inside_update`. Please use `bool` type.')
        assert opt.get('bn_mode', "general") in ["general", "hold", "eval"], \
            TypeError('Got the wrong value of `bn_mode`.')


def ResNet50_metabin(pretrained=False,
                     use_ssld=False,
                     bias_lr_factor=1.0,
                     gate_lr_factor=1.0,
                     **kwargs):
    """
    ResNet50 which replaces all `bn` layer with MetaBIN
    reference: https://arxiv.org/abs/2011.14670
    """

    def bn2metabin(bn, pattern):
        metabin = MetaBIN(bn.weight.shape[0])
        return metabin

    def setup_optimize_attr(model, bias_lr_factor, gate_lr_factor):
        for name, params in model.named_parameters():
            if params.stop_gradient:
                continue
            if "bias" in name:
                params.optimize_attr['learning_rate'] = bias_lr_factor
            elif "gate" in name:
                params.optimize_attr['learning_rate'] = gate_lr_factor

    stride_list = [2, 2, 2, 2, 1]

    pattern = []
    pattern.extend(["blocks[{}].conv{}.bn".format(i, j) \
                    for i in range(16) for j in range(3)])
    pattern.extend(["blocks[{}].short.bn".format(i) for i in [0, 3, 7, 13]])
    pattern.append("stem[0].bn")

    model = ResNet50(
        pretrained=False, use_ssld=use_ssld, stride_list=stride_list, **kwargs)

    model.upgrade_sublayer(pattern, bn2metabin)
    setup_optimize_attr(
        model=model,
        bias_lr_factor=bias_lr_factor,
        gate_lr_factor=gate_lr_factor)

    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model
