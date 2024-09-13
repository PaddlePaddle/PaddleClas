# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/2404.10518

import math
import re
from copy import deepcopy
from functools import partial

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import DropPath, to_2tuple

MODEL_URLS = {
    "MobileNetV4_conv_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_large_pretrained.pdparams",
    "MobileNetV4_conv_medium":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_medium_pretrained.pdparams",
    "MobileNetV4_conv_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_small_pretrained.pdparams",
    "MobileNetV4_hybrid_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_hybrid_large_pretrained.pdparams",
    "MobileNetV4_hybrid_medium":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_hybrid_medium_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class MultiQueryAttention2D(nn.Layer):
    def __init__(self, dim, dim_out=None, num_heads=8,
                 key_dim=None, value_dim=None, query_strides=1,
                 kv_stride=1, dw_kernel_size=3, dilation=1,
                 padding='', attn_drop=0.0, proj_drop=0.0,
                 norm_layer=nn.BatchNorm2D, use_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim or dim // num_heads
        self.value_dim = value_dim or dim // num_heads
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.scale = self.key_dim ** -0.5
        self.drop = attn_drop
        self.query = nn.Sequential()
        self.query.add_sublayer(
            name='proj',
            sublayer=create_conv2D(dim, self.num_heads * self.key_dim,
                                   kernel_size=1, bias=use_bias)
        )
        self.key = nn.Sequential()
        if kv_stride > 1:
            self.key.add_sublayer(
                name='down_conv',
                sublayer=create_conv2D(
                    dim, dim, kernel_size=dw_kernel_size, stride=kv_stride,
                    dilation=dilation, padding=padding, depthwise=True
                )
            )
            self.key.add_sublayer(name='norm', sublayer=norm_layer(dim))
        self.key.add_sublayer(
            name='proj',
            sublayer=create_conv2D(dim, self.key_dim, kernel_size=1,
                                   padding=padding, bias=use_bias)
        )
        self.value = nn.Sequential()
        if kv_stride > 1:
            self.value.add_sublayer(
                name='down_conv',
                sublayer=create_conv2D(
                    dim, dim, kernel_size=dw_kernel_size, stride=kv_stride,
                    dilation=dilation, padding=padding, depthwise=True
                )
            )
            self.value.add_sublayer(name='norm', sublayer=norm_layer(dim))
        self.value.add_sublayer(name='proj', sublayer=create_conv2D(dim,
                                self.value_dim, kernel_size=1, bias=use_bias))
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.output = nn.Sequential()
        self.output.add_sublayer(
            name='proj', sublayer=create_conv2D(
                self.value_dim * self.num_heads,
                dim_out, kernel_size=1, bias=use_bias))
        self.output.add_sublayer(name='drop', sublayer=nn.Dropout(p=proj_drop))

    def _reshape_input(self, t):
        s = tuple(t.shape)
        x = t.reshape([s[0], s[1], -1])
        perm_0 = [0, 2, 1] + list(range(3, x.ndim))
        t = x.transpose(perm=perm_0)
        return t.unsqueeze(axis=1).contiguous()

    def _reshape_projected_query(self, t, num_heads,
                                 key_dim):
        s = tuple(t.shape)
        t = t.reshape([s[0], num_heads, key_dim, -1])
        x = t
        perm_1 = list(range(x.ndim))
        perm_1[-1] = -2
        perm_1[-2] = -1
        return x.transpose(perm=perm_1).contiguous()

    def _reshape_output(self, t, num_heads, h_px,
                        w_px):
        s = tuple(t.shape)
        feat_dim = s[-1] * num_heads
        x = t
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        t = x.transpose(perm=perm_2)
        return t.reshape([s[0], h_px, w_px, feat_dim]).transpose(
            perm=[0, 3, 1, 2]).contiguous()

    def forward(self, x, attn_mask=None):
        B, C, H, W = tuple(x.shape)
        q = self.query(x)
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)
        k = self.key(x)
        k = self._reshape_input(k)
        v = self.value(x)
        v = self._reshape_input(v)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        o = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        o = paddle.transpose(o, perm=[0, 2, 1, 3])
        o = self._reshape_output(o, self.num_heads, H // self.query_strides
                                 [0][0], W // self.query_strides[1][1])
        x = self.output(o)
        return x


class MobileAttention(nn.Layer):
    def __init__(self, in_chs, out_chs, stride=1,
                 dw_kernel_size=3, dilation=1, group_size=1,
                 pad_type='', num_heads=8, key_dim=64,
                 value_dim=64, use_multi_query=False,
                 query_strides=(1, 1), kv_stride=1,
                 cpe_dw_kernel_size=3, noskip=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D,
                 aa_layer=None, drop_path_rate=0.0,
                 attn_drop=0.0, proj_drop=0.0,
                 layer_scale_init_value=1e-05, use_bias=False,
                 use_cpe=False):
        super(MobileAttention, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.norm = norm_act_layer(in_chs, apply_act=False)
        if use_multi_query:
            self.attn = MultiQueryAttention2D(
                in_chs, dim_out=out_chs, num_heads=num_heads,
                key_dim=key_dim, value_dim=value_dim,
                query_strides=query_strides, kv_stride=kv_stride,
                dilation=dilation, padding=pad_type,
                dw_kernel_size=dw_kernel_size,
                attn_drop=attn_drop, proj_drop=proj_drop
            )
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2D(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
        )

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class BatchNormAct2D(nn.BatchNorm2D):
    def __init__(
            self, num_features, eps=1e-05, momentum=0.1, affine=True,
            track_running_stats=True, apply_act=True, act_layer=nn.ReLU,
            act_kwargs=None, inplace=True, drop_layer=None,
            device=None, dtype=None
    ):
        try:
            factory_kwargs = {'device', 'dtype'}
            super(BatchNormAct2D, self).__init__(
                num_features, momentum=momentum,
                track_running_stats=track_running_stats, **factory_kwargs)
        except TypeError:
            super(BatchNormAct2D, self).__init__(num_features,
                                                 momentum=momentum)
        self.drop = drop_layer(
            ) if drop_layer is not None else nn.Identity()
        self.act = self._create_act(act_layer, act_kwargs=act_kwargs,
                                    inplace=inplace, apply_act=apply_act)

    def _create_act(self, act_layer, act_kwargs=None, inplace=False,
                    apply_act=True):
        act_kwargs = act_kwargs or {}
        act_kwargs.setdefault('inplace', inplace)
        act = None
        if apply_act:
            act = create_act_layer(act_layer, **act_kwargs)
        return nn.Identity() if act is None else act

    def forward(self, x):
        exponential_average_factor = self._momentum
        bn_training = (self._mean is None and self._variance is None)
        x = nn.functional.batch_norm(
            x=x, running_mean=self._mean if not self.training or
            self.track_running_stats else None,
            running_var=self._variance if not self.training or self
            .track_running_stats else None, weight=self.weight, bias=self.
            bias, training=bn_training, epsilon=self._epsilon, momentum=1 -
            exponential_average_factor)
        x = self.drop(x)
        x = self.act(x)
        return x


def get_norm_act_layer(norm_layer, act_layer=None):
    norm_act_kwargs = {}
    norm_act_kwargs.update(norm_layer.keywords)
    norm_layer = norm_layer.func
    norm_act_layer = BatchNormAct2D
    norm_act_kwargs.setdefault('act_layer', act_layer)
    norm_act_layer = partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer


class SelectAdaptivePool2D(nn.Layer):
    def __init__(self, output_size=1, pool_type='fast',
                 flatten=False, input_fmt='NCHW'):
        super(SelectAdaptivePool2D, self).__init__()
        self.pool_type = pool_type or ''
        pool_type = pool_type.lower()
        if not pool_type:
            self.pool = nn.Identity()
            self.flatten = nn.Flatten(
                start_axis=1) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            self.flatten = nn.Identity()
        else:
            if pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2D(output_size=output_size)
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2D(output_size=output_size)
            self.flatten = nn.Flatten(
                start_axis=1) if flatten else nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x


def get_padding(kernel_size, stride=1, dilation=1, **_):
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride,
                                                   dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(
            stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def create_conv2D(in_channels, out_channels, kernel_size, **kwargs):
    depthwise = kwargs.pop('depthwise', False)
    groups = in_channels if depthwise else kwargs.pop('groups', 1)
    padding = kwargs.pop('padding', '')
    kwargs.pop('bias', '')
    kwargs.setdefault('bias_attr', False)
    padding = padding.lower()
    padding = get_padding(kernel_size, **kwargs)
    return nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, padding=padding, groups=groups,
                     **kwargs)


def get_act_layer(name='relu'):
    _ACT_LAYER_DEFAULT = {'relu': nn.ReLU, 'gelu': nn.GELU,
                          'identity': nn.Identity}
    if name is None:
        return None
    if not isinstance(name, str):
        return name
    if not name:
        return None
    return _ACT_LAYER_DEFAULT[name]


def create_act_layer(name, inplace=None,
                     **kwargs):
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None
    if inplace is None:
        return act_layer(**kwargs)
    try:
        return act_layer(inplace=inplace, **kwargs)
    except TypeError:
        return act_layer(**kwargs)


class SqueezeExcite(nn.Layer):
    def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None,
                 act_layer=nn.ReLU, gate_layer=nn.Sigmoid,
                 force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2D(
            in_channels=in_chs, out_channels=rd_channels,
            kernel_size=1, bias_attr=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2D(
            in_channels=rd_channels, out_channels=in_chs,
            kernel_size=1, bias_attr=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean(axis=(2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def build_model_with_cfg(
    model_cls, variant, pretrained, pretrained_cfg=None, model_cfg=None,
    feature_cfg=None, pretrained_strict=True, kwargs_filter=None,
    **kwargs
):
    feature_cfg = feature_cfg or {}
    if kwargs.pop('features_only', False):
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
        if 'feature_cls' in kwargs:
            feature_cfg['feature_cls'] = kwargs.pop('feature_cls')
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    return model


def resolve_act_layer(kwargs, default='relu'):
    return get_act_layer(kwargs.pop('act_layer', default))


def resolve_bn_args(kwargs):
    bn_args = {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None,
                   round_limit=0.9):
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, channel_min,
                          round_limit=round_limit)


def named_modules(module, name='',
                  depth_first=True, include_root=False):
    if not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules(
            module=child_module, name=child_name,
            depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        yield name, module


def _init_weight_goog(m, n='', fix_group_fanout=True):
    if isinstance(m, nn.Conv2D):
        fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
        if fix_group_fanout:
            fan_out //= m._groups
        init_Normal = nn.initializer.Normal(mean=0, std=math.sqrt(
            2.0 / fan_out))
        init_Normal(m.weight)
        if m.bias is not None:
            init_Constant = nn.initializer.Constant(value=0.0)
            init_Constant(m.bias)
    elif isinstance(m, nn.BatchNorm2D):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(m.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.shape[0]
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.shape[1]
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        init_Uniform = nn.initializer.Uniform(
            low=-init_range, high=init_range)
        init_Uniform(m.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(m.bias)


def efficientnet_init_weights(model, init_fn=None):
    init_fn = init_fn or _init_weight_goog
    for n, m in model.named_sublayers():
        init_fn(m, n)
    for n, m in named_modules(model):
        if hasattr(m, 'init_weights'):
            m.init_weights()


def _parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split('.')]


def _decode_block_str(block_str):
    ops = block_str.split('_')
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    skip = None
    for op in ops:
        if op == 'noskip':
            skip = False
        elif op == 'skip':
            skip = True
        elif op.startswith('n'):
            key = op[0]
        else:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
    act_layer = options['n'] if 'n' in options else None
    start_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
    end_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1
    force_in_chs = int(options['fc']) if 'fc' in options else 0
    num_repeat = int(options['r'])
    block_args = dict(block_type=block_type, out_chs=int(options['c']),
                      stride=int(options['s']), act_layer=act_layer)
    if block_type == 'er':
        block_args.update(dict(
            exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=end_kernel_size,
            exp_ratio=float(options['e']),
            force_in_chs=force_in_chs,
            se_ratio=float(options.get('se', 0.0)),
            noskip=skip is False
        ))
    elif block_type == 'cn':
        block_args.update(dict(
            kernel_size=int(options['k']),
            skip=skip is True
        ))
    elif block_type == 'uir':
        start_kernel_size = (
            _parse_ksize(options['a']) if 'a' in options else 0
        )
        end_kernel_size = (
            _parse_ksize(options['p']) if 'p' in options else 0
        )
        block_args.update(dict(
            dw_kernel_size_start=start_kernel_size,
            dw_kernel_size_mid=_parse_ksize(options['k']),
            dw_kernel_size_end=end_kernel_size,
            exp_ratio=float(options['e']),
            se_ratio=float(options.get('se', 0.0)),
            noskip=skip is False
        ))
    elif block_type == 'mqa':
        kv_dim = int(options['d'])
        block_args.update(dict(
            dw_kernel_size=_parse_ksize(options['k']),
            num_heads=int(options['h']),
            key_dim=kv_dim,
            value_dim=kv_dim,
            kv_stride=int(options.get('v', 1)),
            noskip=skip is False
        ))
    if 'gs' in options:
        block_args['group_size'] = int(options['gs'])
    return block_args, num_repeat


def _scale_stage_depth(stack_args, repeats, depth_multiplier=1.0,
                       depth_trunc='ceil'):
    num_repeat = sum(repeats)
    if depth_trunc == 'round':
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round(r / num_repeat * num_repeat_scaled))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def decode_arch_def(
    arch_def, depth_multiplier=1.0, depth_trunc='ceil',
    experts_multiplier=1, fix_first_last=False, group_size=None
):
    arch_args = []
    depth_multiplier = (depth_multiplier,) * len(arch_def)
    for stack_idx, (block_strings, multiplier) in enumerate(
        zip(arch_def, depth_multiplier)
    ):
        stack_args = []
        repeats = []
        for block_str in block_strings:
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            if group_size is not None:
                ba.setdefault('group_size', group_size)
            stack_args.append(ba)
            repeats.append(rep)
        if fix_first_last and (
            stack_idx == 0 or
            stack_idx == len(arch_def) - 1
        ):
            arch_args.append(
                _scale_stage_depth(stack_args, repeats, 1.0, depth_trunc)
            )
        else:
            arch_args.append(_scale_stage_depth(stack_args, repeats,
                             multiplier, depth_trunc))
    return arch_args


def get_attn(attn_type):
    if isinstance(attn_type, nn.Layer):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
        else:
            module_cls = attn_type
    return module_cls


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        return channels // group_size


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def create_adaptive_activation(aa_layer, channels=None, stride=2,
                               enable=True, noop=nn.Identity):
    if not aa_layer or not enable:
        return noop() if noop is not None else None
    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.AvgPool2D
    return aa_layer(channels=channels, stride=stride)


class EdgeResidual(nn.Layer):
    def __init__(self, in_chs, out_chs, exp_kernel_size=3,
                 stride=1, dilation=1, group_size=0,
                 pad_type='', force_in_chs=0, noskip=False,
                 exp_ratio=1.0, pw_kernel_size=1, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D, aa_layer=None,
                 se_layer=None, drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1
        self.conv_exp = create_conv2D(
            in_chs, mid_chs, exp_kernel_size, stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=pad_type
        )
        self.bn1 = norm_act_layer(mid_chs, inplace=True)
        self.aa = create_adaptive_activation(aa_layer, channels=mid_chs,
                                             stride=stride, enable=use_aa)
        self.se = (se_layer(mid_chs, act_layer=act_layer)
                   if se_layer else nn.Identity())
        self.conv_pwl = create_conv2D(
            mid_chs, out_chs, pw_kernel_size, padding=pad_type
        )
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = (DropPath(drop_path_rate)
                          if drop_path_rate else nn.Identity())

    def feature_info(self, location):
        return dict(
            module='', num_chs=self.conv_pwl._out_channels
        )

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class ConvNormAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size: int = 1,
                 stride=1, padding='', dilation=1, groups: int = 1,
                 bias=False, apply_norm=True, apply_act=True,
                 norm_layer=nn.BatchNorm2D, act_layer=nn.ReLU,
                 aa_layer=None, drop_layer=None,
                 conv_kwargs=None, norm_kwargs=None,
                 act_kwargs=None):
        super(ConvNormAct, self).__init__()
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        use_aa = aa_layer is not None and stride > 1
        self.conv = create_conv2D(
            in_channels, out_channels,
            kernel_size, stride=1 if use_aa else stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias, **conv_kwargs
        )
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if drop_layer:
            norm_kwargs['drop_layer'] = drop_layer
        self.bn = norm_act_layer(
            out_channels, apply_act=apply_act,
            act_kwargs=act_kwargs, **norm_kwargs
        )
        self.aa = create_adaptive_activation(
            aa_layer, out_channels, stride=stride,
            enable=use_aa, noop=None
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


class ConvBnAct(nn.Layer):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1,
                 dilation=1, group_size=0, pad_type='', skip=False,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D,
                 aa_layer=None, drop_path_rate=0.0):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = aa_layer is not None and stride > 1
        self.conv = create_conv2D(
            in_chs, out_chs, kernel_size,
            stride=1 if use_aa else stride, dilation=dilation,
            groups=groups, padding=pad_type
        )
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.aa = create_adaptive_activation(
            aa_layer, channels=out_chs,
            stride=stride, enable=use_aa
        )
        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate else nn.Identity()
        )

    def feature_info(self, location):
        return dict(module='', num_chs=self.conv._out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.aa(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class LayerScale2D(nn.Layer):
    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x):
        gamma = self.gamma.reshape([1, -1, 1, 1])
        return (
            x.multiply_(y=paddle.to_tensor(gamma))
            if self.inplace
            else x * gamma
        )


class UniversalInvertedResidual(nn.Layer):
    def __init__(self, in_chs, out_chs, dw_kernel_size_start=0,
                 dw_kernel_size_mid=3, dw_kernel_size_end=0,
                 stride=1, dilation=1, group_size=1,
                 pad_type='', noskip=False,
                 exp_ratio=1.0, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D, aa_layer=None,
                 se_layer=None, conv_kwargs=None,
                 drop_path_rate=0.0, layer_scale_init_value=1e-05):
        super(UniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start,
                stride=dw_start_stride, dilation=dilation,
                groups=dw_start_groups, padding=pad_type,
                apply_act=False, act_layer=act_layer,
                norm_layer=norm_layer, aa_layer=aa_layer,
                **conv_kwargs
            )
        else:
            self.dw_start = nn.Identity()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            in_chs, mid_chs, kernel_size=1,
            padding=pad_type, act_layer=act_layer,
            norm_layer=norm_layer, **conv_kwargs
        )
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride, dilation=dilation,
                groups=groups, padding=pad_type,
                act_layer=act_layer, norm_layer=norm_layer,
                aa_layer=aa_layer, **conv_kwargs
            )
        else:
            self.dw_mid = nn.Identity()
        self.se = (
            se_layer(mid_chs, act_layer=act_layer)
            if se_layer else nn.Identity()
        )
        self.pw_proj = ConvNormAct(
            mid_chs, out_chs, 1, padding=pad_type,
            apply_act=False, act_layer=act_layer,
            norm_layer=norm_layer, **conv_kwargs
        )
        self.dw_end = nn.Identity()
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2D(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate
                                  ) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        return dict(module='', num_chs=self.pw_proj.conv._out_channels)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EfficientNetBuilder:
    """ Build Trunk Blocks
    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py
    """
    def __init__(self, output_stride=32, pad_type='',
                 round_chs_fn=round_channels, se_from_exp=False,
                 act_layer=None, norm_layer=None, aa_layer=None,
                 se_layer=None, drop_path_rate=0.0,
                 layer_scale_init_value=None, feature_location=''
                 ):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.aa_layer = aa_layer
        self.se_layer = get_attn(se_layer)
        self.se_layer(8, rd_ratio=1.0)
        self.se_has_ratio = True
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        if feature_location == 'depthwise':
            feature_location = 'expansion'
        self.feature_location = feature_location
        self.verbose = False
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        s2D = ba.get('s2D', 0)
        if s2D > 0:
            ba['out_chs'] *= 4
        if 'force_in_chs' in ba and ba['force_in_chs']:
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = (
            ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        )
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if self.aa_layer is not None:
            ba['aa_layer'] = self.aa_layer
        se_ratio = ba.pop('se_ratio', None)
        if se_ratio and self.se_layer is not None:
            if not self.se_from_exp:
                se_ratio /= ba.get('exp_ratio', 1.0)
            if s2D == 1:
                se_ratio /= 4
            if self.se_has_ratio:
                ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
            else:
                ba['se_layer'] = self.se_layer
        elif bt == 'er':
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            block = ConvBnAct(**ba)
        elif bt == 'uir':
            block = UniversalInvertedResidual(
                **ba,
                layer_scale_init_value=self.layer_scale_init_value
            )
        elif bt == 'mqa':
            block = MobileAttention(
                **ba,
                use_multi_query=True,
                layer_scale_init_value=self.layer_scale_init_value
            )
        self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            feature_info = dict(module='bn1', num_chs=in_chs, stage=0,
                                reduction=current_stride)
            self.features.append(feature_info)
        space2Depth = 0
        for stack_idx, stack_args in enumerate(model_block_args):
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                if block_idx >= 1:
                    block_args['stride'] = 1
                if not space2Depth and block_args.pop('s2D', False):
                    space2Depth = 1
                if space2Depth > 0:
                    if space2Depth == 2 and block_args['stride'] == 2:
                        block_args['stride'] = 1
                        block_args['exp_ratio'] /= 4
                        space2Depth = 0
                    else:
                        block_args['s2D'] = space2Depth
                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = (
                        next_stack_idx >= len(model_block_args) or
                        model_block_args[next_stack_idx][0]['stride'] > 1
                    )
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(block_args, total_block_idx,
                                         total_block_count)
                blocks.append(block)
                if space2Depth == 1:
                    space2Depth = 2
                if extract_features:
                    feature_info = dict(
                        stage=stack_idx + 1, reduction=current_stride,
                        **block.feature_info(self.feature_location)
                    )
                    leaf_name = feature_info.get('module', '')
                    if leaf_name:
                        feature_info['module'] = '.'.join([
                            f'blocks.{stack_idx}.{block_idx}', leaf_name])
                    else:
                        feature_info['module'] = f'blocks.{stack_idx}'
                    self.features.append(feature_info)
                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages


class MobileNetV4(nn.Layer):
    def __init__(self, block_args, class_num=1000, in_chans=3,
                 stem_size=16, fix_stem=False, num_features=1280,
                 head_bias=True, head_norm=False, pad_type='',
                 act_layer=None, norm_layer=None, aa_layer=None,
                 se_layer=None, se_from_exp=True,
                 round_chs_fn=round_channels, drop_rate=0.0,
                 drop_path_rate=0.0, layer_scale_init_value=None,
                 global_pool='avg'):
        super(MobileNetV4, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2D
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.class_num = class_num
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2D(
            in_chans, stem_size, kernel_size=3,
            stride=2, padding=pad_type
        )
        self.bn1 = norm_act_layer(stem_size, inplace=True)
        builder = EfficientNetBuilder(
            output_stride=32, pad_type=pad_type,
            round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer,
            aa_layer=aa_layer, se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.num_features = builder.in_chs
        self.head_hidden_size = num_features
        self.global_pool = SelectAdaptivePool2D(pool_type=global_pool)
        num_pooled_chs = self.num_features
        self.conv_head = create_conv2D(
            num_pooled_chs, self.head_hidden_size,
            kernel_size=1, padding=pad_type
        )
        self.norm_head = norm_act_layer(self.head_hidden_size)
        self.act2 = nn.Identity()
        self.flatten = (
            nn.Flatten(start_axis=1) if global_pool else nn.Identity()
        )
        self.classifier = (
            nn.Linear(self.head_hidden_size, class_num)
            if class_num > 0 else nn.Identity()
        )
        efficientnet_init_weights(self)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.0:
            x = nn.functional.dropout(
                x=x, p=self.drop_rate, training=self.training
            )
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_mnv4(variant, pretrained=False, **kwargs):
    features_mode = ''
    model_cls = MobileNetV4
    kwargs_filter = None
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        features_only=features_mode == 'cfg',
        pretrained_strict=features_mode != 'cls',
        kwargs_filter=kwargs_filter, **kwargs
    )
    return model


def _gen_mobilenet_v4(variant, channel_multiplier=1.0,
                      pretrained=False, **kwargs):
    """Creates a MobileNet-V4 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    num_features = 1280
    if 'hybrid' in variant:
        layer_scale_init_value = 1e-05
        if 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                ['er_r1_k3_s2_e4_c48'],
                ['uir_r1_a3_k5_s2_e4_c80', 'uir_r1_a3_k3_s1_e2_c80'],
                ['uir_r1_a3_k5_s2_e6_c160', 'uir_r1_a0_k0_s1_e2_c160',
                 'uir_r1_a3_k3_s1_e4_c160', 'uir_r1_a3_k5_s1_e4_c160',
                 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k3_s1_e4_c160',
                 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k0_s1_e4_c160',
                 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k3_s1_e4_c160',
                 'mqa_r1_k3_h4_s1_v2_d64_c160', 'uir_r1_a3_k0_s1_e4_c160'],
                ['uir_r1_a5_k5_s2_e6_c256', 'uir_r1_a5_k5_s1_e4_c256',
                 'uir_r2_a3_k5_s1_e4_c256', 'uir_r1_a0_k0_s1_e2_c256',
                 'uir_r1_a3_k5_s1_e2_c256', 'uir_r1_a0_k0_s1_e2_c256',
                 'uir_r1_a0_k0_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256',
                 'uir_r1_a3_k0_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256',
                 'uir_r1_a5_k5_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256',
                 'uir_r1_a5_k0_s1_e4_c256', 'mqa_r1_k3_h4_s1_d64_c256'],
                ['cn_r1_k1_s1_c960']
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'gelu')
            arch_def = [
                ['er_r1_k3_s2_e4_c48'],
                ['uir_r1_a3_k5_s2_e4_c96', 'uir_r1_a3_k3_s1_e4_c96'],
                ['uir_r1_a3_k5_s2_e4_c192', 'uir_r3_a3_k3_s1_e4_c192',
                 'uir_r1_a3_k5_s1_e4_c192', 'uir_r2_a5_k3_s1_e4_c192',
                 'mqa_r1_k3_h8_s1_v2_d48_c192', 'uir_r1_a5_k3_s1_e4_c192',
                 'mqa_r1_k3_h8_s1_v2_d48_c192', 'uir_r1_a5_k3_s1_e4_c192',
                 'mqa_r1_k3_h8_s1_v2_d48_c192', 'uir_r1_a5_k3_s1_e4_c192',
                 'mqa_r1_k3_h8_s1_v2_d48_c192', 'uir_r1_a3_k0_s1_e4_c192'],
                ['uir_r4_a5_k5_s2_e4_c512', 'uir_r1_a5_k0_s1_e4_c512',
                 'uir_r1_a5_k3_s1_e4_c512', 'uir_r2_a5_k0_s1_e4_c512',
                 'uir_r1_a5_k3_s1_e4_c512', 'uir_r1_a5_k5_s1_e4_c512',
                 'mqa_r1_k3_h8_s1_d64_c512', 'uir_r1_a5_k0_s1_e4_c512',
                 'mqa_r1_k3_h8_s1_d64_c512', 'uir_r1_a5_k0_s1_e4_c512',
                 'mqa_r1_k3_h8_s1_d64_c512', 'uir_r1_a5_k0_s1_e4_c512',
                 'mqa_r1_k3_h8_s1_d64_c512', 'uir_r1_a5_k0_s1_e4_c512'],
                ['cn_r1_k1_s1_c960']
            ]
    else:
        layer_scale_init_value = None
        if 'small' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                ['cn_r1_k3_s2_e1_c32', 'cn_r1_k1_s1_e1_c32'],
                ['cn_r1_k3_s2_e1_c96', 'cn_r1_k1_s1_e1_c64'],
                ['uir_r1_a5_k5_s2_e3_c96', 'uir_r4_a0_k3_s1_e2_c96',
                 'uir_r1_a3_k0_s1_e4_c96'],
                ['uir_r1_a3_k3_s2_e6_c128', 'uir_r1_a5_k5_s1_e4_c128',
                 'uir_r1_a0_k5_s1_e4_c128', 'uir_r1_a0_k5_s1_e3_c128',
                 'uir_r2_a0_k3_s1_e4_c128'],
                ['cn_r1_k1_s1_c960']
            ]
        elif 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                ['er_r1_k3_s2_e4_c48'],
                ['uir_r1_a3_k5_s2_e4_c80', 'uir_r1_a3_k3_s1_e2_c80'],
                ['uir_r1_a3_k5_s2_e6_c160', 'uir_r2_a3_k3_s1_e4_c160',
                 'uir_r1_a3_k5_s1_e4_c160', 'uir_r1_a3_k3_s1_e4_c160',
                 'uir_r1_a3_k0_s1_e4_c160', 'uir_r1_a0_k0_s1_e2_c160',
                 'uir_r1_a3_k0_s1_e4_c160'],
                ['uir_r1_a5_k5_s2_e6_c256', 'uir_r1_a5_k5_s1_e4_c256',
                 'uir_r2_a3_k5_s1_e4_c256', 'uir_r1_a0_k0_s1_e4_c256',
                 'uir_r1_a3_k0_s1_e4_c256', 'uir_r1_a3_k5_s1_e2_c256',
                 'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a0_k0_s1_e4_c256',
                 'uir_r1_a5_k0_s1_e2_c256'],
                ['cn_r1_k1_s1_c960']
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                ['er_r1_k3_s2_e4_c48'],
                ['uir_r1_a3_k5_s2_e4_c96', 'uir_r1_a3_k3_s1_e4_c96'],
                ['uir_r1_a3_k5_s2_e4_c192', 'uir_r3_a3_k3_s1_e4_c192',
                 'uir_r1_a3_k5_s1_e4_c192', 'uir_r5_a5_k3_s1_e4_c192',
                 'uir_r1_a3_k0_s1_e4_c192'],
                ['uir_r4_a5_k5_s2_e4_c512', 'uir_r1_a5_k0_s1_e4_c512',
                 'uir_r1_a5_k3_s1_e4_c512', 'uir_r2_a5_k0_s1_e4_c512',
                 'uir_r1_a5_k3_s1_e4_c512', 'uir_r1_a5_k5_s1_e4_c512',
                 'uir_r3_a5_k0_s1_e4_c512'],
                ['cn_r1_k1_s1_c960']
            ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False, head_norm=True, num_features=num_features,
        stem_size=stem_size, fix_stem=channel_multiplier < 1.0,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2D, **resolve_bn_args(kwargs)),
        act_layer=act_layer, layer_scale_init_value=layer_scale_init_value,
        **kwargs
    )
    model = _create_mnv4(variant, pretrained, **model_kwargs)
    return model


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. "
        )


def MobileNetV4_conv_small(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4(
        'MobileNetV4_conv_small', 1.0, pretrained=pretrained, **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV4_conv_small"],
        use_ssld
    )
    return model


def MobileNetV4_conv_medium(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4(
        'MobileNetV4_conv_medium', 1.0, pretrained=pretrained, **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV4_conv_medium"],
        use_ssld
    )
    return model


def MobileNetV4_conv_large(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4(
        'MobileNetV4_conv_large', 1.0, pretrained=pretrained, **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV4_conv_large"],
        use_ssld
    )
    return model


def MobileNetV4_hybrid_medium(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4(
        'MobileNetV4_hybrid_medium', 1.0, pretrained=pretrained, **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV4_hybrid_medium"],
        use_ssld
    )
    return model


def MobileNetV4_hybrid_large(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4(
        'MobileNetV4_hybrid_large', 1.0, pretrained=pretrained, **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV4_hybrid_large"],
        use_ssld
    )
    return model
