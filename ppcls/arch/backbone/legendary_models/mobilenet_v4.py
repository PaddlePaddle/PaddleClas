import sys
import os
import re
import math
from copy import deepcopy
import dataclasses
import types
import functools
from functools import partial
from typing import Callable, List, Optional, Tuple, Union, Type, Sequence, Dict, Any

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import to_2tuple, DropPath, Identity

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

_has_silu = True
_has_mish = True
_has_hardswish = True
_has_hardsigmoid = True


class AvgPool2dSame(nn.AvgPool2D):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=
        False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0),
            ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return nn.functional.avg_pool2d(kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding, ceil_mode=self.
            ceil_mode, x=x, exclusive=not self.count_include_pad)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = get_padding_value(padding, kernel_size, stride=
        stride, **kwargs)
    if is_dynamic:
        if pool_type == 'avg':
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'
    elif pool_type == 'avg':
        return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **
            kwargs)
    elif pool_type == 'max':
        return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **
            kwargs)
    else:
        assert False, f'Unsupported pool type {pool_type}'


def use_fused_attn(experimental=False):
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0


class MultiQueryAttention2d(nn.Layer):
    def __init__(self, dim, dim_out=None, num_heads:
        int=8, key_dim=None, value_dim=None,
        query_strides=1, kv_stride=1, dw_kernel_size=3,
        dilation=1, padding='', attn_drop:
        float=0.0, proj_drop=0.0, norm_layer=paddle
        .nn.BatchNorm2D, use_bias=False):
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim or dim // num_heads
        self.value_dim = value_dim or dim // num_heads
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_strides = False
        self.scale = self.key_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.drop = attn_drop
        self.query = nn.Sequential()
        if self.has_query_strides:
            if padding == 'same':
                self.query.add_sublayer(name='down_pool', sublayer=
                    create_pool2d('avg', kernel_size=self.query_strides,
                    padding='same'))
            else:
                self.query.add_sublayer(name='down_pool', sublayer=paddle.
                    nn.AvgPool2D(kernel_size=query_strides, exclusive=False))
            self.query.add_sublayer(name='norm', sublayer=norm_layer(dim))
        self.query.add_sublayer(name='proj', sublayer=create_conv2d(dim, 
            self.num_heads * self.key_dim, kernel_size=1, bias=use_bias))
        self.key = nn.Sequential()
        if kv_stride > 1:
            self.key.add_sublayer(name='down_conv', sublayer=create_conv2d(
                dim, dim, kernel_size=dw_kernel_size, stride=kv_stride,
                dilation=dilation, padding=padding, depthwise=True))
            self.key.add_sublayer(name='norm', sublayer=norm_layer(dim))
        self.key.add_sublayer(name='proj', sublayer=create_conv2d(dim, self
            .key_dim, kernel_size=1, padding=padding, bias=use_bias))
        self.value = nn.Sequential()
        if kv_stride > 1:
            self.value.add_sublayer(name='down_conv', sublayer=
                create_conv2d(dim, dim, kernel_size=dw_kernel_size, stride=
                kv_stride, dilation=dilation, padding=padding, depthwise=True))
            self.value.add_sublayer(name='norm', sublayer=norm_layer(dim))
        self.value.add_sublayer(name='proj', sublayer=create_conv2d(dim,
            self.value_dim, kernel_size=1, bias=use_bias))
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.output = nn.Sequential()
        if self.has_query_strides:
            self.output.add_sublayer(name='upsample', sublayer=nn.
                Upsample(scale_factor=self.query_strides, mode='bilinear',
                align_corners=False))
        self.output.add_sublayer(name='proj', sublayer=create_conv2d(self.
            value_dim * self.num_heads, dim_out, kernel_size=1, bias=use_bias))
        self.output.add_sublayer(name='drop', sublayer=nn.Dropout(p=
            proj_drop))
        self.einsum = False

    def init_weights(self):
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.query.proj.weight)
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.key.proj.weight)
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.value.proj.weight)
        if self.kv_stride > 1:
            init_XavierUniform = nn.initializer.XavierUniform()
            init_XavierUniform(self.key.down_conv.weight)
            init_XavierUniform = nn.initializer.XavierUniform()
            init_XavierUniform(self.value.down_conv.weight)
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.output.proj.weight)

    def _reshape_input(self, t):
        s = tuple(t.shape)
        x = t.reshape([s[0], s[1], -1])
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        t = x.transpose(perm=perm_0)
        if self.einsum:
            return t
        else:
            return t.unsqueeze(axis=1).contiguous()

    def _reshape_projected_query(self, t, num_heads,
        key_dim):
        s = tuple(t.shape)
        t = t.reshape([s[0], num_heads, key_dim, -1])
        if self.einsum:
            return t.transpose(perm=[0, 3, 1, 2]).contiguous()
        else:
            x = t
            perm_1 = list(range(x.ndim))
            perm_1[-1] = -2
            perm_1[-2] = -1
            return x.transpose(perm=perm_1).contiguous()

    def _reshape_output(self, t, num_heads, h_px,
        w_px):
        s = tuple(t.shape)
        feat_dim = s[-1] * num_heads
        if not self.einsum:
            x = t
            perm_2 = list(range(x.ndim))
            perm_2[1] = 2
            perm_2[2] = 1
            t = x.transpose(perm=perm_2)
        return t.reshape([s[0], h_px, w_px, feat_dim]).transpose(perm=[0, 3, 
            1, 2]).contiguous()

    def forward(self, x, attn_mask=None):
        """Run layer computation."""
        B, C, H, W = s = tuple(x.shape)
        q = self.query(x)
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)
        k = self.key(x)
        k = self._reshape_input(k)
        v = self.value(x)
        v = self._reshape_input(v)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        if self.einsum:
            attn = paddle.einsum('blhk,bpk->blhp', q, k) * self.scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            o = paddle.einsum('blhp,bpk->blhk', attn, v)
        elif self.fused_attn:
            o = nn.functional.scaled_dot_product_attention(query=q,
                key=k, value=v, attn_mask=attn_mask, dropout_p=self.
                attn_drop.p if self.training else 0.0)
            o = paddle.transpose(o, perm=[0, 2, 1, 3])           
        else:
            q = q * self.scale
            x = k
            perm_3 = list(range(x.ndim))
            perm_3[-1] = -2
            perm_3[-2] = -1
            attn = q @ x.transpose(perm=perm_3)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            o = attn @ v
        o = self._reshape_output(o, self.num_heads, H // self.query_strides
            [0][0], W // self.query_strides[1][1])
        x = self.output(o)
        return x


class Attention2d(nn.Layer):
    def __init__(self, dim, dim_out=None, num_heads:
        int=32, bias=True, expand_first=False, head_first
        =False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = num_heads
        self.dim_head = dim_attn // num_heads
        self.head_first = head_first
        self.scale = num_heads ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Conv2D(in_channels=dim, out_channels=dim_attn *
            3, kernel_size=1, bias_attr=bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv2D(in_channels=dim_attn, out_channels=
            dim_out, kernel_size=1, bias_attr=bias)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x, attn_mask=None):
        B, C, H, W = tuple(x.shape)
        if self.head_first:
            q, k, v = self.qkv(x).view(B, self.num_heads, self.dim_head * 3, -1
                ).chunk(chunks=3, axis=2)
        else:
            q, k, v = self.qkv(x).reshape(B, 3, self.num_heads, self.
                dim_head, -1).unbind(axis=1)
        if self.fused_attn:
            x = q
            perm_4 = list(range(x.ndim))
            perm_4[-1] = -2
            perm_4[-2] = -1
            x = k
            perm_5 = list(range(x.ndim))
            perm_5[-1] = -2
            perm_5[-2] = -1
            x = v
            perm_6 = list(range(x.ndim))
            perm_6[-1] = -2
            perm_6[-2] = -1
            x = nn.functional.scaled_dot_product_attention(query=x.
                transpose(perm=perm_4).contiguous(), key=x.transpose(perm=
                perm_5).contiguous(), value=x.transpose(perm=perm_6).
                contiguous(), attn_mask=attn_mask, dropout_p=self.attn_drop
                .p if self.training else 0.0)
            perm_7 = list(range(x.ndim))
            perm_7[-1] = -2
            perm_7[-2] = -1
            x = x.transpose(perm=perm_7).reshape(B, -1, H, W)
        else:
            q = q * self.scale
            x = q
            perm_8 = list(range(x.ndim))
            perm_8[-2] = -1
            perm_8[-1] = -2
            attn = x.transpose(perm=perm_8) @ k
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn
            perm_9 = list(range(x.ndim))
            perm_9[-2] = -1
            perm_9[-1] = -2
            x = (v @ x.transpose(perm=perm_9)).view(B, -1, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MobileAttention(nn.Layer):
    def __init__(self, in_chs, out_chs, stride=1,
        dw_kernel_size=3, dilation=1, group_size=1, pad_type:
        str='', num_heads=8, key_dim=64, value_dim=64,
        use_multi_query=False, query_strides=(1, 1), kv_stride:
        int=1, cpe_dw_kernel_size=3, noskip=False, act_layer=nn.ReLU, norm_layer=nn.
        BatchNorm2D, aa_layer=None, drop_path_rate:
        float=0.0, attn_drop=0.0, proj_drop=0.0,
        layer_scale_init_value=1e-05, use_bias=False,
        use_cpe=False):
        super(MobileAttention, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_stride = False   
        if use_cpe:
            self.conv_cpe_dw = create_conv2d(in_chs, in_chs, kernel_size=
                cpe_dw_kernel_size, dilation=dilation, depthwise=True, bias
                =True)
        else:
            self.conv_cpe_dw = None
        self.norm = norm_act_layer(in_chs, apply_act=False)
        if num_heads is None:
            assert in_chs % key_dim == 0
            num_heads = in_chs // key_dim
        if use_multi_query:
            self.attn = MultiQueryAttention2d(in_chs, dim_out=out_chs,
                num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
                query_strides=query_strides, kv_stride=kv_stride, dilation=
                dilation, padding=pad_type, dw_kernel_size=dw_kernel_size,
                attn_drop=attn_drop, proj_drop=proj_drop)
        else:
            self.attn = Attention2d(in_chs, dim_out=out_chs, num_heads=
                num_heads, attn_drop=attn_drop, proj_drop=proj_drop, bias=
                use_bias)
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate
            ) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=
                self.conv_pw.in_channels)
        else:
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        if self.conv_cpe_dw is not None:
            x_cpe = self.conv_cpe_dw(x)
            x = x + x_cpe
        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


def _create_act(act_layer, act_kwargs=None, inplace=False, apply_act=True):
    act_kwargs = act_kwargs or {}
    act_kwargs.setdefault('inplace', inplace)
    act = None
    if apply_act:
        act = create_act_layer(act_layer, **act_kwargs)
    return nn.Identity() if act is None else act


class InplaceAbn(nn.Layer):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        apply_act=True, act_layer='leaky_relu', act_param=0.01, drop_layer=None
        ):
        super(InplaceAbn, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if apply_act:
            if isinstance(act_layer, str):
                assert act_layer in ('leaky_relu', 'elu', 'identity', '')
                self.act_name = act_layer if act_layer else 'identity'
            elif act_layer == nn.ELU:
                self.act_name = 'elu'
            elif act_layer == nn.LeakyReLU:
                self.act_name = 'leaky_relu'
            elif act_layer is None or act_layer == nn.Identity:
                self.act_name = 'identity'
            else:
                assert False, f'Invalid act layer {act_layer.__name__} for IABN'
        else:
            self.act_name = 'identity'
        self.act_param = act_param
        if self.affine:
            self.weight = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.ones(shape=num_features))
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor
                =paddle.zeros(shape=num_features))
        else:
            self.add_parameter(name='weight', parameter=None)
            self.add_parameter(name='bias', parameter=None)
        self.register_buffer(name='_mean', tensor=paddle.zeros(shape
            =num_features))
        self.register_buffer(name='_variance', tensor=paddle.ones(shape=
            num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=0)
        init_Constant(self._mean)
        init_Constant = nn.initializer.Constant(value=1)
        init_Constant(self._variance)
        if self.affine:
            init_Constant = nn.initializer.Constant(value=1)
            init_Constant(self.weight)
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(self.bias)

    def forward(self, x):
        output = inplace_abn(x, self.weight, self.bias, self._mean,
            self._variance, self.training, self.momentum, self.eps, self.
            act_name, self.act_param)
        if isinstance(output, tuple):
            output = output[0]
        return output


class FilterResponseNormAct2d(nn.Layer):
    def __init__(self, num_features, apply_act=True, act_layer=nn.
        ReLU, inplace=None, rms=True, eps=1e-05, **_):
        super(FilterResponseNormAct2d, self).__init__()
        if act_layer is not None and apply_act:
            self.act = create_act_layer(act_layer, inplace=inplace)
        else:
            self.act = nn.Identity()
        self.rms = rms
        self.eps = eps
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.ones(shape=num_features))
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.zeros(shape=num_features))
        self.reset_parameters()

    def reset_parameters(self):
        init_Constant = nn.initializer.Constant(value=1.0)
        init_Constant(self.weight)
        init_Constant = nn.initializer.Constant(value=0.0)
        init_Constant(self.bias)

    def forward(self, x):
        x_dtype = x.dtype
        v_shape = 1, -1, 1, 1
        x = x * inv_instance_rms(x, self.eps)
        x = x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(
            v_shape).to(dtype=x_dtype)
        return self.act(x)


class LayerNormAct(nn.LayerNorm):
    def __init__(self, normalization_shape,
        eps=1e-05, affine=True, apply_act=True, act_layer=nn.ReLU,
        act_kwargs=None, inplace=True, drop_layer=None):
        super(LayerNormAct, self).__init__(normalization_shape, eps=eps,
            elementwise_affine=affine)
        self.drop = drop_layer(
            ) if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=
            inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self
                .bias, self.eps)
        else:
            x = nn.functional.layer_norm(x=x, normalized_shape=self.
                normalized_shape, weight=self.weight, bias=self.bias,
                epsilon=self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class LayerNormAct2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-05, affine=True, apply_act=True,
        act_layer=nn.ReLU, act_kwargs=None, inplace=True, drop_layer
        =None):
        super(LayerNormAct2d, self).__init__(num_channels, eps=eps,
            elementwise_affine=affine)
        self.drop = drop_layer(
            ) if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=
            inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        x = x.transpose(perm=[0, 2, 3, 1])
        if self._fast_norm:
            x = fast_layer_norm(x, self.normalized_shape, self.weight, self
                .bias, self.eps)
        else:
            x = nn.functional.layer_norm(x=x, normalized_shape=self.
                normalized_shape, weight=self.weight, bias=self.bias,
                epsilon=self.eps)
        x = x.transpose(perm=[0, 3, 1, 2])
        x = self.drop(x)
        x = self.act(x)
        return x


class GroupNormAct(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-05, affine=True,
        group_size=None, apply_act=True, act_layer=nn.ReLU,
        act_kwargs=None, inplace=True, drop_layer=None):
        super(GroupNormAct, self).__init__(_num_groups(num_channels,
            num_groups, group_size), num_channels, eps=eps, affine=affine)
        self.drop = drop_layer(
            ) if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=
            inplace, apply_act=apply_act)
        self._fast_norm = is_fast_norm()

    def forward(self, x):
        if self._fast_norm:
            x = fast_group_norm(x, self.num_groups, self.weight, self.bias,
                self.eps)
        else:
            x = nn.functional.group_norm(x=x, num_groups=self.
                num_groups, weight=self.weight, bias=self.bias, epsilon=
                self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


class BatchNormAct2d(nn.BatchNorm2D):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        track_running_stats=True, apply_act=True, act_layer=nn.ReLU,
        act_kwargs=None, inplace=True, drop_layer=None, device=None, dtype=None
        ):
        try:
            factory_kwargs = {'device', 'dtype'}
            super(BatchNormAct2d, self).__init__(num_features,
                momentum=momentum, track_running_stats=track_running_stats, **factory_kwargs)
        except TypeError:
            super(BatchNormAct2d, self).__init__(num_features,
                momentum=momentum)
        self.drop = drop_layer(
            ) if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=
            inplace, apply_act=apply_act)

    def forward(self, x):
        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(y=paddle.to_tensor(1))
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.
                        num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self._mean is None and self._variance is
                None)
        x = nn.functional.batch_norm(x=x, running_mean=self._mean if not self.training or self.track_running_stats else
            None, running_var=self._variance if not self.training or self
            .track_running_stats else None, weight=self.weight, bias=self.
            bias, training=bn_training, epsilon=self._epsilon, momentum=1 -
            exponential_average_factor)
        x = self.drop(x)
        x = self.act(x)
        return x


_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct, LayerNormAct,
    LayerNormAct2d, FilterResponseNormAct2d, InplaceAbn}


def get_norm_act_layer(norm_layer, act_layer=None):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, functools
        .partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.
        FunctionType, functools.partial))
    norm_act_kwargs = {}
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func
    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP[layer_name]
    elif isinstance(norm_layer, types.FunctionType):
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnorm'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            norm_act_layer = GroupNormAct
        elif type_name.startswith('groupnorm1'):
            norm_act_layer = functools.partial(GroupNormAct, num_groups=1)
        elif type_name.startswith('layernorm2d'):
            norm_act_layer = LayerNormAct2d
        elif type_name.startswith('layernorm'):
            norm_act_layer = LayerNormAct
        else:
            assert False, f'No equivalent norm_act layer for {type_name}'
    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer


class HardMishMe(nn.Layer):
    def __init__(self, inplace=False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishAutoFn.apply(x)

_ACT_LAYER_ME = dict(silu=nn.Silu if _has_silu else SwishMe, swish=
    nn.Silu if _has_silu else SwishMe, mish=nn.Mish if
    _has_mish else MishMe, hard_sigmoid=nn.Hardsigmoid if
    _has_hardsigmoid else HardSigmoidMe, hard_swish=nn.Hardswish if
    _has_hardswish else HardSwishMe, hard_mish=HardMishMe)

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


class SelectAdaptivePool2d(nn.Layer):
    def __init__(self, output_size=1, pool_type='fast',
        flatten=False, input_fmt='NCHW'):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''
        pool_type = pool_type.lower()
        if not pool_type:
            self.pool = nn.Identity()
            self.flatten = nn.Flatten(start_axis=1
                ) if flatten else nn.Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            if pool_type.endswith('catavgmax'):
                self.pool = FastAdaptiveCatAvgMaxPool(flatten, input_fmt=
                    input_fmt)
            elif pool_type.endswith('avgmax'):
                self.pool = FastAdaptiveAvgMaxPool(flatten, input_fmt=input_fmt
                    )
            elif pool_type.endswith('max'):
                self.pool = FastAdaptiveMaxPool(flatten, input_fmt=input_fmt)
            elif pool_type == 'fast' or pool_type.endswith('avg'):
                self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            else:
                assert False, 'Invalid pool type' % pool_type
            self.flatten = nn.Identity()
        else:
            assert input_fmt == 'NCHW'
            if pool_type == 'avgmax':
                self.pool = AdaptiveAvgMaxPool2d(output_size)
            elif pool_type == 'catavgmax':
                self.pool = AdaptiveCatAvgMaxPool2d(output_size)
            elif pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2D(output_size=output_size
                    )
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2D(output_size=output_size
                    )
            else:
                assert False, 'Invalid pool type' % pool_type
            self.flatten = nn.Flatten(start_axis=1
                ) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return (self.__class__.__name__ + '(' + 'pool_type=' + self.
            pool_type + ', flatten=' + str(self.flatten) + ')')


class Linear(nn.Linear):
    def forward(self, input):
        return nn.functional.linear(x=input, weight=self.weight, bias=self.bias)


def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False,
    preserve_rng_state=True):
    def run_function(start, end, functions):

        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward
    if isinstance(functions, nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)
    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = paddle.distributed.fleet.utils.recompute(run_function(start,
            end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def get_padding(kernel_size, stride=1, dilation=1, **_):
    if any([isinstance(v, (tuple, list)) for v in [kernel_size, stride,
        dilation]]):
        kernel_size, stride, dilation = to_2tuple(kernel_size), to_2tuple(
            stride), to_2tuple(dilation)
        return [get_padding(*a) for a in zip(kernel_size, stride, dilation)]
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.pop('bias', '')
    kwargs.setdefault('bias_attr', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if _USE_EXPORT_CONV and is_exportable():
            assert not is_scriptable()
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2D(in_channels=in_chs, out_channels=out_chs, kernel_size=kernel_size, padding=padding, **kwargs)


class CondConv2d(nn.Layer):
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super(CondConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        padding_val, is_padding_dynamic = get_padding_value(padding,
            kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = to_2tuple(padding_val)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts
        self.weight_shape = (self.out_channels, self.in_channels // self.groups
            ) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.empty(shape=[self.num_experts, weight_num_param]))
        if bias:
            self.bias_shape = self.out_channels,
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor
                =paddle.empty(shape=[self.num_experts, self.out_channels]))
        else:
            self.add_parameter(name='bias', parameter=None)
        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(partial(nn.
            initializer.KaimingUniform, a=math.sqrt(5)), self.num_experts,
            self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(axis=self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.
                initializer.Uniform, a=-bound, b=bound), self.num_experts,
                self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = tuple(x.shape)
        weight = paddle.matmul(x=routing_weights, y=self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self
            .groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = paddle.matmul(x=routing_weights, y=self.bias)
            bias = bias.view(B * self.out_channels)
        x = x.reshape(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(x, weight, bias, stride=self.stride, padding=
                self.padding, dilation=self.dilation, groups=self.groups * B)
        else:
            out = nn.functional.conv2d(x=x, weight=weight, bias=bias,
                stride=self.stride, padding=self.padding, dilation=self.
                dilation, groups=self.groups * B)
        out = out.transpose(perm=[1, 0, 2, 3]).view(B, self.out_channels,
            tuple(out.shape)[-2], tuple(out.shape)[-1])
        return out


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.LayerDict):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [
            kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits,
            out_splits)):
            conv_groups = in_ch if depthwise else 1
            self.add_sublayer(name=str(idx), sublayer=create_conv2d_pad(
                in_ch, out_ch, k, stride=stride, padding=padding, dilation=
                dilation, groups=conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = paddle.split(x=x, num_or_sections=self.splits, axis=1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = paddle.concat(x=x_out, axis=1)
        return x


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs
        if 'groups' in kwargs:
            groups = kwargs.pop('groups')
            if groups == in_channels:
                kwargs['depthwise'] = True
            else:
                assert groups == 1
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=
                groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size,
                groups=groups, **kwargs)
    return m


class PReLU(nn.PReLU):
    def __init__(self, num_parameters=1, init=0.25, inplace:
        bool=False):
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input):
        return nn.functional.prelu(x=input, weight=self.weight)


class GELUTanh(nn.Layer):
    def __init__(self, inplace=False):
        super(GELUTanh, self).__init__()

    def forward(self, input):
        return nn.functional.gelu(x=input, approximate=True)


class QuickGELU(nn.Layer):
    def __init__(self, inplace=False):
        super(QuickGELU, self).__init__()

    def forward(self, input):
        return quick_gelu(input)


class Sigmoid(nn.Layer):
    def __init__(self, inplace=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


class Tanh(nn.Layer):
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


class HardMish(nn.Layer):
    def __init__(self, inplace=False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


_ACT_LAYER_DEFAULT = dict(silu=nn.Silu if _has_silu else Swish,
    swish=nn.Silu if _has_silu else Swish, mish=nn.Mish if
    _has_mish else Mish, relu=nn.ReLU, relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU, elu=nn.ELU, prelu=PReLU, celu=
    nn.CELU, selu=nn.SELU, gelu=nn.GELU, gelu_tanh=
    GELUTanh, quick_gelu=QuickGELU, sigmoid=Sigmoid, tanh=Tanh,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoid,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwish,
    hard_mish=HardMish, identity=nn.Identity)


def get_act_layer(name='relu'):
    if name is None:
        return None
    if not isinstance(name, str):
        return name
    if not name:
        return None
    if not (is_exportable() or is_scriptable()):
        if name in _ACT_LAYER_ME:
            return _ACT_LAYER_ME[name]
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
    def __init__(self, in_chs, rd_ratio=0.25, rd_channels:
        Optional[int]=None, act_layer=nn.ReLU, 
        gate_layer=nn.Sigmoid, force_act_layer=
        None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2D(in_channels=in_chs,
            out_channels=rd_channels, kernel_size=1, bias_attr=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2D(in_channels=rd_channels,
            out_channels=in_chs, kernel_size=1, bias_attr=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean(axis=(2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class FeatureDictNet(nn.LayerDict):
    def __init__(self, model, out_indices=None, output_fmt:
        str='NCHW', feature_concat=False, flatten_sequential=False
        ):
        super(FeatureDictNet, self).__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.output_fmt = Format(output_fmt)
        self.concat = feature_concat
        self.grad_checkpointing = False
        self.return_layers = {}
        return_layers = _get_return_layers(self.feature_info, out_map)
        modules = _module_list(model, flatten_sequential=flatten_sequential)
        remaining = set(return_layers.keys())
        layers = OrderedDict()
        for new_name, old_name, module in modules:
            layers[new_name] = module
            if old_name in remaining:
                self.return_layers[new_name] = str(return_layers[old_name])
                remaining.remove(old_name)
            if not remaining:
                break
        assert not remaining and len(self.return_layers) == len(return_layers
            ), f'Return layers ({remaining}) are not present in model'
        self.update(layers)

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class FeatureListNet(FeatureDictNet):
    def __init__(self, model, out_indices=(0,
        1, 2, 3, 4), output_fmt='NCHW', feature_concat=False,
        flatten_sequential=False):
        super().__init__(model, out_indices=out_indices, output_fmt=
            output_fmt, feature_concat=feature_concat, flatten_sequential=
            flatten_sequential)

    def forward(self, x):
        return list(self._collect(x).values())


class FeatureHookNet(nn.LayerDict):
    def __init__(self, model, out_indices=(0,
        1, 2, 3, 4), out_map=None,
        return_dict=False, output_fmt='NCHW', no_rewrite:
        Optional[bool]=None, flatten_sequential=False,
        default_hook_type='forward'):
        super().__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        self.return_dict = return_dict
        self.output_fmt = Format(output_fmt)
        self.grad_checkpointing = False
        if no_rewrite is None:
            no_rewrite = not flatten_sequential
        layers = OrderedDict()
        hooks = []
        if no_rewrite:
            assert not flatten_sequential
            if hasattr(model, 'reset_classifier'):
                model.reset_classifier(0)
            layers['body'] = model
            hooks.extend(self.feature_info.get_dicts())
        else:
            modules = _module_list(model, flatten_sequential=flatten_sequential
                )
            remaining = {f['module']: (f['hook_type'] if 'hook_type' in f else
                default_hook_type) for f in self.feature_info.get_dicts()}
            for new_name, old_name, module in modules:
                layers[new_name] = module
                for fn, fm in module.named_sublayers(prefix=old_name):
                    if fn in remaining:
                        hooks.append(dict(module=fn, hook_type=remaining[fn]))
                        del remaining[fn]
                if not remaining:
                    break
            assert not remaining, f'Return layers ({remaining}) are not present in model'
        self.update(layers)
        self.hooks = FeatureHooks(hooks, model.named_sublayers(), out_map=
            out_map)

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class FeatureGetterNet(nn.LayerDict):
    def __init__(self, model, out_indices=4,
        out_map=None, return_dict:
        bool=False, output_fmt='NCHW', norm=False, prune=True
        ):
        super().__init__()
        if prune and hasattr(model, 'prune_intermediate_layers'):
            out_indices = model.prune_intermediate_layers(out_indices,
                prune_norm=not norm)
        self.feature_info = _get_feature_info(model, out_indices)
        self.model = model
        self.out_indices = out_indices
        self.out_map = out_map
        self.return_dict = return_dict
        self.output_fmt = Format(output_fmt)
        self.norm = norm

    def forward(self, x):
        features = self.model.forward_intermediates(x, indices=self.
            out_indices, norm=self.norm, output_fmt=self.output_fmt,
            intermediates_only=True)
        return features


def resolve_pretrained_cfg(variant, pretrained_cfg=None,
    pretrained_cfg_overlay=None):
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)
    if not pretrained_cfg:
        _logger.warning(
            f'No pretrained configuration specified for {model_with_tag} model. Using a default. Please add a config to the model pretrained_cfg registry or pass explicitly.'
            )
        pretrained_cfg = PretrainedCfg()
    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **
        pretrained_cfg_overlay)
    return pretrained_cfg


def _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    default_kwarg_names = 'class_num', 'global_pool', 'in_chans'
    if pretrained_cfg.get('fixed_input_size', False):
        default_kwarg_names += 'img_size',
    for n in default_kwarg_names:
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        elif n == 'class_num':
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None and default_val >= 0:
                kwargs.setdefault(n, pretrained_cfg[n])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])
    _filter_kwargs(kwargs, names=kwargs_filter)


def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def build_model_with_cfg(model_cls, variant, pretrained, pretrained_cfg=None, pretrained_cfg_overlay=None, model_cfg=None, feature_cfg
    =None, pretrained_strict=True, pretrained_filter_fn=None, kwargs_filter=None, **kwargs
    ):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    if kwargs.pop('features_only', False):
        features = True
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
    if pruned:
        model = adapt_model_from_file(model, variant)
    class_num_pretrained = 0 if features else getattr(model,
        'class_num', kwargs.get('class_num', 1000))
    if features:
        use_getter = False
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if feature_cls not in ('dict', 'list', 'hook'):
                    feature_cfg.pop('flatten_sequential', None)
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'list':
                    feature_cls = FeatureListNet
                elif feature_cls == 'dict':
                    feature_cls = FeatureDictNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                elif feature_cls == 'getter':
                    use_getter = True
                    feature_cls = FeatureGetterNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        else:
            feature_cls = FeatureListNet
        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None and not use_getter:
            feature_cfg.setdefault('output_fmt', output_fmt)
        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)
        model.default_cfg = model.pretrained_cfg
    return model


def is_scriptable():
    return False


def is_exportable():
    return False


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


def named_modules(module, name='', depth_first=
    True, include_root=False):
    if not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules(module=child_module, name=child_name,
            depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        yield name, module


def _init_weight_goog(m, n='', fix_group_fanout=True):
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_Normal = nn.initializer.Normal(mean=0, std=math.sqrt(
            2.0 / fan_out))
        init_weight_fn = get_condconv_initializer(lambda w: init_Normal(w),
            m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            init_Constant = nn.initializer.Constant(value=0.0)
            init_Constant(m.bias)
    elif isinstance(m, nn.Conv2D):
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
        init_Uniform = nn.initializer.Uniform(low=-init_range, high=
            init_range)
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
    assert isinstance(block_str, str)
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
            v = op[1:]
            if v == 're':
                value = get_act_layer('relu')
            elif v == 'r6':
                value = get_act_layer('relu6')
            elif v == 'hs':
                value = get_act_layer('hard_swish')
            elif v == 'sw':
                value = get_act_layer('swish')
            elif v == 'mi':
                value = get_act_layer('mish')
            else:
                continue
            options[key] = value
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
    if block_type == 'ir':
        block_args.update(dict(dw_kernel_size=_parse_ksize(options['k']),
            exp_kernel_size=start_kernel_size, pw_kernel_size=
            end_kernel_size, exp_ratio=float(options['e']), se_ratio=float(
            options.get('se', 0.0)), noskip=skip is False, s2d=int(options.
            get('d', 0)) > 0))
        if 'cc' in options:
            block_args['num_experts'] = int(options['cc'])
    elif block_type == 'ds' or block_type == 'dsa':
        block_args.update(dict(dw_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=end_kernel_size, se_ratio=float(options.get('se',
            0.0)), pw_act=block_type == 'dsa', noskip=block_type == 'dsa' or
            skip is False, s2d=int(options.get('d', 0)) > 0))
    elif block_type == 'er':
        block_args.update(dict(exp_kernel_size=_parse_ksize(options['k']),
            pw_kernel_size=end_kernel_size, exp_ratio=float(options['e']),
            force_in_chs=force_in_chs, se_ratio=float(options.get('se', 0.0
            )), noskip=skip is False))
    elif block_type == 'cn':
        block_args.update(dict(kernel_size=int(options['k']), skip=skip is 
            True))
    elif block_type == 'uir':
        start_kernel_size = _parse_ksize(options['a']) if 'a' in options else 0
        end_kernel_size = _parse_ksize(options['p']) if 'p' in options else 0
        block_args.update(dict(dw_kernel_size_start=start_kernel_size,
            dw_kernel_size_mid=_parse_ksize(options['k']),
            dw_kernel_size_end=end_kernel_size, exp_ratio=float(options['e'
            ]), se_ratio=float(options.get('se', 0.0)), noskip=skip is False))
    elif block_type == 'mha':
        kv_dim = int(options['d'])
        block_args.update(dict(dw_kernel_size=_parse_ksize(options['k']),
            num_heads=int(options['h']), key_dim=kv_dim, value_dim=kv_dim,
            kv_stride=int(options.get('v', 1)), noskip=skip is False))
    elif block_type == 'mqa':
        kv_dim = int(options['d'])
        block_args.update(dict(dw_kernel_size=_parse_ksize(options['k']),
            num_heads=int(options['h']), key_dim=kv_dim, value_dim=kv_dim,
            kv_stride=int(options.get('v', 1)), noskip=skip is False))
    else:
        assert False, 'Unknown block type (%s)' % block_type
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


def decode_arch_def(arch_def, depth_multiplier=1.0, depth_trunc='ceil',
    experts_multiplier=1, fix_first_last=False, group_size=None):
    arch_args = []
    if isinstance(depth_multiplier, tuple):
        assert len(depth_multiplier) == len(arch_def)
    else:
        depth_multiplier = (depth_multiplier,) * len(arch_def)
    for stack_idx, (block_strings, multiplier) in enumerate(zip(arch_def,
        depth_multiplier)):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            ba, rep = _decode_block_str(block_str)
            if ba.get('num_experts', 0) > 0 and experts_multiplier > 1:
                ba['num_experts'] *= experts_multiplier
            if group_size is not None:
                ba.setdefault('group_size', group_size)
            stack_args.append(ba)
            repeats.append(rep)
        if fix_first_last and (stack_idx == 0 or stack_idx == len(arch_def) - 1
            ):
            arch_args.append(_scale_stage_depth(stack_args, repeats, 1.0,
                depth_trunc))
        else:
            arch_args.append(_scale_stage_depth(stack_args, repeats,
                multiplier, depth_trunc))
    return arch_args


def _log_info_if(msg, condition):
    if condition:
        _logger.info(msg)


def get_attn(attn_type):
    if isinstance(attn_type, nn.Layer):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            attn_type = attn_type.lower()
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ecam':
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'ge':
                module_cls = GatherExcite
            elif attn_type == 'gc':
                module_cls = GlobalContext
            elif attn_type == 'gca':
                module_cls = partial(GlobalContext, fuse_add=True,
                    fuse_scale=False)
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule
            elif attn_type == 'sk':
                module_cls = SelectiveKernel
            elif attn_type == 'splat':
                module_cls = SplitAttn
            elif attn_type == 'lambda':
                return LambdaLayer
            elif attn_type == 'bottleneck':
                return BottleneckAttn
            elif attn_type == 'halo':
                return HaloAttn
            elif attn_type == 'nl':
                module_cls = NonLocalAttn
            elif attn_type == 'bat':
                module_cls = BatNonLocalAttn
            else:
                assert False, 'Invalid attn module (%s)' % attn_type
        elif isinstance(attn_type, bool):
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def create_aa(aa_layer, channels=None, stride:
    int=2, enable=True, noop=paddle.
    nn.Identity):
    """ Anti-aliasing """
    if not aa_layer or not enable:
        return noop() if noop is not None else None
    if isinstance(aa_layer, str):
        aa_layer = aa_layer.lower().replace('_', '').replace('-', '')
        if aa_layer == 'avg' or aa_layer == 'avgpool':
            aa_layer = nn.AvgPool2D
        elif aa_layer == 'blur' or aa_layer == 'blurpool':
            aa_layer = BlurPool2d
        elif aa_layer == 'blurpc':
            aa_layer = partial(BlurPool2d, pad_mode='constant')
        else:
            assert False, f'Unknown anti-aliasing layer ({aa_layer}).'
    try:
        return aa_layer(channels=channels, stride=stride)
    except TypeError as e:
        return aa_layer(stride)


class EdgeResidual(nn.Layer):
    def __init__(self, in_chs, out_chs, exp_kernel_size=3,
        stride=1, dilation=1, group_size=0, pad_type='',
        force_in_chs=0, noskip=False, exp_ratio=1.0,
        pw_kernel_size=1, act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2D, aa_layer=None, se_layer=None,
        drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size,
            stride=1 if use_aa else stride, dilation=dilation, groups=
            groups, padding=pad_type)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride,
            enable=use_aa)
        self.se = se_layer(mid_chs, act_layer=act_layer
            ) if se_layer else nn.Identity()
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size,
            padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate
            ) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs
                =self.conv_pwl.in_channels)
        else:
            return dict(module='', num_chs=self.conv_pwl._out_channels)

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
    def __init__(self, in_channels, out_channels, kernel_size:
        int=1, stride=1, padding='', dilation=1, groups:
        int=1, bias=False, apply_norm=True, apply_act=
        True, norm_layer=nn.BatchNorm2D, act_layer
        =nn.ReLU, aa_layer=None, drop_layer=None, 
        conv_kwargs=None,norm_kwargs=None, act_kwargs=None):
        super(ConvNormAct, self).__init__()
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        use_aa = aa_layer is not None and stride > 1
        self.conv = create_conv2d(in_channels, out_channels, kernel_size,
            stride=1 if use_aa else stride, padding=padding, dilation=
            dilation, groups=groups, bias=bias, **conv_kwargs)
        if apply_norm:
            norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
            self.bn = norm_act_layer(out_channels, apply_act=apply_act,
                act_kwargs=act_kwargs, **norm_kwargs)
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
                self.bn.add_sublayer(name='drop', sublayer=drop_layer())
        self.aa = create_aa(aa_layer, out_channels, stride=stride, enable=
            use_aa, noop=None)

    def in_channels(self):
        return self.conv.in_channels

    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


class ConvBnAct(nn.Layer):
    def __init__(self, in_chs, out_chs, kernel_size, stride:
        int=1, dilation=1, group_size=0, pad_type='', skip:
        bool=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2D, 
        aa_layer=None, drop_path_rate=0.0):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = aa_layer is not None and stride > 1
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=1 if
            use_aa else stride, dilation=dilation, groups=groups, padding=
            pad_type)
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride,
            enable=use_aa)
        self.drop_path = DropPath(drop_path_rate
            ) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='bn1', hook_type='forward', num_chs=self.
                conv.out_channels)
        else:
            return dict(module='', num_chs=self.conv._out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.aa(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
        
class LayerScale2d(nn.Layer):
    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.create_parameter(shape=[dim], dtype='float32', default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        gamma = self.gamma.reshape([1, -1, 1, 1])
        return x.multiply_(y=paddle.to_tensor(gamma)
            ) if self.inplace else x * gamma


class UniversalInvertedResidual(nn.Layer):
    def __init__(self, in_chs, out_chs, dw_kernel_size_start
        =0, dw_kernel_size_mid=3, dw_kernel_size_end=0, stride:
        int=1, dilation=1, group_size=1, pad_type='', noskip:
        bool=False, exp_ratio=1.0, act_layer=nn.
        ReLU, norm_layer=nn.BatchNorm2D, aa_layer=None, se_layer=None,
        conv_kwargs=None, drop_path_rate=0.0,
        layer_scale_init_value=1e-05):
        super(UniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(in_chs, in_chs,
                dw_kernel_size_start, stride=dw_start_stride, dilation=
                dilation, groups=dw_start_groups, padding=pad_type,
                apply_act=False, act_layer=act_layer, norm_layer=norm_layer,
                aa_layer=aa_layer, **conv_kwargs)
        else:
            self.dw_start = nn.Identity()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(in_chs, mid_chs, 1, padding=pad_type,
            act_layer=act_layer, norm_layer=norm_layer, **conv_kwargs)
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride, dilation=dilation, groups=groups, padding=
                pad_type, act_layer=act_layer, norm_layer=norm_layer,
                aa_layer=aa_layer, **conv_kwargs)
        else:
            self.dw_mid = nn.Identity()
        self.se = se_layer(mid_chs, act_layer=act_layer
            ) if se_layer else nn.Identity()
        self.pw_proj = ConvNormAct(mid_chs, out_chs, 1, padding=pad_type,
            apply_act=False, act_layer=act_layer, norm_layer=norm_layer, **
            conv_kwargs)
        if dw_kernel_size_end:
            dw_end_stride = (stride if not dw_kernel_size_start and not
                dw_kernel_size_mid else 1)
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = ConvNormAct(out_chs, out_chs, dw_kernel_size_end,
                stride=dw_end_stride, dilation=dilation, groups=
                dw_end_groups, padding=pad_type, apply_act=False, act_layer
                =act_layer, norm_layer=norm_layer, **conv_kwargs)
        else:
            self.dw_end = nn.Identity()
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate
            ) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='pw_proj.conv', hook_type='forward_pre',
                num_chs=self.pw_proj.conv.in_channels)
        else:
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
    def __init__(self, output_stride=32, pad_type='',
        round_chs_fn=round_channels, se_from_exp=False,
        act_layer=None, norm_layer=None, aa_layer=None, se_layer
        =None, drop_path_rate=0.0, layer_scale_init_value=None, feature_location=''):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.aa_layer = aa_layer
        self.se_layer = get_attn(se_layer)
        try:
            self.se_layer(8, rd_ratio=1.0)
            self.se_has_ratio = True
        except TypeError:
            self.se_has_ratio = False
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        if feature_location == 'depthwise':
            _logger.warning(
                "feature_location=='depthwise' is deprecated, using 'expansion'"
                )
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = False
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        s2d = ba.get('s2d', 0)
        if s2d > 0:
            ba['out_chs'] *= 4
        if 'force_in_chs' in ba and ba['force_in_chs']:
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'
            ] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if self.aa_layer is not None:
            ba['aa_layer'] = self.aa_layer
        se_ratio = ba.pop('se_ratio', None)
        if se_ratio and self.se_layer is not None:
            if not self.se_from_exp:
                se_ratio /= ba.get('exp_ratio', 1.0)
            if s2d == 1:
                se_ratio /= 4
            if self.se_has_ratio:
                ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
            else:
                ba['se_layer'] = self.se_layer
        if bt == 'ir':
            _log_info_if('  InvertedResidual {}, Args: {}'.format(block_idx,
                str(ba)), self.verbose)
            block = CondConvResidual(**ba) if ba.get('num_experts', 0
                ) else InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            _log_info_if('  DepthwiseSeparable {}, Args: {}'.format(
                block_idx, str(ba)), self.verbose)
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx,
                str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(
                ba)), self.verbose)
            block = ConvBnAct(**ba)
        elif bt == 'uir':
            _log_info_if('  UniversalInvertedResidual {}, Args: {}'.format(
                block_idx, str(ba)), self.verbose)
            block = UniversalInvertedResidual(**ba, layer_scale_init_value=
                self.layer_scale_init_value)
        elif bt == 'mqa':
            _log_info_if('  MobileMultiQueryAttention {}, Args: {}'.format(
                block_idx, str(ba)), self.verbose)
            block = MobileAttention(**ba, use_multi_query=True,
                layer_scale_init_value=self.layer_scale_init_value)
        elif bt == 'mha':
            _log_info_if('  MobileMultiHeadAttention {}, Args: {}'.format(
                block_idx, str(ba)), self.verbose)
            block = MobileAttention(**ba, layer_scale_init_value=self.
                layer_scale_init_value)
        else:
            assert False, 'Unknown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']
        return block

    def __call__(self, in_chs, model_block_args):
        _log_info_if('Building model trunk with %d stages...' % len(
            model_block_args), self.verbose)
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
        space2depth = 0
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            _log_info_if('Stack: {}'.format(stack_idx), self.verbose)
            assert isinstance(stack_args, list)
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                _log_info_if(' Block: {}'.format(block_idx), self.verbose)
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:
                    block_args['stride'] = 1
                if not space2depth and block_args.pop('s2d', False):
                    assert block_args['stride'] == 1
                    space2depth = 1
                if space2depth > 0:
                    if space2depth == 2 and block_args['stride'] == 2:
                        block_args['stride'] = 1
                        block_args['exp_ratio'] /= 4
                        space2depth = 0
                    else:
                        block_args['s2d'] = space2depth
                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args
                        ) or model_block_args[next_stack_idx][0]['stride'] > 1
                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        _log_info_if(
                            '  Converting stride to dilation to maintain output_stride=={}'
                            .format(self.output_stride), self.verbose)
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(block_args, total_block_idx,
                    total_block_count)
                blocks.append(block)
                if space2depth == 1:
                    space2depth = 2
                if extract_features:
                    feature_info = dict(stage=stack_idx + 1, reduction=
                        current_stride, **block.feature_info(self.
                        feature_location))
                    leaf_name = feature_info.get('module', '')
                    if leaf_name:
                        feature_info['module'] = '.'.join([
                            f'blocks.{stack_idx}.{block_idx}', leaf_name])
                    else:
                        assert last_block
                        feature_info['module'] = f'blocks.{stack_idx}'
                    self.features.append(feature_info)
                total_block_idx += 1
            stages.append(nn.Sequential(*blocks))
        return stages


class MobileNetV3(nn.Layer):
    def __init__(self, block_args, class_num=1000,
        in_chans=3, stem_size=16, fix_stem=False,
        num_features=1280, head_bias=True, head_norm=False,
        pad_type='', act_layer=None, norm_layer=None, aa_layer=None,
        se_layer=None, se_from_exp=True,
        round_chs_fn=round_channels, drop_rate=0.0,
        drop_path_rate=0.0, layer_scale_init_value=
        None, global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2D
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)
        builder = EfficientNetBuilder(output_stride=32, pad_type=pad_type,
            round_chs_fn=round_chs_fn, se_from_exp=se_from_exp, act_layer=
            act_layer, norm_layer=norm_layer, aa_layer=aa_layer, se_layer=
            se_layer, drop_path_rate=drop_path_rate, layer_scale_init_value
            =layer_scale_init_value)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        self.stage_ends = [f['stage'] for f in self.feature_info]
        self.num_features = builder.in_chs
        self.head_hidden_size = num_features
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = self.num_features * self.global_pool.feat_mult()
        if head_norm:
            self.conv_head = create_conv2d(num_pooled_chs, self.
                head_hidden_size, 1, padding=pad_type)
            self.norm_head = norm_act_layer(self.head_hidden_size)
            self.act2 = nn.Identity()
        else:
            self.conv_head = create_conv2d(num_pooled_chs, self.
                head_hidden_size, 1, padding=pad_type, bias=head_bias)
            self.norm_head = nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(start_axis=1
            ) if global_pool else nn.Identity()
        self.classifier = Linear(self.head_hidden_size, class_num ) if class_num > 0 else nn.Identity()
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
            x = nn.functional.dropout(x=x, p=self.drop_rate,
                training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_mnv3(variant, pretrained=False, **kwargs):
    features_mode = ''
    model_cls = MobileNetV3
    kwargs_filter = None
    model = build_model_with_cfg(model_cls, variant, pretrained,
        features_only=features_mode == 'cfg', pretrained_strict=
        features_mode != 'cls', kwargs_filter=kwargs_filter, **kwargs)
    return model


def _gen_mobilenet_v4(variant, channel_multiplier=1.0,
    pretrained=False, **kwargs):
    num_features = 1280
    if 'hybrid' in variant:
        layer_scale_init_value = 1e-05
        if 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [['er_r1_k3_s2_e4_c48'], ['uir_r1_a3_k5_s2_e4_c80',
                'uir_r1_a3_k3_s1_e2_c80'], ['uir_r1_a3_k5_s2_e6_c160',
                'uir_r1_a0_k0_s1_e2_c160', 'uir_r1_a3_k3_s1_e4_c160',
                'uir_r1_a3_k5_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160',
                'uir_r1_a3_k3_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160',
                'uir_r1_a3_k0_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160',
                'uir_r1_a3_k3_s1_e4_c160', 'mqa_r1_k3_h4_s1_v2_d64_c160',
                'uir_r1_a3_k0_s1_e4_c160'], ['uir_r1_a5_k5_s2_e6_c256',
                'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a3_k5_s1_e4_c256',
                'uir_r1_a0_k0_s1_e2_c256', 'uir_r1_a3_k5_s1_e2_c256',
                'uir_r1_a0_k0_s1_e2_c256', 'uir_r1_a0_k0_s1_e4_c256',
                'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a3_k0_s1_e4_c256',
                'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a5_k5_s1_e4_c256',
                'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a5_k0_s1_e4_c256',
                'mqa_r1_k3_h4_s1_d64_c256', 'uir_r1_a5_k0_s1_e4_c256'], [
                'cn_r1_k1_s1_c960']]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'gelu')
            arch_def = [['er_r1_k3_s2_e4_c48'], ['uir_r1_a3_k5_s2_e4_c96',
                'uir_r1_a3_k3_s1_e4_c96'], ['uir_r1_a3_k5_s2_e4_c192',
                'uir_r3_a3_k3_s1_e4_c192', 'uir_r1_a3_k5_s1_e4_c192',
                'uir_r2_a5_k3_s1_e4_c192', 'mqa_r1_k3_h8_s1_v2_d48_c192',
                'uir_r1_a5_k3_s1_e4_c192', 'mqa_r1_k3_h8_s1_v2_d48_c192',
                'uir_r1_a5_k3_s1_e4_c192', 'mqa_r1_k3_h8_s1_v2_d48_c192',
                'uir_r1_a5_k3_s1_e4_c192', 'mqa_r1_k3_h8_s1_v2_d48_c192',
                'uir_r1_a3_k0_s1_e4_c192'], ['uir_r4_a5_k5_s2_e4_c512',
                'uir_r1_a5_k0_s1_e4_c512', 'uir_r1_a5_k3_s1_e4_c512',
                'uir_r2_a5_k0_s1_e4_c512', 'uir_r1_a5_k3_s1_e4_c512',
                'uir_r1_a5_k5_s1_e4_c512', 'mqa_r1_k3_h8_s1_d64_c512',
                'uir_r1_a5_k0_s1_e4_c512', 'mqa_r1_k3_h8_s1_d64_c512',
                'uir_r1_a5_k0_s1_e4_c512', 'mqa_r1_k3_h8_s1_d64_c512',
                'uir_r1_a5_k0_s1_e4_c512', 'mqa_r1_k3_h8_s1_d64_c512',
                'uir_r1_a5_k0_s1_e4_c512'], ['cn_r1_k1_s1_c960']]
        else:
            assert False, f'Unknown variant {variant}.'
    else:
        layer_scale_init_value = None
        if 'small' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [['cn_r1_k3_s2_e1_c32', 'cn_r1_k1_s1_e1_c32'], [
                'cn_r1_k3_s2_e1_c96', 'cn_r1_k1_s1_e1_c64'], [
                'uir_r1_a5_k5_s2_e3_c96', 'uir_r4_a0_k3_s1_e2_c96',
                'uir_r1_a3_k0_s1_e4_c96'], ['uir_r1_a3_k3_s2_e6_c128',
                'uir_r1_a5_k5_s1_e4_c128', 'uir_r1_a0_k5_s1_e4_c128',
                'uir_r1_a0_k5_s1_e3_c128', 'uir_r2_a0_k3_s1_e4_c128'], [
                'cn_r1_k1_s1_c960']]
        elif 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [['er_r1_k3_s2_e4_c48'], ['uir_r1_a3_k5_s2_e4_c80',
                'uir_r1_a3_k3_s1_e2_c80'], ['uir_r1_a3_k5_s2_e6_c160',
                'uir_r2_a3_k3_s1_e4_c160', 'uir_r1_a3_k5_s1_e4_c160',
                'uir_r1_a3_k3_s1_e4_c160', 'uir_r1_a3_k0_s1_e4_c160',
                'uir_r1_a0_k0_s1_e2_c160', 'uir_r1_a3_k0_s1_e4_c160'], [
                'uir_r1_a5_k5_s2_e6_c256', 'uir_r1_a5_k5_s1_e4_c256',
                'uir_r2_a3_k5_s1_e4_c256', 'uir_r1_a0_k0_s1_e4_c256',
                'uir_r1_a3_k0_s1_e4_c256', 'uir_r1_a3_k5_s1_e2_c256',
                'uir_r1_a5_k5_s1_e4_c256', 'uir_r2_a0_k0_s1_e4_c256',
                'uir_r1_a5_k0_s1_e2_c256'], ['cn_r1_k1_s1_c960']]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [['er_r1_k3_s2_e4_c48'], ['uir_r1_a3_k5_s2_e4_c96',
                'uir_r1_a3_k3_s1_e4_c96'], ['uir_r1_a3_k5_s2_e4_c192',
                'uir_r3_a3_k3_s1_e4_c192', 'uir_r1_a3_k5_s1_e4_c192',
                'uir_r5_a5_k3_s1_e4_c192', 'uir_r1_a3_k0_s1_e4_c192'], [
                'uir_r4_a5_k5_s2_e4_c512', 'uir_r1_a5_k0_s1_e4_c512',
                'uir_r1_a5_k3_s1_e4_c512', 'uir_r2_a5_k0_s1_e4_c512',
                'uir_r1_a5_k3_s1_e4_c512', 'uir_r1_a5_k5_s1_e4_c512',
                'uir_r3_a5_k0_s1_e4_c512'], ['cn_r1_k1_s1_c960']]
        else:
            assert False, f'Unknown variant {variant}.'
    model_kwargs = dict(block_args=decode_arch_def(arch_def), head_bias=
        False, head_norm=True, num_features=num_features, stem_size=
        stem_size, fix_stem=channel_multiplier < 1.0, round_chs_fn=partial(
        round_channels, multiplier=channel_multiplier), norm_layer=partial(
        nn.BatchNorm2D, **resolve_bn_args(kwargs)), act_layer=
        act_layer, layer_scale_init_value=layer_scale_init_value, **kwargs)
    model = _create_mnv3(variant, pretrained, **model_kwargs)
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
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def MobileNetV4_conv_small(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4('MobileNetV4_conv_small', 1.0, pretrained=
        pretrained, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_small"], use_ssld)
    return model


def MobileNetV4_conv_medium(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4('MobileNetV4_conv_medium', 1.0, pretrained=
        pretrained, **kwargs)        
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_medium"], use_ssld)
    return model


def MobileNetV4_conv_large(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4('MobileNetV4_conv_large', 1.0, pretrained=
        pretrained, **kwargs)        
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_large"], use_ssld)
    return model


def MobileNetV4_hybrid_medium(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4('MobileNetV4_hybrid_medium', 1.0, pretrained=
        pretrained, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_hybrid_medium"], use_ssld)
    return model


def MobileNetV4_hybrid_large(pretrained=False, use_ssld=False, **kwargs):
    model = _gen_mobilenet_v4('MobileNetV4_hybrid_large', 1.0, pretrained=
        pretrained, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_hybrid_large"], use_ssld)
    return model