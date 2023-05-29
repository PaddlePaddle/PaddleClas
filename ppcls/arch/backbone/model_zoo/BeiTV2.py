import math
from functools import partial
import numpy as np
from collections.abc import Callable

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy import interpolate

from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from collections import OrderedDict

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob).astype(x.dtype)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the original BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Attention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter(shape=(all_head_dim,), default_initializer=zeros_)
            self.add_parameter("q_bias", self.q_bias)
            self.v_bias = self.create_parameter(shape=(all_head_dim,), default_initializer=zeros_)
            self.add_parameter("v_bias", self.v_bias)
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(shape=(self.num_relative_distance, num_heads), default_initializer=ones_)  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls
            self.add_parameter("relative_position_bias_table", self.relative_position_bias_table)

            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(window_size[0])
            coords_w = paddle.arange(window_size[1])
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

            coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
            coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
            relative_coords = coords_flatten_1 - coords_flatten_2
            relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                paddle.zeros([(window_size[0] * window_size[1] + 1), ]*2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index",
                             relative_position_index)

        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_position_bias_table is not None:
            trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = paddle.concat([self.q_bias, paddle.zeros_like(self.v_bias), self.v_bias])
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        if self.relative_position_bias_table is not None:
            index = self.relative_position_index.reshape([-1])

            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index)
            relative_position_bias = relative_position_bias.reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1
            ])  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
        
        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv
        
        return x


class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer="nn.LayerNorm",
                 window_size=None, attn_head_dim=None):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=1e-6)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=1e-6)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            x = init_values * paddle.ones([dim], dtype="float32")
            self.gamma_1 = self.create_parameter(shape=(dim,), default_initializer=ones_)
            self.add_parameter("gamma_1", self.gamma_1)
            self.gamma_2 = self.create_parameter(shape=(dim,), default_initializer=ones_)
            self.add_parameter("gamma_2", self.gamma_2)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv
        
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__() 
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = \
            self.create_parameter(shape=(self.num_relative_distance, num_heads), default_initializer=ones_)  # 2*Wh-1 * 2*Ww-1, nH
        self.add_parameter("relative_position_bias_table", self.relative_position_bias_table)
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros([(window_size[0] * window_size[1] + 1), ]*2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                        relative_position_index)
    
    def forward(self, ):
        index = self.relative_position_index.reshape([-1])
        relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1
            ])  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww  # nH, Wh*Ww, Wh*Ww



class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer='nn.LayerNorm', init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        # self.mask_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("pos_embed", self.pos_embed)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = np.linspace(0, drop_path_rate, depth, dtype=np.float32)
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None) 
                for i in range(depth)])
        self.norm = Identity() if use_mean_pooling else eval(norm_layer)(embed_dim, epsilon=1e-6)
        self.fc_norm = eval(norm_layer)(embed_dim, epsilon=1e-6) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        # trunc_normal_(self.mask_token)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            x = self.head.weight.multiply(paddle.to_tensor([init_values]))
            self.head.weight = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))
            x = self.head.bias.multiply(paddle.to_tensor([init_values]))                            
            self.head.bias = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))

    def fix_init_weight(self):
        def rescale(param, layer_id):
            x = param.divide(paddle.to_tensor([math.sqrt(2.0 * layer_id)]))
            param = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def get_num_layers(self):
        return len(self.blocks)
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).transpose([0, 3, 1, 2]),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.transpose([0, 2, 3, 1]).reshape(1, -1, dim)
        return paddle.concat([class_pos_embed.unsqueeze(0), patch_pos_embed], dim=1)

    def forward_features(self, x, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.pos_embed
        
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]
        
    def forward(self, x, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        x = self.forward_features(x, return_patch_tokens=return_patch_tokens, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x

    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat([cls_tokens, x], axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                # use last norm for all intermediate layers
                if l in layer_id:
                    if norm_output:
                        x_norm = self.fc_norm(self.norm(x[:, 1:]))
                        output_list.append(x_norm)
                    else:
                        output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_intermediate_layers(self, x, use_last_norm=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat([cls_tokens, x], axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            if use_last_norm:
                features.append(self.norm(x))
            else:
                features.append(x)

        return features

def beit_base_patch16_224(pretrained=False, finetune_weight=None, model_filter_name='', **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, # qkv_bias=True,
        norm_layer="nn.LayerNorm", **kwargs)
    if finetune_weight is not None:
        checkpoint = paddle.load(finetune_weight)

        print("Load ckpt from %s" % finetune_weight)
        checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (model_filter_name != ''):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('encoder.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
            print("Expand the shared relative position embedding to each transformer block. ")
            num_layers = model.get_num_layers()
            rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
            for i in range(num_layers):
                checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

            checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "Teacher" in key:
                checkpoint_model.pop(key)
            elif "Student" in key:
                checkpoint_model[key.strip("Student.")] = checkpoint_model.pop(key)

            if "relative_position_index" in key:
                checkpoint_model.pop(key)

            if "relative_position_bias_table" in key:
                rel_pos_bias = checkpoint_model[key]
                src_num_pos, num_attn_heads = rel_pos_bias.shape
                dst_num_pos, _ = model.state_dict()[key].shape
                dst_patch_shape = model.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)

                if src_size != dst_size:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                        key, src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))
                    
                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            paddle.to_tensor(f(dx, dy), place=rel_pos_bias.place).reshape([-1, 1]))

                    rel_pos_bias = paddle.concat(all_rel_pos_bias, axis=-1)

                    new_rel_pos_bias = paddle.concat([rel_pos_bias, extra_tokens], axis=0)
                    checkpoint_model[key] = new_rel_pos_bias
        # interpolate position embedding
        if ('pos_embed' in checkpoint_model) and (model.pos_embed is not None):
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = paddle.nn.functional.interpolate(
                    pos_tokens, size=[new_size, new_size], mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = paddle.concat([extra_tokens, pos_tokens], axis=1)
                checkpoint_model['pos_embed'] = new_pos_embed
        model.set_dict(checkpoint_model)
    model.default_cfg = _cfg()
    return model

''' pretrain '''

class VisionTransformerForMaskedImageModeling(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads

        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)
        self.mask_token = self.create_parameter(shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("mask_token", self.mask_token)
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("pos_embed", self.pos_embed)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = np.linspace(0, drop_path_rate, depth, dtype=np.float32)
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim)
                for i in range(depth)
                ])
        self.norm = eval(norm_layer)(embed_dim, epsilon=1e-6)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        trunc_normal_(self.mask_token)
        trunc_normal_(self.lm_head.weight)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            x = param.divide(paddle.to_tensor([math.sqrt(2.0 * layer_id)]))
            param = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)
    
    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand((batch_size, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand((batch_size, seq_len, -1))

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).astype(mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
        
        x = self.norm(x)
        return x

    
    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = paddle.zeros([x.shape[0], self.patch_embed.num_patches], dtype=paddle.bool).set_device(x.device)
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])

    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = paddle.zeros([x.shape[0], self.patch_embed.num_patches], dtype=paddle.bool).set_device(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand((batch_size, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand((batch_size, seq_len, -1))

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).astype(mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)
        
        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x)
            q, k, v = x.chunks(3, axis=-1)
            b, n, c = q.shape
            q = q.reshape(b, n, self.num_heads, -1).transpose([0, 2, 1, 3])
            k = k.reshape(b, n, self.num_heads, -1).transpose([0, 2, 1, 3])
            v = v.reshape(b, n, self.num_heads, -1).transpose([0, 2, 1, 3])
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]
        
        return x, q, k, v

    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = paddle.zeros([x.shape[0], self.patch_embed.num_patches], dtype=paddle.bool).set_device(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand((batch_size, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand((batch_size, seq_len, -1))

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).astype(mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
    
    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand((batch_size, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.patch_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
        
        
class VisionTransformerForMaskedImageModelingCLS(VisionTransformerForMaskedImageModeling):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 early_layers=6, head_layers=2, shared_lm_head=True):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias, init_std=init_std)

        self.early_layers = early_layers
        print(f'early layer {early_layers}, late layer {depth - early_layers}, condenser head layers {head_layers}, shared_lm_head {shared_lm_head}')

        dpr = np.linspace(0, drop_path_rate, max(depth, early_layers + head_layers), dtype=np.float32)
        self.cls_pt_layers = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim)
                for i in range(early_layers, early_layers + head_layers)
                ])
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head
        if not self.shared_lm_head:
            self.cls_pt_norm = norm_layer(embed_dim)
            self.cls_pt_lm_head = nn.Linear(embed_dim, vocab_size)

            self.cls_pt_norm.apply(self._init_weights)
            self.cls_pt_lm_head.apply(self._init_weights)

    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            x = param.divide(paddle.to_tensor([math.sqrt(2.0 * layer_id)]))
            param = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, self.early_layers + layer_id + 1)
            rescale(layer.mlp.fc2.weight, self.early_layers + layer_id + 1)
    
    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand((batch_size, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand((batch_size, seq_len, -1))
        
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).astype(mask_token.dtype)
        x = x * (1 - w) + mask_token * w

        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 1:]

        x_cls_pt = paddle.concat((x[:, 0].unsqueeze(1), early_states), axis=1)
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)
        
    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = paddle.zeros([x.shape[0], self.patch_embed.num_patches], dtype=paddle.bool).set_device(x.device)
        x, x_cls_pt = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_all_tokens:
            return [self.lm_head(x), self.lm_head(x_cls_pt) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            # return the masked tokens
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]



def beit_base_patch16_224_8k_vocab_cls_pt(pretrained=False, pretrained_weight=None, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer="nn.LayerNorm", vocab_size=vocab_size, **kwargs)
    if pretrained:
        weight = paddle.load(pretrained_weight)
        model.set_dict(weight)
    model.default_cfg = _cfg()
    return model