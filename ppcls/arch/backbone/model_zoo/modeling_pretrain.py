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

# Code was heavily based on https://github.com/facebookresearch/deit
# reference: https://arxiv.org/abs/2012.12877

import math
import numpy as np
import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from .modeling_finetune import Block, PatchEmbed, RelativePositionBias, _cfg, zeros_, ones_, Identity
trunc_normal_ = TruncatedNormal(std=.02)


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