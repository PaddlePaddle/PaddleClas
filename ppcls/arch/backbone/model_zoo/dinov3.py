import math
from functools import lru_cache

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "DINOv3_vits16": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vits16.pdparams",
    "DINOv3_vitb16": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vitb16.pdparams",
    "DINOv3_vitl16": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/dinov3-vitl16.pdparams"
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.full(shape=[], fill_value=1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


class LayerScale(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.lambda1 = self.create_parameter(
            shape=[config.hidden_size],
            default_initializer=Constant(value=config.layerscale_value))

    def forward(self, hidden_state):
        return hidden_state * self.lambda1


class DINOv3ViTEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        self.cls_token = self.create_parameter(
            shape=[1, 1, config.hidden_size],
            default_initializer=Normal(std=1.0))
        self.add_parameter("cls_token", self.cls_token)

        self.mask_token = self.create_parameter(
            shape=[1, 1, config.hidden_size],
            default_initializer=zeros_)
        self.add_parameter("mask_token", self.mask_token)

        if config.num_register_tokens > 0:
            self.register_tokens = self.create_parameter(
                shape=[1, config.num_register_tokens, config.hidden_size],
                default_initializer=Normal(std=0.02))
            self.add_parameter("register_tokens", self.register_tokens)
        else:
            self.register_tokens = paddle.zeros([1, 0, config.hidden_size])

        self.patch_embeddings = nn.Conv2D(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size)

    def forward(self, pixel_values, bool_masked_pos=None):
        batch_size = pixel_values.shape[0]

        patch_embeddings = self.patch_embeddings(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose([0, 2, 1])

        if bool_masked_pos is not None:
            mask_token = self.mask_token.astype(patch_embeddings.dtype)
            patch_embeddings = paddle.where(
                bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        cls_token = self.cls_token.expand([batch_size, -1, -1]).astype(patch_embeddings.dtype)
        register_tokens = self.register_tokens.expand([batch_size, -1, -1]).astype(patch_embeddings.dtype)
        embeddings = paddle.concat([cls_token, register_tokens, patch_embeddings], axis=1)

        return embeddings


@lru_cache(maxsize=32)
def get_patches_center_coordinates(num_patches_h, num_patches_w, dtype_str):
    coords_h = paddle.arange(0.5, num_patches_h, dtype='float32')
    coords_w = paddle.arange(0.5, num_patches_w, dtype='float32')

    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w

    mesh_h, mesh_w = paddle.meshgrid(coords_h, coords_w)
    coords = paddle.stack([mesh_h, mesh_w], axis=-1)
    coords = coords.reshape([-1, 2])

    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(coords, shift=None, jitter=None, rescale=None):
    if shift is not None:
        shift_hw = paddle.uniform([1, 2], min=-shift, max=shift, dtype=coords.dtype)
        coords = coords + shift_hw

    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = paddle.uniform([1, 2], min=-jitter_range, max=jitter_range, dtype=coords.dtype).exp()
        coords = coords * jitter_hw

    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = paddle.uniform([1], min=-rescale_range, max=rescale_range, dtype=coords.dtype).exp()
        coords = coords * rescale_hw

    return coords


class DINOv3ViTRopePositionEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads

        inv_freq = 1 / (self.base ** paddle.arange(0, 1, 4 / self.head_dim, dtype='float32'))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_size
        num_patches_w = width // self.config.patch_size

        dtype_str = str(pixel_values.dtype)
        coords = get_patches_center_coordinates(num_patches_h, num_patches_w, dtype_str)

        if self.training:
            shift = getattr(self.config, 'pos_embed_shift', None)
            jitter = getattr(self.config, 'pos_embed_jitter', None)
            rescale = getattr(self.config, 'pos_embed_rescale', None)
            coords = augment_patches_center_coordinates(coords, shift, jitter, rescale)

        angles = 2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2)
        angles = paddle.tile(angles, [1, 2])

        cos = paddle.cos(angles)
        sin = paddle.sin(angles)

        dtype = pixel_values.dtype
        return cos.astype(dtype), sin.astype(dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix_tokens, q_patches = q.split([num_prefix_tokens, num_patches], axis=-2)
    k_prefix_tokens, k_patches = k.split([num_prefix_tokens, num_patches], axis=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = paddle.concat([q_prefix_tokens, q_patches], axis=-2)
    k = paddle.concat([k_prefix_tokens, k_patches], axis=-2)

    return q, k


class DINOv3ViTAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scaling = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.value_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.query_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim,
                                bias_attr=config.proj_bias)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch_size, patches, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([batch_size, patches, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, axis=-1)

        if self.training and self.dropout > 0.0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([batch_size, patches, -1])

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DINOv3ViTMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                 bias_attr=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias_attr=config.mlp_bias)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTGatedMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                    bias_attr=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,
                                 bias_attr=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,
                                   bias_attr=config.mlp_bias)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DINOv3ViTLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)

        layer_scale_init = Constant(value=config.layerscale_value)
        self.layer_scale1 = LayerScale(config)

        if config.drop_path_rate > 0.0:
            self.drop_path = DropPath(config.drop_path_rate)
        else:
            self.drop_path = Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)

        self.layer_scale2 = LayerScale(config)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings)
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class DINOv3ViTModel(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=384,
                 depth=12,
                 num_heads=6,
                 mlp_ratio=4,
                 qkv_bias=False,
                 query_bias=None,
                 key_bias=None,
                 value_bias=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_gated_mlp=False,
                 num_register_tokens=0,
                 layerscale_value=1.0,
                 rope_theta=100.0,
                 pos_embed_shift=None,
                 pos_embed_jitter=None,
                 pos_embed_rescale=None,
                 out_indices=None,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 use_gradient_checkpointing=False,
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        self.out_indices = out_indices if out_indices is not None else []
        self.use_gradient_checkpointing = use_gradient_checkpointing

        class Config:
            pass
        config = Config()
        config.image_size = img_size
        config.patch_size = patch_size
        config.num_channels = in_chans
        config.hidden_size = embed_dim
        config.intermediate_size = int(embed_dim * mlp_ratio)
        config.num_hidden_layers = depth
        config.num_attention_heads = num_heads
        config.hidden_act = 'gelu'
        config.attention_dropout = attn_drop_rate
        config.initializer_range = 0.02
        config.layer_norm_eps = epsilon
        config.rope_theta = rope_theta

        if query_bias is not None:
            config.query_bias = query_bias
        else:
            config.query_bias = qkv_bias
        if key_bias is not None:
            config.key_bias = key_bias
        else:
            config.key_bias = qkv_bias
        if value_bias is not None:
            config.value_bias = value_bias
        else:
            config.value_bias = qkv_bias

        config.proj_bias = True
        config.mlp_bias = True
        config.layerscale_value = layerscale_value
        config.drop_path_rate = drop_path_rate
        config.use_gated_mlp = use_gated_mlp
        config.num_register_tokens = num_register_tokens
        config.pos_embed_shift = pos_embed_shift
        config.pos_embed_jitter = pos_embed_jitter
        config.pos_embed_rescale = pos_embed_rescale
        self.config = config

        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)

        self.layers = nn.LayerList([
            DINOv3ViTLayer(config) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, epsilon=epsilon)

        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        hidden_states = self.embeddings(x)
        position_embeddings = self.rope_embeddings(x)

        backbone_features = []
        if 0 in self.out_indices:
            backbone_features.append(hidden_states)

        for idx, layer_module in enumerate(self.layers):
            if self.use_gradient_checkpointing and self.training:
                try:
                    hidden_states = paddle.distributed.fleet.utils.recompute(
                        layer_module,
                        hidden_states,
                        position_embeddings=position_embeddings,
                        preserve_rng_state=False,
                        use_reentrant=False)
                except:
                    hidden_states = layer_module(
                        hidden_states,
                        position_embeddings=position_embeddings)
            else:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings)
            if (idx + 1) in self.out_indices:
                backbone_features.append(hidden_states)

        sequence_output = self.norm(hidden_states)
        if -1 in self.out_indices or len(self.out_indices) == 0:
            backbone_features.append(sequence_output)

        pooled_output = sequence_output[:, 0, :]

        if len(self.out_indices) > 0:
            return backbone_features
        return pooled_output

    def forward(self, x):
        features = self.forward_features(x)
        if isinstance(features, list):
            return features
        x = self.head(features)
        return x

def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type.")


def DINOv3_vits16(pretrained=False, use_ssld=False, **kwargs):
    model = DINOv3ViTModel(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        num_register_tokens=4,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vits16"],
        use_ssld=use_ssld)
    return model



def DINOv3_vitb16(pretrained=False, use_ssld=False, **kwargs):
    model = DINOv3ViTModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        num_register_tokens=4,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DINOv3_vitb16"],
        use_ssld=use_ssld)
    return model


def DINOv3_vitl16(pretrained=False, use_ssld=False, **kwargs):
    model = DINOv3ViTModel(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        num_register_tokens=4,
        drop_path_rate=0.0,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS.get("DINOv3_vitl16", ""),
        use_ssld=use_ssld)
    return model
