# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# reference: https://arxiv.org/abs/2010.11929

import paddle
import paddle.nn as nn
from paddle.framework import LayerHelper, in_dynamic_mode
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from paddle.incubate.nn.functional import (
    fused_layer_norm,
    fused_linear,
    variable_length_memory_efficient_attention
)
from paddle.nn.quant import weight_quantize, weight_only_linear
from ....utils.save_load import get_pretrain_state_dict, get_pretrain_state_dict_from_url
from ....utils.import_utils import is_paddleclas_ops_available

if is_paddleclas_ops_available():
    from paddleclas_ops import (
        qkv_transpose_split,
        transpose_remove_padding
    )
else:
    raise RuntimeError(
        "The paddleclas_ops is not installed. You can read the docs and install it by hand,"
        "you can refer to: csrc/README.md"
    )


MODEL_URLS = {
    "Fused_ViT_small_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "Fused_ViT_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "Fused_ViT_base_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
    "Fused_ViT_base_patch32_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
    "Fused_ViT_large_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
    "Fused_ViT_large_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_384_pretrained.pdparams",
    "Fused_ViT_large_patch32_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch32_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

def to_2tuple(x):
    return tuple([x] * 2)

def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["bias"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


class FusedVisionTransformer(nn.Layer):
    """ Fused Vision Transformer with support for patch input
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 use_weight_only=False,
                 quant_type="weight_only_int8",
                 **kwargs):
        super().__init__()
        self.dtype = self._helper.get_default_dtype()

        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.depth = depth
        self.scale = qk_scale or self.head_dim**-0.5
        self.norm_func = fused_layer_norm
        self.linear = fused_linear

        self.use_weight_only = use_weight_only
        self.quant_type = quant_type
        self.create_params_type = self.get_weight_create_dtype()
        self._norm_weight_dtype = "float32"

        if self.use_weight_only:
            assert (
                self.quant_type == "weight_only_int8" or self.quant_type == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4' \
                but received quant_type: {}".format(
                self.quant_type
            )
            self.quant_bits = int(self.quant_type[-1])
            self.weight_dtype = "int" + str(self.quant_bits)

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_embed_proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (self.img_size[1] // self.patch_size[1]) * \
            (self.img_size[0] // self.patch_size[0])

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=trunc_normal_
        )
        self.add_parameter("pos_embed", self.pos_embed)
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=trunc_normal_
        )
        self.add_parameter("cls_token", self.cls_token)

        self.norm1_weights, self.norm1_biases = [], []
        self.attn_qkv_weights, self.attn_qkv_biases = [], []
        self.attn_proj_weights, self.attn_proj_biases = [], []
        self.norm2_weights, self.norm2_biases = [], []
        self.mlp_fc1_weights, self.mlp_fc1_biases = [], []
        self.mlp_fc2_weights, self.mlp_fc2_biases = [], []

        if self.use_weight_only:
            self.attn_qkv_weights_scale = []
            self.attn_proj_weights_scale = []
            self.mlp_fc1_weights_scale = []
            self.mlp_fc2_weights_scale = []

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self._init_weight_shape(mlp_hidden_dim)

        for i in range(self.depth):
            norm1_weight = self.create_parameter(
                shape=self.norm1_weight_shape,
                default_initializer=ones_,
                dtype=self._norm_weight_dtype
            )
            norm1_bias = self.create_parameter(
                shape=self.norm1_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self._norm_weight_dtype
            )

            attn_qkv_weight = self.create_parameter(
                shape=self.attn_qkv_weight_shape,
                default_initializer=ones_,
                dtype=self.create_params_type
            )
            attn_qkv_bias = self.create_parameter(
                shape=self.attn_qkv_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self.dtype
            )

            attn_proj_weight = self.create_parameter(
                shape=self.attn_proj_weight_shape,
                default_initializer=ones_,
                dtype=self.create_params_type
            )
            attn_proj_bias = self.create_parameter(
                shape=self.attn_proj_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self.dtype
            )

            norm2_weight = self.create_parameter(
                shape=self.norm2_weight_shape,
                default_initializer=ones_,
                dtype=self._norm_weight_dtype
            )
            norm2_bias = self.create_parameter(
                shape=self.norm2_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self._norm_weight_dtype
            )

            mlp_fc1_weight = self.create_parameter(
                shape=self.mlp_fc1_weight_shape,
                default_initializer=ones_,
                dtype=self.create_params_type
            )
            mlp_fc1_bias = self.create_parameter(
                shape=self.mlp_fc1_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self.dtype
            )

            mlp_fc2_weight = self.create_parameter(
                shape=self.mlp_fc2_weight_shape,
                default_initializer=ones_,
                dtype=self.create_params_type
            )
            mlp_fc2_bias = self.create_parameter(
                shape=self.mlp_fc2_bias_shape,
                default_initializer=zeros_,
                is_bias=True,
                dtype=self.dtype
            )

            self.norm1_weights.append(norm1_weight)
            self.norm1_biases.append(norm1_bias)
            self.attn_qkv_weights.append(attn_qkv_weight)
            self.attn_qkv_biases.append(attn_qkv_bias)
            self.attn_proj_weights.append(attn_proj_weight)
            self.attn_proj_biases.append(attn_proj_bias)
            self.norm2_weights.append(norm2_weight)
            self.norm2_biases.append(norm2_bias)
            self.mlp_fc1_weights.append(mlp_fc1_weight)
            self.mlp_fc1_biases.append(mlp_fc1_bias)
            self.mlp_fc2_weights.append(mlp_fc2_weight)
            self.mlp_fc2_biases.append(mlp_fc2_bias)

            self.add_parameter("blocks_{}_norm1_weight".format(i), norm1_weight)
            self.add_parameter("blocks_{}_norm1_bias".format(i), norm1_bias)
            self.add_parameter("blocks_{}_attn_qkv_weight".format(i), attn_qkv_weight)
            self.add_parameter("blocks_{}_attn_qkv_bias".format(i), attn_qkv_bias)
            self.add_parameter("blocks_{}_attn_proj_weight".format(i), attn_proj_weight)
            self.add_parameter("blocks_{}_attn_proj_bias".format(i), attn_proj_bias)
            self.add_parameter("blocks_{}_norm2_weight".format(i), norm2_weight)
            self.add_parameter("blocks_{}_norm2_bias".format(i), norm2_bias)
            self.add_parameter("blocks_{}_mlp_fc1_weight".format(i), mlp_fc1_weight)
            self.add_parameter("blocks_{}_mlp_fc1_bias".format(i), mlp_fc1_bias)
            self.add_parameter("blocks_{}_mlp_fc2_weight".format(i), mlp_fc2_weight)
            self.add_parameter("blocks_{}_mlp_fc2_bias".format(i), mlp_fc2_bias)

            if self.use_weight_only:
                attn_qkv_weight_scale = self.create_parameter(
                    shape=[3 * self.num_heads * self.head_dim],
                    default_initializer=zeros_,
                    dtype=paddle.float32,
                    is_bias=False
                )
                attn_proj_weight_scale = self.create_parameter(
                    shape=[self.embed_dim],
                    default_initializer=zeros_,
                    dtype=paddle.float32,
                    is_bias=False
                )
                mlp_fc1_weight_scale = self.create_parameter(
                    shape=[mlp_hidden_dim],
                    default_initializer=zeros_,
                    dtype=paddle.float32,
                    is_bias=False
                )
                mlp_fc2_weight_scale = self.create_parameter(
                    shape=[self.embed_dim],
                    default_initializer=zeros_,
                    dtype=paddle.float32,
                    is_bias=False
                )

                self.attn_qkv_weights_scale.append(attn_qkv_weight_scale)
                self.attn_proj_weights_scale.append(attn_proj_weight_scale)
                self.mlp_fc1_weights_scale.append(mlp_fc1_weight_scale)
                self.mlp_fc2_weights_scale.append(mlp_fc2_weight_scale)

                self.add_parameter("blocks_{}_attn_qkv_weight_scale".format(i), attn_qkv_weight_scale)
                self.add_parameter("blocks_{}_attn_proj_weight_scale".format(i), attn_proj_weight_scale)
                self.add_parameter("blocks_{}_mlp_fc1_weight_scale".format(i), mlp_fc1_weight_scale)
                self.add_parameter("blocks_{}_mlp_fc2_weight_scale".format(i), mlp_fc2_weight_scale)

        self.norm_weight = self.create_parameter(
            shape=[embed_dim],
            default_initializer=ones_,
            dtype=self._norm_weight_dtype
        )
        self.norm_bias = self.create_parameter(
            shape=[embed_dim],
            is_bias=True,
            default_initializer=zeros_,
            dtype=self._norm_weight_dtype
        )
        self.head_weight = self.create_parameter(
            shape=[embed_dim, class_num],
            default_initializer=ones_,
            dtype=self.dtype
        )
        self.head_bias = self.create_parameter(
            shape=[class_num],
            is_bias=True,
            default_initializer=zeros_,
            dtype=self.dtype
        )

    def _init_weight_shape(self, mlp_hidden_dim):
        self.norm1_weight_shape = [self.embed_dim]
        self.norm1_bias_shape = [self.embed_dim]
        self.attn_qkv_weight_shape = (
            [3 * self.num_heads * self.head_dim, self.embed_dim]
            if self.use_weight_only
            else [self.embed_dim, 3 * self.num_heads * self.head_dim, ]
        )
        self.attn_qkv_bias_shape = [3 * self.num_heads * self.head_dim]
        self.attn_proj_weight_shape = (
            [self.embed_dim, self.num_heads * self.head_dim]
            if self.use_weight_only
            else [self.num_heads * self.head_dim, self.embed_dim]
        )
        self.attn_proj_bias_shape = [self.num_heads * self.head_dim]
        self.norm2_weight_shape = [self.embed_dim]
        self.norm2_bias_shape = [self.embed_dim]
        self.mlp_fc1_weight_shape = (
            [mlp_hidden_dim, self.embed_dim]
            if self.use_weight_only
            else [self.embed_dim, mlp_hidden_dim]
        )
        self.mlp_fc1_bias_shape = [mlp_hidden_dim]
        self.mlp_fc2_weight_shape = (
            [self.embed_dim, mlp_hidden_dim]
            if self.use_weight_only
            else [mlp_hidden_dim, self.embed_dim]
        )
        self.mlp_fc2_bias_shape = [self.embed_dim]

        if self.use_weight_only and self.quant_bits == 4:
            self.attn_qkv_weight_shape[0] //= 2
            self.attn_proj_weight_shape[0] //= 2
            self.mlp_fc1_weight_shape[0] //= 2
            self.mlp_fc2_weight_shape[0] //= 2

    def get_weight_create_dtype(self):
        if self.use_weight_only:
            return "int8"
        else:
            return self.dtype

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        self.pos_embed.set_value(state_dict["pos_embed"].astype(self.dtype))
        self.cls_token.set_value(state_dict["cls_token"].astype(self.dtype))
        self.patch_embed_proj.weight.set_value(state_dict["patch_embed.proj.weight"].astype(self.dtype))
        self.patch_embed_proj.bias.set_value(state_dict["patch_embed.proj.bias"].astype(self.dtype))
        for i in range(self.depth):
            self.norm1_weights[i].set_value(state_dict["blocks.{}.norm1.weight".format(i)].astype(self._norm_weight_dtype))
            self.norm1_biases[i].set_value(state_dict["blocks.{}.norm1.bias".format(i)].astype(self._norm_weight_dtype))

            if self.use_weight_only:
                attn_qkv_weight_tensor = paddle.to_tensor(state_dict["blocks.{}.attn.qkv.weight".format(i)].astype(self.dtype))
                attn_qkv_quanted_weight_tensor, attn_qkv_weight_scale_tensor = weight_quantize(
                    attn_qkv_weight_tensor, algo=self.quant_type
                )
                self.attn_qkv_weights[i].set_value(attn_qkv_quanted_weight_tensor)
                self.attn_qkv_weights_scale[i].set_value(attn_qkv_weight_scale_tensor)
            else:
                self.attn_qkv_weights[i].set_value(state_dict["blocks.{}.attn.qkv.weight".format(i)].astype(self.dtype))
            self.attn_qkv_biases[i].set_value(state_dict["blocks.{}.attn.qkv.bias".format(i)].astype(self.dtype))

            if self.use_weight_only:
                attn_proj_weight_tensor = paddle.to_tensor(state_dict["blocks.{}.attn.proj.weight".format(i)].astype(self.dtype))
                attn_proj_quanted_weight_tensor, attn_proj_weight_scale_tensor = weight_quantize(
                    attn_proj_weight_tensor, algo=self.quant_type
                )
                self.attn_proj_weights[i].set_value(attn_proj_quanted_weight_tensor)
                self.attn_proj_weights_scale[i].set_value(attn_proj_weight_scale_tensor)
            else:
                self.attn_proj_weights[i].set_value(state_dict["blocks.{}.attn.proj.weight".format(i)].astype(self.dtype))
            self.attn_proj_biases[i].set_value(state_dict["blocks.{}.attn.proj.bias".format(i)].astype(self.dtype))

            self.norm2_weights[i].set_value(state_dict["blocks.{}.norm2.weight".format(i)].astype(self._norm_weight_dtype))
            self.norm2_biases[i].set_value(state_dict["blocks.{}.norm2.bias".format(i)].astype(self._norm_weight_dtype))

            if self.use_weight_only:
                mlp_fc1_weight_tensor = paddle.to_tensor(state_dict["blocks.{}.mlp.fc1.weight".format(i)].astype(self.dtype))
                mlp_fc1_quanted_weight_tensor, mlp_fc1_weight_scale_tensor = weight_quantize(
                    mlp_fc1_weight_tensor, algo=self.quant_type
                )
                self.mlp_fc1_weights[i].set_value(mlp_fc1_quanted_weight_tensor)
                self.mlp_fc1_weights_scale[i].set_value(mlp_fc1_weight_scale_tensor)
            else:
                self.mlp_fc1_weights[i].set_value(state_dict["blocks.{}.mlp.fc1.weight".format(i)].astype(self.dtype))
            self.mlp_fc1_biases[i].set_value(state_dict["blocks.{}.mlp.fc1.bias".format(i)].astype(self.dtype))

            if self.use_weight_only:
                mlp_fc2_weight_tensor = paddle.to_tensor(state_dict["blocks.{}.mlp.fc2.weight".format(i)].astype(self.dtype))
                mlp_fc2_quanted_weight_tensor, mlp_fc2_weight_scale_tensor = weight_quantize(
                    mlp_fc2_weight_tensor, algo=self.quant_type
                )
                self.mlp_fc2_weights[i].set_value(mlp_fc2_quanted_weight_tensor)
                self.mlp_fc2_weights_scale[i].set_value(mlp_fc2_weight_scale_tensor)
            else:
                self.mlp_fc2_weights[i].set_value(state_dict["blocks.{}.mlp.fc2.weight".format(i)].astype(self.dtype))
            self.mlp_fc2_biases[i].set_value(state_dict["blocks.{}.mlp.fc2.bias".format(i)].astype(self.dtype))

        self.norm_weight.set_value(state_dict["norm.weight"].astype(self._norm_weight_dtype))
        self.norm_bias.set_value(state_dict["norm.bias"].astype(self._norm_weight_dtype))
        self.head_weight.set_value(state_dict["head.weight"].astype(self.dtype))
        self.head_bias.set_value(state_dict["head.bias"].astype(self.dtype))

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0:
            ln_out = self.norm_func(src, self.norm1_weights[i], self.norm1_biases[i], self.epsilon)
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if self.use_weight_only:
            return weight_only_linear(
                ln_out,
                weight=self.attn_qkv_weights[i],
                bias=self.attn_qkv_biases[i],
                weight_scale=self.attn_qkv_weights_scale[i],
                weight_dtype=self.weight_dtype
            )

        if float(paddle.version.cuda()) < 11.6:
            qkv_out = paddle.matmul(ln_out, self.attn_qkv_weights[i])
            if self.attn_qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.attn_qkv_biases[i])
            return qkv_out
        else:
            return self.linear(ln_out, self.attn_qkv_weights[i], self.attn_qkv_biases[i])

    def compute_qkv(self, src, residual_input, i):
        ln_out = self.compute_layernorm_before_qkv(src, i)
        qkv_out = self.compute_qkv_linear(ln_out, i)
        return qkv_out, residual_input

    def compute_fmha(self, qkv_out, padding_offset, seq_lens, input_ids, i):
        q_out, k_out, v_out = qkv_transpose_split(
            qkv_out, padding_offset, seq_lens, input_ids, self.num_heads, self.head_dim
        )
        # cutlass fmha
        qktv_out = variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens,
            None,
            scale=self.scale
        )
        return transpose_remove_padding(qktv_out, seq_lens, padding_offset)

    def compute_out_linear(self, fmha_out, i):
        if self.use_weight_only:
            return weight_only_linear(
                fmha_out,
                weight=self.attn_proj_weights[i],
                weight_scale=self.attn_proj_weights_scale[i],
                weight_dtype=self.weight_dtype
            )

        return paddle.matmul(fmha_out, self.attn_proj_weights[i])

    def compute_attn(self, qkv_out, padding_offset, seq_lens, input_ids, i):
        fmha_out = self.compute_fmha(qkv_out, padding_offset, seq_lens, input_ids, i)
        out_linear_out = self.compute_out_linear(fmha_out, i)
        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        """
        tmp_out = layernorm(out_linear_out + attn_proj_biases[i] + residual_input)
        """
        norm_out = self.norm_func(
            out_linear_out,
            norm_weight=self.norm2_weights[i],
            norm_bias=self.norm2_biases[i],
            epsilon=self.epsilon,
            bias=self.attn_proj_biases[i],
            residual=residual_input,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]
        return tmp_out, residual_input

    def compute_ffn1(self, tmp_out, i):
        if self.use_weight_only:
            return weight_only_linear(
                tmp_out,
                weight=self.mlp_fc1_weights[i],
                weight_scale=self.mlp_fc1_weights_scale[i],
                weight_dtype=self.weight_dtype,
            )

        return paddle.matmul(tmp_out, self.mlp_fc1_weights[i])

    def compute_ffn2(self, ffn1_out, i):
        if self.use_weight_only:
            return weight_only_linear(
                ffn1_out,
                weight=self.mlp_fc2_weights[i],
                weight_scale=self.mlp_fc2_weights_scale[i],
                weight_dtype=self.weight_dtype,
            )

        return paddle.matmul(ffn1_out, self.mlp_fc2_weights[i])

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                norm_weight=self.norm1_weights[i + 1],
                norm_bias=self.norm1_biases[i + 1],
                epsilon=self.epsilon,
                bias=self.mlp_fc2_biases[i],
                residual=residual_input
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = self.norm_func(
                ffn2_out,
                norm_weight=self.norm_weight,
                norm_bias=self.norm_bias,
                epsilon=self.epsilon,
                bias=self.mlp_fc2_biases[i],
                residual=residual_input
            )[0]
        return tmp_out, residual_input

    def compute_head_linear(self, ln_out):
        if float(paddle.version.cuda()) < 11.6:
            qkv_out = paddle.matmul(ln_out, self.head_weight)
            if self.head_bias is not None:
                qkv_out = paddle.add(qkv_out, self.head_bias)
            return qkv_out
        else:
            return self.linear(ln_out, self.head_weight, self.head_bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.patch_embed_proj(x).flatten(2).transpose((0, 2, 1))

        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed

        batch, seq_len, _ = x.shape
        padding_offset = paddle.zeros([seq_len * batch], dtype='int32')
        seq_lens = paddle.full([batch], seq_len, dtype='int32')
        input_ids = paddle.full([batch, seq_len], 0, dtype='int32')

        x = x.reshape([-1, x.shape[-1]])
        residual_input = x
        for i in range(self.depth):
            qkv_out, residual_input = self.compute_qkv(x, residual_input, i)
            out_linear_out = self.compute_attn(
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                i
            )

            # qkv proj linear + layernorm2
            tmp_out, residual_input = self.compute_ffn_layernorm(out_linear_out, residual_input, i)

            # mlp ffn1 matmul
            ffn1_out = self.compute_ffn1(tmp_out, i)
            ffn1_out = fused_act_bias_wrapper(ffn1_out, self.mlp_fc1_biases[i])

            # mlp ffn2 matmul
            ffn2_out = self.compute_ffn2(ffn1_out, i)

            # layernorm1 + residual_add_bias
            tmp_out, residual_input = self.compute_bias_residual_layernorm(ffn2_out, residual_input, i, self.depth)
            x = tmp_out
        x = x.reshape((batch, seq_len, -1))
        index = paddle.zeros([1], dtype="int32")
        x = paddle.index_select(x, index, axis=1).reshape((batch, self.embed_dim))
        x = self.compute_head_linear(x)

        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        weight_state_dict = get_pretrain_state_dict_from_url(model_url, use_ssld=use_ssld)
        model.set_state_dict(weight_state_dict)
    elif isinstance(pretrained, str):
        weight_state_dict = get_pretrain_state_dict(pretrained)
        model.set_state_dict(weight_state_dict)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def Fused_ViT_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_small_patch16_224"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_base_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_base_patch16_384"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_base_patch32_384(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_base_patch32_384"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_large_patch16_224"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_large_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_large_patch16_384"],
        use_ssld=use_ssld)
    return model


def Fused_ViT_large_patch32_384(pretrained=False, use_ssld=False, **kwargs):
    model = FusedVisionTransformer(
        img_size=384,
        patch_size=32,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["Fused_ViT_large_patch32_384"],
        use_ssld=use_ssld)
    return model