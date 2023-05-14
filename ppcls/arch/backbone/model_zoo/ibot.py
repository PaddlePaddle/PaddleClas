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

from collections.abc import Callable
import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from .vision_transformer import VisionTransformer, Identity, trunc_normal_
import os
from ..legendary_models.swin_transformer import SwinTransformer
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "IBOT_ViT_small_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "IBOT_ViT_base_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "IBOT_ViT_large_patch16_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch7_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch14_224": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)

def to_2tuple(x):
    return tuple([x] * 2)

class IBOT_PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        pass
    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = self.proj(x).flatten(2).transpose((0, 2, 1))
        x = self.proj(x)
        return x

class MultiCropWrapper(nn.Layer):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = paddle.cumsum(
            paddle.unique_consecutive(
                paddle.to_tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx, output = 0, paddle.empty((0,))
        for end_idx in idx_crops:
            inp_x = paddle.concat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = paddle.concat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = paddle.concat((output, _out))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


class DINOHead(nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm=None,
        act_layer=nn.GELU,
        last_norm=None,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
        epsilon=1e-5,
        **kwargs
    ):
        super().__init__()
        if norm is not None:
            self.norm = eval(norm)(hidden_dim, epsilon=epsilon)
        if last_norm is not None:
            self.last_norm = eval(last_norm)(out_dim, epsilon=epsilon)
        else:
            self.last_norm = None
        if act_layer is not None:
            self.act = act_layer()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(self.act)

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(self.act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias_attr=False), dim=1
            )
            ones_(self.last_layer.weight_g)
            if norm_last_layer:
                self.last_layer.weight_g.stop_gradient = False

        else:
            self.last_layer = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        if self.last_layer is not None:
            x = nn.functional.normalize(x, axis=-1, p=2)
            x = self.last_layer(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return x


class IBOTHead(DINOHead):
    def __init__(
        self,
        *args,
        patch_out_dim=8192,
        norm=None,
        act_layer=nn.GELU,
        last_norm=None,
        nlayers=3,
        epsilon=1e-5,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
        shared_head=False,
        **kwargs
    ):
        super(IBOTHead, self).__init__(
            *args,
            norm=norm,
            act_layer=act_layer,
            last_norm=last_norm,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            norm_last_layer=norm_last_layer,
            **kwargs
        )
        if not shared_head:
            if bottleneck_dim > 0:
                self.last_layer2 = nn.utils.weight_norm(
                    nn.Linear(bottleneck_dim, patch_out_dim, bias_attr=False), dim=1
                )
                ones_(self.last_layer2.weight_g)
                if norm_last_layer:
                    self.last_layer2.weight_g.stop_gradient = False
            else:
                self.mlp2 = nn.Linear(hidden_dim, patch_out_dim)
                self.last_layer2 = None

            if last_norm is not None:
                self.last_norm2 = eval(last_norm)(patch_out_dim, epsilon=epsilon)
        else:
            if bottleneck_dim > 0:
                self.last_layer2 = self.last_layer
            else:
                self.mlp2 = self.mlp[-1]
                self.last_layer2 = None
            if last_norm is not None:
                self.last_norm2 = self.last_norm

    def forward(self, x):
        if len(x.shape) == 2:
            return super(IBOTHead, self).forward(x)

        if self.last_layer is not None:
            x = self.mlp(x)
            x = nn.functional.normalize(x, axis=-1, p=2)
            x1 = self.last_layer(x[:, 0])
            x2 = self.last_layer2(x[:, 1:])
        else:
            x = self.mlp[:-1](x)
            x1 = self.mlp[-1](x[:, 0])
            x2 = self.mlp2(x[:, 1:])

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        return x1, x2


class IBOTVisionTransformer(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        class_num=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        return_all_tokens=False,
        masked_im_modeling=False,
        **kwargs
    ):
        super(IBOTVisionTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            class_num,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            epsilon,
            **kwargs
        )
        self.return_all_tokens = return_all_tokens
        self.masked_im_modeling = masked_im_modeling
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = IBOT_PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        if self.masked_im_modeling:
            self.masked_embed = self.create_parameter(
                shape=[1, embed_dim], default_initializer=zeros_
            )
        # trunc_normal_(self.masked_embed)
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape((1, int(math.sqrt(N)), int(math.sqrt(N)), dim)).transpose((0, 3, 1, 2)),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.transpose((0, 2, 3, 1)).reshape((1, -1, dim))
        return paddle.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)


    def forward_features(self, x, mask=None, return_all_tokens=None):
        B, nc, w, h = x.shape
        x= self.patch_embed(x)
        # x = paddle.transpose(x, perm=[0, 2, 1])
        # C,N,HW = x.shape
        # H,W = int(self.img_size/self.patch_size),int(self.img_size/self.patch_size)
        # x = x.reshape([C,N,H,W])
        # mask image modeling
        if self.masked_im_modeling:
            assert mask is not None
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # if self.fc_norm is not None:
        #     x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = (
            self.return_all_tokens if return_all_tokens is None else return_all_tokens
        )

        if return_all_tokens:
            return x

        return x[:, 0]

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask, return_all_tokens=self.return_all_tokens)
        # x = self.head(x)

        return x

    def mask_model(self, x, mask):
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = paddle.where(mask.unsqueeze(-1), paddle.cast(self.masked_embed, x.dtype), x)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        return x

    def get_intermediate_layers(self, x, n=1,mask=None):

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # mask image modeling
        if self.masked_im_modeling:
            assert mask is not None
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def IBOT_ViT_small_patch16_224(patch_size=16, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def IBOT_ViT_base_patch16_224(patch_size=16, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def IBOT_ViT_large_patch16_224(patch_size=16, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )

    return model


class LinearClassifier(nn.Layer):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        normal_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x):
        # flatten
        x = x.reshape((x.shape[0], -1))

        # linear layer
        return self.linear(x)


class IBOT(nn.Layer):
    def __init__(self, **arch_config):
        super(IBOT, self).__init__()
        assert arch_config['arch'] in ['ViT_small', 'ViT_base','ViT_large'], f"arch can be only ['ViT_small', 'ViT_base','ViT_large']"
        model_name = "IBOT_" + arch_config['arch'] + "_patch" + str(arch_config['patch_size']) + "_224"
        model_name = eval(model_name)
        self.train_stage = arch_config['mode']
        self.arch_config = arch_config
        if arch_config['mode'] == 'pretrain':
            student = model_name(
                    patch_size=arch_config["patch_size"],
                    drop_path_rate=arch_config["drop_path"],
                    return_all_tokens=True,
                    masked_im_modeling=arch_config["use_masked_im_modeling"]
                ),
            teacher = model_name(
                patch_size=arch_config["patch_size"],
                return_all_tokens=True,
            ),
            embed_dim = student.embed_dim
            # multi-crop wrapper handles forward with inputs of different resolutions
            self.student = MultiCropWrapper(
                student,
                IBOTHead(
                    embed_dim,
                    arch_config["out_dim"],
                    patch_out_dim=arch_config["patch_out_dim"],
                    norm=arch_config["norm_in_head"],
                    act=arch_config["act_in_head"],
                    norm_last_layer=arch_config["norm_last_layer"],
                    shared_head=arch_config["shared_head"],
                )
            )
            self.teacher = MultiCropWrapper(
                teacher,
                IBOTHead(
                    embed_dim,
                    arch_config["out_dim"],
                    patch_out_dim=arch_config["patch_out_dim"],
                    norm=arch_config["norm_in_head"],
                    act=arch_config["act_in_head"],
                    shared_head=arch_config["shared_head_teacher"],
                )
            )

            # vit_s8 and vit_s16 are batch norm free models. here, we don't check bn
            self.teacher = paddle.DataParallel(self.teacher)
            self.teacher_without_ddp = self.teacher._layers
            self.student = paddle.DataParallel(self.student)

            # teacher and student start with the same weights
            self.teacher_without_ddp.load_dict(self.student.state_dict())

            # there is no backpropagation through the teacher, so no need for gradients
            for p in self.teacher.parameters():
                p.stop_gradient = True

        else:
            self.model = model_name(patch_size=arch_config['patch_size'], num_classes=0,use_mean_pooling=arch_config["avgpool_patchtokens"]== 1)
            feat_dim = self.model.embed_dim * (arch_config['n_last_blocks'] * int(arch_config["avgpool_patchtokens"] != 1) + int(arch_config["avgpool_patchtokens"] > 0) )
            self.model.eval()
            for p in self.model.parameters():
                p.stop_gradient = True

            self.linear_clf = paddle.DataParallel(LinearClassifier(feat_dim, arch_config['num_labels']))

            if os.path.isfile(arch_config['pretrained_weights']):
                state_dict = paddle.load(arch_config['pretrained_weights'])[arch_config['checkpoint_key']]
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                self.model.set_state_dict(new_state_dict)

            self.n_last_blocks = arch_config['n_last_blocks']
            self.avgpool = arch_config['avgpool_patchtokens']

    def forward(self, images,masks):
        if self.train_stage == 'pretrain':
            teacher_output = self.teacher(
                images[:self.arch_config["global_crops_number"]])  # only the 2 global views pass through the teacher
            student_output = self.student(images[:self.arch_config["global_crops_number"]],
                                     mask=masks[:self.arch_config["global_crops_number"]])  # all views pass through the student

            self.student.sublayers()[0].backbone.masked_im_modeling = False
            student_local_cls = self.student(images[self.arch_config["global_crops_number"]:])[0] if len(
                images) > self.arch_config["global_crops_number"] else None
            self.student.sublayers()[0].backbone.masked_im_modeling = self.arch_config["use_masked_im_modeling"]

            return teacher_output, student_output, student_local_cls

        else:    # finetune
            self.linear_clf.train()

            # forward
            with paddle.no_grad():
                intermediate_output = self.model.get_intermediate_layers(images, self.n_last_blocks)
                if self.avgpool == 0:
                    # norm(x[:, 0])
                    output = [x[:, 0] for x in intermediate_output]
                elif self.avgpool == 1:
                    # x[:, 1:].mean(1)
                    output = [paddle.mean(intermediate_output[-1][:, 1:], axis=1)]
                elif self.avgpool == 2:
                    # norm(x[:, 0]) + x[:, 1:].mean(1)
                    output = [x[:, 0] for x in intermediate_output] + [
                        paddle.mean(intermediate_output[-1][:, 1:], axis=1)
                    ]
                else:
                    assert False, "Unkown avgpool type {}".format(self.avgpool)

                output = paddle.concat(output, axis=-1)

            return self.linear_clf.forward(output)