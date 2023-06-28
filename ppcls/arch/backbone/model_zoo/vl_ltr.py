import os.path as osp
import os
from typing import Tuple, Union
import numpy as np
import paddle
from paddle.nn import functional as F
from paddle import nn
from paddle.nn.initializer import Assign, Normal, Constant, TruncatedNormal
from ....utils.save_load import load_dygraph_pretrain_from_url_state_dict
from .foundation_vit import MODEL_URLS, VisionTransformer, QuickGELU


def interpolate_pos_embed(pos_embed_checkpoint: paddle.Tensor,
                          new_patch_size,
                          num_extra_tokens=1):
    # interpolate position embedding
    if pos_embed_checkpoint.ndim > 2:
        pos_embed_checkpoint = paddle.squeeze(pos_embed_checkpoint)
    embedding_size = pos_embed_checkpoint.shape[1]
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[0] - num_extra_tokens)**0.5)
    # height (== width) for the new position embedding
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:num_extra_tokens, :]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[num_extra_tokens:, :]
    pos_tokens = paddle.reshape(pos_tokens,
                                [-1, orig_size, orig_size, embedding_size])
    pos_tokens = paddle.transpose(pos_tokens, (0, 3, 1, 2))
    pos_tokens = nn.functional.interpolate(
        pos_tokens,
        size=(new_patch_size, new_patch_size),
        mode='bicubic',
        align_corners=False)

    pos_tokens = paddle.transpose(pos_tokens, (0, 2, 3, 1))
    pos_tokens = paddle.flatten(pos_tokens, 1, 2)
    pos_tokens = paddle.squeeze(pos_tokens, axis=0)

    new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=0)
    new_pos_embed = paddle.unsqueeze(new_pos_embed, axis=-1)
    return new_pos_embed


class ResidualAttentionBlock(nn.Layer):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: paddle.Tensor=None):
        super().__init__()

        self.attn = nn.MultiHeadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(("c_fc", nn.Linear(d_model, d_model * 4)),
                                 ("gelu", QuickGELU()), ("c_proj", nn.Linear(
                                     d_model * 4, d_model)))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: paddle.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=self.attn_mask)[0]

    def forward(self, x: paddle.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Layer):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: paddle.Tensor=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: paddle.Tensor):
        return self.resblocks(x)


### pretrain model similar with clip
class CVLP(nn.Layer):
    def __init__(
            self,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int,
            model_type="",
            pretrained_clip=None, ):
        super().__init__()

        self.context_length = context_length
        self.model_type = model_type

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            model_name="CLIP",
            img_size=image_resolution,
            patch_size=vision_patch_size,
            depth=vision_layers,
            num_heads=vision_heads,
            embed_dim=embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))))
        self.add_parameter("positional_embedding", self.positional_embedding)
        self.ln_final = nn.LayerNorm(transformer_width)

        self.text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))))
        self.proj = self.create_parameter(
            shape=(embed_dim, ),
            default_initializer=Assign(embed_dim**-0.5 * paddle.randn((
                (embed_dim, embed_dim)))))
        self.logit_scale = self.create_parameter(
            shape=(1, ), default_initializer=Assign(paddle.ones([1])))

        self.initialize_parameters(pretrained_clip)

    def initialize_parameters(self, pretrained_clip):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:

            normal_ = Normal(std=attn_std)
            normal_(block.attn.q_proj.weight)
            normal_(block.attn.k_proj.weight)
            normal_(block.attn.v_proj.weight)

            Normal(std=proj_std)(block.attn.out_proj.weight)
            Normal(std=fc_std)(block.mlp.c_fc.weight)
            Normal(std=proj_std)(block.mlp.c_proj.weight)

        if self.text_projection is not None:
            Normal(std=self.transformer.width**-0.5)(self.text_projection)
        if pretrained_clip is not None:
            if self.model_type == "vit":
                pretrained_state_dict = load_dygraph_pretrain_from_url_state_dict(
                    MODEL_URLS["CLIP_vit_base_patch16_224"])
            else:
                pretrained_state_dict = paddle.load(pretrained_clip)
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in pretrained_state_dict:
                    del pretrained_state_dict[key]
            if isinstance(self.visual, VisionTransformer):
                num_extra_tokens = 1
                new_size = int((self.visual.pos_embed.shape[1] -
                                num_extra_tokens)**0.5)
                new_pos_embed = interpolate_pos_embed(
                    pretrained_state_dict['pos_embed'],
                    new_size,
                    num_extra_tokens=num_extra_tokens)
                pretrained_state_dict['pos_embed'] = new_pos_embed

            info = self.set_state_dict(pretrained_state_dict)
            print('loaded pretrained clip.')
            print(info)

    def build_attention_mask(self):
        mask = paddle.empty((self.context_length, self.context_length))
        mask = paddle.full_like(mask, float("-inf"))
        mask = paddle.triu(mask, diagonal=1)
        return mask

    @property
    def dtype(self):
        return paddle.float32

    def encode_image(self, image) -> paddle.Tensor:
        image = paddle.cast(image, self.dtype)
        image = self.visual(image)
        image = self.ln_post(image[:, 0, :])
        if self.proj is not None:
            image = image @self.proj
        return image

    def encode_text(self, text) -> paddle.Tensor:
        text = paddle.cast(text, paddle.int32)
        x = paddle.cast(self.token_embedding(text), self.dtype)

        x = x + paddle.cast(self.positional_embedding, self.dtype)

        x = self.transformer(x)

        x = paddle.cast(self.ln_final(x), self.dtype)

        _buff = paddle.argmax(text, axis=1)
        x = x[paddle.arange(x.shape[0]), _buff] @self.text_projection

        return x

    def forward(self, x):
        image, text = x
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(
            axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(
            axis=-1, keepdim=True)

        logit_scale = paddle.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @text_features.t()
        logits_per_text = logits_per_image.transpose((0, 1))

        return [image.detach(), text.detach()], (logits_per_image,
                                                 logits_per_text)


def CVLP_vit16(**kwargs):
    model_type = "vit"
    pretrained_clip = kwargs[
        'pretrained_clip'] if "pretrained_clip" in kwargs.keys() else None
    model = CVLP(
        embed_dim=768,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        context_length=kwargs['context_length'] + 2,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        model_type=model_type,
        pretrained_clip=pretrained_clip)

    return model


#finetune model base on cvlp
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.norm1q = nn.LayerNorm(dim)
        self.norm1k = nn.LayerNorm(dim)

        self.wq = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self,
                qx: paddle.Tensor,
                kx: paddle.Tensor,
                key_padding_mask: paddle.Tensor=None):
        assert qx.shape[-1] == kx.shape[-1] and qx.shape[1] == 1
        Bq, _, C = qx.shape
        Bk, Nk, _ = kx.shape
        q = self.wq(self.norm1q(qx))
        q = paddle.reshape(q, (Bq, 1, self.num_heads, C // self.num_heads))
        q = paddle.transpose(q, (0, 2, 1, 3))

        k = self.wq(self.norm1k(kx))
        k = paddle.reshape(k, (Bk, Nk, self.num_heads, C // self.num_heads))
        k = paddle.transpose(k, (0, 2, 1, 3))
        v = paddle.unsqueeze(kx, axis=1)
        attn = paddle.einsum('qhoc,khnc->qkhn', q, k) * self.scale
        if key_padding_mask is not None:
            attn = masked_fill(
                attn,
                paddle.unsqueeze(
                    paddle.unsqueeze(
                        key_padding_mask, axis=0), axis=2),
                float('-inf'))
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.einsum('khnc,qkhn->qkhc', v, attn)
        x = paddle.reshape(x, (Bq, Bk, C))

        return x


class TextBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 op_type='two_branch',
                 num_classes=0,
                 use_constant_norm=False,
                 v_detach=False):
        super().__init__()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.op_type = op_type
        self.use_constant_norm = use_constant_norm
        self.v_detach = v_detach
        if self.op_type == 'concat':
            self.fc = nn.Linear(in_features=dim * 2, out_features=1, bias=True)
        elif self.op_type == 'add':
            self.fc = nn.Linear(in_features=dim, out_features=1, bias=True)
        elif self.op_type == 'cosine':
            self.fc = None
        elif self.op_type == 'two_branch':
            self.cos = nn.CosineSimilarity(axis=2, eps=1e-6)
            self.visual_fc = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.ReLU(), nn.Linear(4 * dim, num_classes))
        else:
            self.fc = None

    def forward(self,
                qx: paddle.Tensor,
                kx: paddle.Tensor,
                key_padding_mask: paddle.Tensor=None,
                logit_scale=None):
        v = self.attn(qx, kx, key_padding_mask=key_padding_mask)
        if self.op_type == 'concat':
            x = paddle.expand(qx, (qx.shape[0], kx.shape[0], qx.shape[-1]))
            x = paddle.concat((x, v), axis=-1)
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'cosine':
            if logit_scale is not None:
                qx_ = F.normalize(qx, p=2, axis=-1)
                if self.v_detach:
                    v_buff = paddle.linalg.norm(
                        v, axis=-1, keepdim=True).detach()
                    v_ = v / v_buff
                else:
                    v_ = F.normalize(v, p=2, axis=-1)
                x = paddle.einsum('qkc,qoc->qk', v_,
                                  qx_) * paddle.exp(logit_scale)
            else:
                x = paddle.einsum('qkc,qoc->qk', v, qx)
        elif self.op_type == 'add':
            x = paddle.expand(qx, (qx.shape[0], kx.shape[0], qx.shape[-1]))
            x = x + v
            x = self.fc(x)  # [Bq, Bk, 1]
        elif self.op_type == 'two_branch':
            x1 = self.visual_fc(paddle.squeeze(qx, axis=1))

            if logit_scale is not None:
                if self.use_constant_norm:
                    qx_ = F.normalize(qx, p=2, axis=-1)
                    v_ = v / 21.1578
                    x2 = paddle.einsum('qkc,qoc->qk', v_,
                                       qx_) * paddle.exp(logit_scale)
                else:
                    qx_ = F.normalize(qx, p=2, axis=-1)
                    if self.v_detach:
                        v_buff = paddle.linalg.norm(
                            v, axis=-1, keepdim=True).detach()
                        v_ = v / v_buff
                    else:
                        v_ = F.normalize(v, p=2, axis=-1)
                    x2 = paddle.einsum('qkc,qoc->qk', v_,
                                       qx_) * paddle.exp(logit_scale)
            else:
                x2 = paddle.einsum('qkc,qoc->qk', v, qx)

            return x1, x2

        return paddle.squeeze(x, axis=-1)


class LGR(nn.Layer):
    def __init__(
            self,
            num_classes: int,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            sent_length: int,
            attn_heads: int,
            sent_idxs=None,
            op_type="two_branch",
            use_norm=False,
            use_constant_norm=False,
            v_detach=False,
            img_grad=True,
            attn_grad=True,
            text_tokens=None,
            select_sent=None,
            sent_offset=0, ):
        super().__init__()
        self.num_classes = num_classes
        self.text_token = text_tokens
        self.sent_offset = sent_offset
        self.sent_length = sent_length
        self.sent_idxs = sent_idxs
        self.select_sent = select_sent
        self.cvlp = CVLP_vit16(pretrained_clip="vit", context_length=75)
        self.proj = self.create_parameter(
            shape=(embed_dim, ),
            default_initializer=Assign(embed_dim**-0.5 * paddle.randn((
                (embed_dim, embed_dim)))))
        self.use_norm = use_norm
        self.img_grad = img_grad
        self.attn_grad = attn_grad
        self.ln_post = nn.LayerNorm(embed_dim)
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            model_name="CLIP",
            img_size=image_resolution,
            patch_size=vision_patch_size,
            depth=vision_layers,
            num_heads=vision_heads,
            embed_dim=embed_dim)

        if op_type is None:
            print("do not use text features")
            self.text_embeddings = None
            self.text_block = None
            self.text_padding_mask = None
            self.fc = nn.Linear(embed_dim, num_classes)
        else:
            self.fc = None
            if self.use_norm:
                self.logit_scale = self.create_parameter(
                    (1, ),
                    default_initializer=Assign(
                        paddle.ones([1]) * np.log(1 / 0.07)))
                self.logit_scale.stop_gradient = True
            else:
                self.logit_scale = None
            self.text_embeddings = self.create_parameter(
                (self.num_classes, self.sent_length, embed_dim))

            self.text_block = TextBlock(
                dim=embed_dim,
                num_heads=attn_heads,
                qkv_bias=False,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                op_type=op_type,
                num_classes=num_classes,
                use_constant_norm=use_constant_norm,
                v_detach=v_detach)
            self.text_padding_mask = self.build_key_padding_mask(
                paddle.to_tensor(self.sent_idxs))

        if self.img_grad is False:
            print('freeze visual norm')
            for m in self.visual.parameters():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()
        if self.attn_grad is False:
            print('freeze attn norm')
            for m in self.text_block.attn.parameters():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                if isinstance(m, nn.BatchNorm2D):
                    m.eval()
        self.initialize_parameters()

    @property
    def dtype(self):
        return paddle.float32

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            TruncatedNormal(std=0.02)(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                Constant(0.0)(m.bias)

        elif isinstance(m, nn.LayerNorm):
            Constant(0.0)(m.bias)
            Constant(1.0)(m.weight)
        if self.text_block is not None:
            Assign(paddle.eye(self.text_block.attn.wq.weight.shape[0]))(
                self.text_block.attn.wq.weight)
            Assign(paddle.eye(self.text_block.attn.wk.weight.shape[0]))(
                self.text_block.attn.wk.weight)

    def initialize_parameters(self):
        self.apply(self._init_weights)

    def build_key_padding_mask(self, idxs: paddle.Tensor):
        mask = paddle.arange(0, self.sent_length)
        mask = paddle.cast(mask, idxs.dtype)
        mask = paddle.unsqueeze(mask, axis=0)
        mask = paddle.expand(mask, (idxs.shape[0], self.sent_length))
        mask = paddle.greater_than(mask, paddle.unsqueeze(idxs, axis=1) - 1)
        return mask

    @paddle.no_grad()
    def load_pretrained_model(self,
                              vis_backbone_path=None,
                              img_grad=True,
                              attn_grad=True):
        # load image part
        assert vis_backbone_path is not None
        self._load_vis_backbone(vis_backbone_path)
        self.visual.stop_gradient = not img_grad
        self.text_block.attn.stop_gradient = not attn_grad

        #load text part
        text_emb = []
        for clip_text in self.text_token:
            text_emb.append(self.cvlp.encode_text(clip_text).detach())
        text_emb = paddle.concat(text_emb)
        split_text_embeddings = paddle.split(text_emb, self.sent_idxs)
        split_text_embeddings = [
            s[self.sent_offset:self.sent_length + self.sent_offset, :]
            for s in split_text_embeddings
        ]
        split_text_embeddings = paddle.to_tensor(split_text_embeddings)
        split_text_embeddings = paddle.transpose(split_text_embeddings, (0, 1,
                                                                         2))
        self.text_embeddings.set_value(split_text_embeddings)

    def _load_vis_backbone(self, vis_backbone_path):
        assert osp.exists(vis_backbone_path)
        pretrained_state_dict = paddle.load(vis_backbone_path)
        self.cvlp.set_state_dict(pretrained_state_dict)
        self.cvlp.eval()
        self.ln_post.set_state_dict(self.cvlp.ln_post.state_dict())
        self.proj = self.cvlp.proj

        if isinstance(self.visual, VisionTransformer):
            num_extra_tokens = 1
            new_size = int((self.visual.pos_embed.shape[1] - num_extra_tokens)
                           **0.5)
            new_pos_embed = interpolate_pos_embed(
                pretrained_state_dict['visual.pos_embed'],
                new_size,
                num_extra_tokens=num_extra_tokens)
            pretrained_state_dict['visual.pos_embed'] = new_pos_embed
        if self.use_norm:
            vis_state_dict = {
                k: v
                for k, v in pretrained_state_dict.items()
                if k.startswith("visual") or k.startswith('logit_scale')
            }
        else:
            vis_state_dict = {
                k: v
                for k, v in pretrained_state_dict.items()
                if k.startswith("visual")
            }

        info = self.set_state_dict(vis_state_dict)
        print('pretrained visual backbone loaded')
        print(info)

    def encode_image(self, image) -> paddle.Tensor:
        self.visual.eval()
        x = self.visual(paddle.cast(image, self.dtype))
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @self.proj
        return x

    def forward(self, x):
        x = self.encode_image(x)
        if self.text_block is not None:

            x = self.text_block(
                paddle.unsqueeze(
                    x, axis=1),
                paddle.cast(self.text_embeddings, x.dtype),
                key_padding_mask=self.text_padding_mask,
                logit_scale=self.logit_scale)
        else:
            x = self.fc(x)
        return x


def LGR_vit16(pretrained=False, **kwargs):
    cache_root = kwargs["cache_root"]
    clip_token_path = os.path.join(cache_root, 'IMNET_LT_text_tokens.pkl')
    assert os.path.exists(clip_token_path)
    text_tokens = paddle.load(clip_token_path)
    sent_idxs = [len(sents) for sents in text_tokens]

    model = LGR(num_classes=kwargs['class_num'],
                embed_dim=768,
                image_resolution=224,
                vision_layers=12,
                vision_width=768,
                vision_patch_size=16,
                sent_length=kwargs['sent_length'],
                attn_heads=1,
                use_norm=True,
                img_grad=False,
                sent_idxs=sent_idxs,
                text_tokens=text_tokens,
                select_sent="val")

    model.load_pretrained_model(
        vis_backbone_path=kwargs['pretrain_model_path'], img_grad=False)

    return model
