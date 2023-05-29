from typing import Tuple, Union
import paddle.distributed as dist
import numpy as np
import paddle
from paddle.nn import functional as F
from paddle import nn
from .utils import interpolate_pos_embed,MODEL_URLS,load_dygraph_pretrain_from_url
from .utils import GatherLayer
from paddle.nn.initializer import Assign, Normal, Constant

__all__ = ["CVLP_r50", "CVLP_vit16"]



class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * nn.functional.sigmoid(1.702 * x)
    
class MultiHeadAttention(nn.MultiHeadAttention):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 output_dim=None):
        super(MultiHeadAttention, self).__init__(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, output_dim or embed_dim)

class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1,bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(stride)),
                ("0", nn.Conv2D(inplanes, planes * self.expansion, 1, stride=1, bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion))
            )

    def forward(self, x: paddle.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Layer):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = self.create_parameter(
            shape=(spacial_dim ** 2 + 1, embed_dim),
            default_initializer=Assign(
                paddle.randn((spacial_dim ** 2 + 1, embed_dim)) /
                embed_dim ** 0.5
            )
        )
        self.add_parameter("positional_embedding", self.positional_embedding)

        self.attn = MultiHeadAttention(embed_dim, num_heads, output_dim)

    def forward(self, x:paddle.Tensor):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] *
                       x.shape[3])).transpose((2, 0, 1))
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)
        x = x + self.positional_embedding.unsqueeze(1)
        x = x.transpose((1, 0, 2))
        x = self.attn(query=x, key=x, value=x)
        x = x.transpose((1, 0, 2))
        return x[0]


class ModifiedResNet(nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.avgpool = nn.AvgPool2D(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        #x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
    


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        #ret = super().forward(x.type(torch.float32))
        ret = super().forward(paddle.cast(x,paddle.float32))
        return paddle.cast(ret,orig_type)



class ResidualAttentionBlock(nn.Layer):
    def __init__(self, d_model: int, n_head: int, attn_mask: paddle.Tensor = None):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: paddle.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=self.attn_mask)[0]

    def forward(self, x: paddle.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Layer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: paddle.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: paddle.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Layer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False)

        scale = width ** -0.5
        self.class_embedding = scale * self.create_parameter(
            shape=(width,),
            default_initializer=Assign(
                scale * paddle.randn((width,))
            )
        )
        self.positional_embedding = self.create_parameter(
            shape=(width,),
            default_initializer=Assign(
                scale *
                paddle.randn(
                    ((input_resolution // patch_size) ** 2 + 1, width))
            )
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = self.create_parameter(
            shape=(width,),
            default_initializer=Assign(
                scale * paddle.randn(((width, output_dim)))
            )
        )

    def forward(self, x: paddle.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        #x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = paddle.reshape(x,(x.shape[0], x.shape[1], -1))
        x = paddle.transpose(x,(0, 2, 1))
        #x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = paddle.concat([paddle.cast(self.class_embedding,x.dtype) + paddle.zeros((x.shaoe[0],1,x.shape[-1]),dtype=x.dtype),x],axis=1)
        #x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #x = x + self.positional_embedding.to(x.dtype)
        x = x + paddle.cast(self.positional_embedding,x.dtype)
        x = self.ln_pre(x)

        #x = x.permute(1, 0, 2)  # NLD -> LND
        x = paddle.transpose(x,(1,0,2))# NLD -> LND
        x = self.transformer(x)
        x = paddle.transpose(x,(1,0,2))# LND -> NLD
        #x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CVLP(nn.Layer):
    def __init__(self,
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
                 model_type = "",
                 pretrained_clip=None,
                 ):
        super().__init__()

        self.context_length = context_length
        self.model_type = model_type

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
        self.embed_dim = embed_dim
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))
            )
        )
        self.add_parameter("positional_embedding", self.positional_embedding)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))
            )
        )
        self.add_parameter("text_projection", self.text_projection)
        self.logit_scale = self.create_parameter(
            shape=(1,),
            default_initializer=Assign(paddle.ones([1]))
        )
        self.add_parameter("logit_scale", self.logit_scale)
        

        self.initialize_parameters(pretrained_clip)

    def initialize_parameters(self, pretrained_clip):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)
        #nn.init.normal_(self.token_embedding.weight, std=0.02)
        #nn.init.normal_(self.positional_embedding, std=0.01)
        #nn.initializer.normal()
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.embed_dim ** -0.5
                normal_ = Normal(std=std)
                normal_(self.visual.attnpool.attn.q_proj.weight)
                normal_(self.visual.attnpool.attn.k_proj.weight)
                normal_(self.visual.attnpool.attn.v_proj.weight)
                normal_(self.visual.attnpool.attn.out_proj.weight)
                #nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                #nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                #nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                #nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        #nn.init.zeros_(param)
                         Constant(value=0.0)(param)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:

            normal_ = Normal(std = attn_std)
            normal_(block.attn.q_proj.weight)
            normal_(block.attn.k_proj.weight)
            normal_(block.attn.v_proj.weight)


            Normal(std=proj_std)(block.attn.out_proj.weight)
            Normal(std=fc_std)(block.mlp.c_fc.weight)
            Normal(std=proj_std)(block.mlp.c_proj.weight)
            #nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            #nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            Normal(std=self.transformer.width ** -0.5)(self.text_projection)
        if pretrained_clip is not None:
            if self.model_type == "vit":
                pretrained_state_dict = load_dygraph_pretrain_from_url(MODEL_URLS["CLIP_vit_base_patch16_224"],False)
            else:
                pretrained_state_dict = paddle.load(pretrained_clip)
            #pretrained_state_dict = model.state_dict()
            for key in ["input_resolution", "context_length", "vocab_size"]:
                if key in pretrained_state_dict:
                    del pretrained_state_dict[key]
            if isinstance(self.visual, VisionTransformer):
                num_extra_tokens = 1
                new_size = int((self.visual.positional_embedding.shape[0] - num_extra_tokens) ** 0.5)
                new_pos_embed = interpolate_pos_embed(pretrained_state_dict['visual.positional_embedding'], 
                                                        new_size, num_extra_tokens=num_extra_tokens)
                pretrained_state_dict['visual.positional_embedding'] = new_pos_embed

            info = self.set_state_dict(pretrained_state_dict)
            print('loaded pretrained clip.')
            print(info)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = paddle.empty((self.context_length, self.context_length))
        mask = paddle.full_like(mask,float("-inf"))
        #mask.fill_(float("-inf"))
        mask = paddle.triu(mask,diagonal=1)
        #mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image) -> paddle.Tensor:
        image = paddle.cast(image,self.dtype)
        return self.visual(image)

    def encode_text(self, text) -> paddle.Tensor:
        x = paddle.cast(self.token_embedding(text),self.dtype)
        #x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + paddle.cast(self.positional_embedding,self.dtype)
        #x = paddle.transpose(x,(1,0,2))
        #x = x.permute(1, 0, 2)  # NLD -> LND
        
        x = self.transformer(x)

        #x = paddle.transpose(x,(1,0,2))
        #x = x.permute(1, 0, 2)  # LND -> NLD
        x = paddle.cast(self.ln_final(x),self.dtype)
        #x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        _buff = paddle.argmax(text,axis=1)
        x = x[paddle.arange(x.shape[0]), _buff] @ self.text_projection

        return x

    def forward(self, x):
        image, text = x
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(axis=-1, keepdim=True)

        if dist.is_initialized():
            image_features = paddle.concat(GatherLayer.apply(image_features), 0)
            text_features = paddle.concat(GatherLayer.apply(text_features), 0)

        # cosine similarity as logits
        logit_scale = paddle.exp(self.logit_scale) 
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.transpose((0, 1))

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



def CVLP_r50(pretrained=False,args=None):
    args = args

    model = CVLP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=args.context_length + 2,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        args=args,
    )

    return model



def CVLP_vit16(pretrained=False, **kwargs):
    model_type = "vit"
    pretrained_clip = kwargs['pretrained_clip'] if "pretrained_clip" in kwargs.keys() else None
    model = CVLP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        context_length=kwargs['context_length'] + 2,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        model_type = model_type,
        pretrained_clip = pretrained_clip
    )

    return model