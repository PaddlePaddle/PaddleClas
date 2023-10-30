import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.initializer import Assign, Normal, Constant
from ...legendary_models.swin_transformer import masked_fill
import math
from typing import Optional


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * nn.functional.sigmoid(1.702 * x)


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 output_dim=None,
                 qkv_bias=True,
                 scaled_cosine=False,
                 scale_heads=False,
                 logit_scale_max=math.log(1. / 0.01),
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        in_proj_weight = self.create_parameter(
            shape=(dim * 3, dim),
            default_initializer=Assign(
                paddle.randn((dim * 3, dim)) * self.scale))
        self.add_parameter("in_proj_weight", in_proj_weight)

        in_proj_bias = self.create_parameter(
            shape=(dim * 3, ),
            default_initializer=Assign(paddle.zeros((dim * 3))))
        self.add_parameter("in_proj_bias", in_proj_bias)

        if self.scaled_cosine:
            logit_scale = self.create_parameter(
                shape=(num_heads, 1, 1),
                default_initializer=Assign(
                    paddle.log(10 * paddle.ones((num_heads, 1, 1)))))
            self.add_parameter("logit_scale", logit_scale)
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            head_scale = self.create_parameter(
                shape=(num_heads, 1, 1),
                default_initializer=Assign(paddle.ones((num_heads, 1, 1))))
            self.add_parameter("head_scale", head_scale)
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, output_dim or dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[paddle.Tensor]=None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight.T, self.in_proj_bias).chunk(
            3, axis=-1)
        q = q.reshape((L, N * self.num_heads, -1)).transpose((1, 0, 2))
        k = k.reshape((L, N * self.num_heads, -1)).transpose((1, 0, 2))
        v = v.reshape((L, N * self.num_heads, -1)).transpose((1, 0, 2))

        q = q * self.scale
        attn = paddle.bmm(q, k.transpose((0, 2, 1)))

        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                new_attn_mask = paddle.zeros_like(attn_mask, dtype=q.dtype)
                masked_fill(new_attn_mask, attn_mask, float("-inf"))
                #new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.bmm(attn, v)
        x = x.transpose((1, 0, 2))
        x = x.reshape((L, N, C))
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(
            stride, exclusive=False) if stride > 1 else Identity()

        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(
                    stride, exclusive=False)), ("0", nn.Conv2D(
                        inplanes,
                        planes * self.expansion,
                        1,
                        stride=1,
                        bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion)))

    def forward(self, x):
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


class AttentionPool2D(nn.Layer):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        positional_embedding = self.create_parameter(
            shape=(spacial_dim**2 + 1, embed_dim),
            default_initializer=Assign(
                paddle.randn((spacial_dim**2 + 1, embed_dim)) / embed_dim
                **0.5))
        self.add_parameter("positional_embedding", positional_embedding)

        self.attn = MultiHeadAttention(embed_dim, num_heads, output_dim)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1],
                       x.shape[2] * x.shape[3])).transpose((2, 0, 1))
        x = paddle.concat([x.mean(axis=0, keepdim=True), x], axis=0)
        x = x + self.positional_embedding.unsqueeze(1)
        x = x.transpose((1, 0, 2))
        x = self.attn(query=x, key=x, value=x)
        x = x.transpose((1, 0, 2))
        return x[0]


class ModifiedResNet(nn.Layer):
    def __init__(self,
                 layers,
                 output_dim,
                 heads,
                 input_resolution=224,
                 width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2D(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)

        self.conv2 = nn.Conv2D(
            width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)

        self.conv3 = nn.Conv2D(
            width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)

        self.avgpool = nn.AvgPool2D(2, exclusive=False)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2D(input_resolution // 32, embed_dim,
                                        heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def stem(self, x):
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                         (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))

        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class ResidualAttentionBlock(nn.Layer):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(("c_fc", nn.Linear(d_model, d_model * 4)),
                                 ("gelu", QuickGELU()),
                                 ("c_proj", nn.Linear(d_model * 4, d_model)))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, attn_mask=self.attn_mask)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Layer):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x):
        return self.resblocks(x)


class VisualTransformer(nn.Layer):
    def __init__(self, input_resolution, patch_size, width, layers, heads,
                 output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=False)

        scale = width**-0.5

        class_embedding = self.create_parameter(
            shape=(width, ),
            default_initializer=Assign(scale * paddle.randn((width, ))))
        self.add_parameter("class_embedding", class_embedding)

        positional_embedding = self.create_parameter(
            shape=(width, ),
            default_initializer=Assign(scale * paddle.randn((
                (input_resolution // patch_size)**2 + 1, width))))
        self.add_parameter("positional_embedding", positional_embedding)

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)

        proj = self.create_parameter(
            shape=(width, ),
            default_initializer=Assign(scale * paddle.randn((
                (width, output_dim)))))
        self.add_parameter("proj", proj)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        zeros = paddle.zeros((x.shape[0], 1, x.shape[-1]), dtype='float32')
        x = paddle.concat([self.class_embedding + zeros, x], axis=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.transpose([1, 0, 2])
        x = self.transformer(x)
        x = x.transpose([1, 0, 2])
        pooled, tokens = x[:, 0], x[:, 1:]
        x = self.ln_post(pooled)

        if self.proj is not None:
            x = x @self.proj

        return x


class CLIP(nn.Layer):
    def __init__(
            self,
            embed_dim,
            # vision
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            vision_head,
            # text
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width)
        else:
            vision_heads = vision_width // vision_head
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))))
        self.add_parameter("positional_embedding", positional_embedding)

        self.ln_final = nn.LayerNorm(transformer_width)

        text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))))
        self.add_parameter("text_projection", text_projection)

        logit_scale = self.create_parameter(
            shape=(1, ), default_initializer=Assign(paddle.ones([1])))
        self.add_parameter("logit_scale", logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.embed_dim**-0.5
                normal_(self.visual.attnpool.attn.in_proj_weight.weight)
                normal_(self.visual.attnpool.attn.in_proj_bias.weight)
                normal_(self.visual.attnpool.attn.out_proj.weight)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        Constant(value=0.0)(param)

        proj_std = (self.transformer.width ** -0.5) * \
            ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5

        for resblock in self.transformer.resblocks:
            normal_ = Normal(std=attn_std)
            normal_(resblock.attn.in_proj_weight)
            normal_(resblock.attn.in_proj_bias)

            Normal(std=proj_std)(resblock.attn.out_proj.weight)
            Normal(std=fc_std)(resblock.mlp.c_fc.weight)
            Normal(std=proj_std)(resblock.mlp.c_proj.weight)

        if self.text_projection is not None:
            Normal(std=self.transformer.width**-0.5)(self.text_projection)

    def build_attention_mask(self):
        mask = paddle.full((self.context_length, self.context_length),
                           float("-inf"))
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        select = []
        index = zip(paddle.arange(x.shape[0]).numpy(),
                    text.argmax(axis=-1).numpy())
        for i, j in index:
            select.append(x[int(i), int(j)])

        x = paddle.stack(select) @self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @text_features.t()
        logits_per_text = logit_scale * text_features @image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class LaCLIP(nn.Layer):
    def __init__(
            self,
            embed_dim,
            # vision
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            vision_head,
            # text
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width)
        else:
            vision_heads = vision_width // vision_head
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))))
        self.add_parameter("positional_embedding", positional_embedding)

        self.ln_final = nn.LayerNorm(transformer_width)

        image_projection = self.create_parameter(
            shape=(vision_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((vision_width, embed_dim))))
        self.add_parameter("image_projection", image_projection)
        text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))))
        self.add_parameter("text_projection", text_projection)

        logit_scale = self.create_parameter(
            shape=(1, ), default_initializer=Assign(paddle.ones([1])))
        self.add_parameter("logit_scale", logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.embed_dim**-0.5
                normal_(self.visual.attnpool.attn.in_proj_weight)
                normal_(self.visual.attnpool.attn.in_proj_bias)
                normal_(self.visual.attnpool.attn.out_proj.weight)

            for resnet_block in [
                    self.visual.layer1, self.visual.layer2, self.visual.layer3,
                    self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        Constant(value=0.0)(param)

        proj_std = (self.transformer.width ** -0.5) * \
            ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5

        for resblock in self.transformer.resblocks:
            normal_ = Normal(std=attn_std)
            normal_(resblock.attn.in_proj_weight)
            normal_(resblock.attn.in_proj_bias)

            Normal(std=proj_std)(resblock.attn.out_proj.weight)
            Normal(std=fc_std)(resblock.mlp.c_fc.weight)
            Normal(std=proj_std)(resblock.mlp.c_proj.weight)

        if self.text_projection is not None:
            Normal(std=self.transformer.width**-0.5)(self.text_projection)

    def build_attention_mask(self):
        mask = paddle.full((self.context_length, self.context_length),
                           float("-inf"))
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        x = x @self.image_projection
        return x

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        select = []
        index = zip(paddle.arange(x.shape[0]).numpy(),
                    text.argmax(axis=-1).numpy())
        for i, j in index:
            select.append(x[int(i), int(j)])

        x = paddle.stack(select) @self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @text_features.t()
        logits_per_text = logit_scale * text_features @image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
