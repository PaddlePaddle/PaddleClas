import paddle
import paddle.nn as nn
from paddle.nn.initializer import Assign, Normal, Constant, TruncatedNormal

trunc_normal_ = TruncatedNormal(std=0.02)
constant_0 = Constant(0)
constant_1 = Constant(1)


class VisionTransformer(nn.Layer):
    def __init__(self,
                 input_size=224,
                 patch_size=32,
                 in_channels=3,
                 dim=768,
                 embedding_size=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 drop_path_rate=0.0,
                 using_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbedding(
            input_size,
            patch_size,
            in_channels,
            dim, )
        self.pos_embed = paddle.zeros((1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.LayerList([
            Block(dim, num_heads, mlp_ratio, dpr[i],
                  self.patch_embed.num_patches, using_checkpoint)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        self.feature = nn.Sequential(
            nn.Linear(
                dim * self.patch_embed.num_patches, dim, bias_attr=False),
            nn.BatchNorm1D(
                dim, epsilon=2e-5),
            nn.Linear(
                dim, embedding_size, bias_attr=False),
            nn.BatchNorm1D(
                embedding_size, epsilon=2e-5))

        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_0(m.bias)
        elif isinstance(m, nn.LayerNorm):
            constant_0(m.bias)
            constant_1(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for func in self.blocks:
            x = func(x)
        x = self.norm(x)
        return paddle.reshape(x, (B, self.patch_embed.num_patches * self.dim))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.feature(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape((B, L, 3, self.num_heads, D //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @k.transpose((0, 1, 3, 2))) * self.scale
        attn = nn.functional.softmax(attn, axis=1)
        x = (attn @v).transpose((0, 2, 1, 3)).reshape((B, L, D))
        x = self.proj(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int=4,
                 drop_path: float=0.0,
                 patch_n: int=32,
                 using_checkpoint=False):
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) *
                             patch_n * 2) / (1000**3)

    def forward_impl(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        return self.forward_impl(x)


class PatchEmbedding(nn.Layer):
    def __init__(self,
                 input_size=224,
                 patch_size=32,
                 in_channels: int=3,
                 dim: int=768):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2D(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


def build_model(name="ViT-L/14@336px"):
    if name == "ViT-B/32":
        model = VisionTransformer(
            input_size=224,
            patch_size=32,
            in_channels=3,
            dim=768,
            embedding_size=512,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=True)
    elif name == "ViT-B/16":
        model = VisionTransformer(
            input_size=224,
            patch_size=16,
            in_channels=3,
            dim=768,
            embedding_size=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            using_checkpoint=True)
    elif name == "ViT-L/14":
        model = VisionTransformer(
            input_size=224,
            patch_size=14,
            in_channels=3,
            dim=1024,
            embedding_size=768,
            depth=24,
            num_heads=16,
            drop_path_rate=0.1,
            using_checkpoint=True)
    elif name == "ViT-L/14@336px":
        model = VisionTransformer(
            input_size=336,
            patch_size=14,
            in_channels=3,
            dim=1024,
            embedding_size=768,
            depth=24,
            num_heads=16,
            drop_path_rate=0.1,
            using_checkpoint=True)
    return model


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def load_model(name="ViT-L/14@336px"):
    if name == "ViT-B/32":
        return build_model(name)
    elif name == "ViT-B/16":
        return build_model(name)
    elif name == "ViT-L/14":
        return build_model(name)
    elif name == "ViT-L/14@336px":
        return build_model(name)
    else:
        raise
