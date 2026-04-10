# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ssl
import urllib
import math
import numpy as np
from typing import Optional, Tuple, Union, Dict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from ppcls.utils import load_dygraph_pretrain
    HAS_PPCLS = True
except ImportError:
    HAS_PPCLS = False
    load_dygraph_pretrain = None


MODEL_URLS = {
    "vit_base_patch32_siglip_256": "https://huggingface.co/timm/vit_base_patch32_siglip_256.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_224": "https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_256": "https://huggingface.co/timm/vit_base_patch16_siglip_256.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_384": "https://huggingface.co/timm/vit_base_patch16_siglip_384.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_512": "https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_256": "https://huggingface.co/timm/vit_large_patch16_siglip_256.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_384": "https://huggingface.co/timm/vit_large_patch16_siglip_384.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_512": "https://huggingface.co/timm/vit_large_patch16_siglip_512.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_224": "https://huggingface.co/timm/vit_so400m_patch14_siglip_224.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_378": "https://huggingface.co/timm/vit_so400m_patch14_siglip_378.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_384": "https://huggingface.co/timm/vit_so400m_patch14_siglip_384.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_256": "https://huggingface.co/timm/vit_so400m_patch16_siglip_256.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_384": "https://huggingface.co/timm/vit_so400m_patch16_siglip_384.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_512": "https://huggingface.co/timm/vit_so400m_patch16_siglip_512.v2_webli/resolve/main/model.safetensors",
    "vit_giantopt_patch16_siglip_256": "https://huggingface.co/timm/vit_giantopt_patch16_siglip_256.v2_webli/resolve/main/model.safetensors",
    "vit_giantopt_patch16_siglip_384": "https://huggingface.co/timm/vit_giantopt_patch16_siglip_384.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch32_siglip_gap_256": "https://huggingface.co/timm/vit_base_patch32_siglip_gap_256.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_gap_224": "https://huggingface.co/timm/vit_base_patch16_siglip_gap_224.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_gap_256": "https://huggingface.co/timm/vit_base_patch16_siglip_gap_256.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_gap_384": "https://huggingface.co/timm/vit_base_patch16_siglip_gap_384.v2_webli/resolve/main/model.safetensors",
    "vit_base_patch16_siglip_gap_512": "https://huggingface.co/timm/vit_base_patch16_siglip_gap_512.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_gap_256": "https://huggingface.co/timm/vit_large_patch16_siglip_gap_256.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_gap_384": "https://huggingface.co/timm/vit_large_patch16_siglip_gap_384.v2_webli/resolve/main/model.safetensors",
    "vit_large_patch16_siglip_gap_512": "https://huggingface.co/timm/vit_large_patch16_siglip_gap_512.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_gap_224": "https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_224.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_gap_378": "https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_378.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_gap_384": "https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_384.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_gap_448": "https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_448.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch14_siglip_gap_896": "https://huggingface.co/timm/vit_so400m_patch14_siglip_gap_896.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_gap_256": "https://huggingface.co/timm/vit_so400m_patch16_siglip_gap_256.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_gap_384": "https://huggingface.co/timm/vit_so400m_patch16_siglip_gap_384.v2_webli/resolve/main/model.safetensors",
    "vit_so400m_patch16_siglip_gap_512": "https://huggingface.co/timm/vit_so400m_patch16_siglip_gap_512.v2_webli/resolve/main/model.safetensors",
    "vit_giantopt_patch16_siglip_gap_256": "https://huggingface.co/timm/vit_giantopt_patch16_siglip_gap_256.v2_webli/resolve/main/model.safetensors",
    "vit_giantopt_patch16_siglip_gap_384": "https://huggingface.co/timm/vit_giantopt_patch16_siglip_gap_384.v2_webli/resolve/main/model.safetensors",
}


def download_weight(url, model_name=None, max_retries=3):
    """Download weight file from URL with SSL fix and retry mechanism"""
    cache_dir = os.path.expanduser("~/.cache/paddle/siglip_weights")
    
    url_path = url.split("huggingface.co/")[1] if "huggingface.co/" in url else url
    url_path = url_path.replace("/resolve/main/", "/")
    filepath = os.path.join(cache_dir, url_path)
    
    if os.path.exists(filepath):
        print(f"  Weight file already exists: {filepath}")
        return filepath
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"  Downloading weight from: {url}")
    print(f"  Saving to: {filepath}")
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}/{max_retries}...")
            
            try:
                import requests
                response = requests.get(url, stream=True, verify=False, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size * 100
                                print(f"\r  Downloading: {progress:.1f}%", end="", flush=True)
                
                print(f"\n  Download completed!")
                return filepath
                
            except ImportError:
                def _create_ssl_opener():
                    https_handler = urllib.request.HTTPSHandler(context=ssl_context)
                    opener = urllib.request.build_opener(https_handler)
                    return opener
                
                opener = _create_ssl_opener()
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, filepath)
                print(f"  Download completed!")
                return filepath
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Download failed: {e}")
                print(f"  Retrying in 2 seconds...")
                import time
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed to download weight after {max_retries} attempts: {e}")
    
    return filepath


def load_safetensors_weights(model, filepath):
    """Load weights from safetensors file"""
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors is required. Install with: pip install safetensors")
    
    state_dict = {}
    with safe_open(filepath, framework="np", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    model_state_dict = model.state_dict()
    
    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("visual."):
            new_key = key[7:]
        elif "encoder" in key:
            new_key = key.replace("encoder.", "")
        
        if new_key in model_state_dict:
            model_shape = model_state_dict[new_key].shape
            if value.shape != model_shape:
                if len(value.shape) == 2 and len(model_shape) == 2:
                    if value.shape[0] == model_shape[1] and value.shape[1] == model_shape[0]:
                        value = value.T
                elif len(value.shape) == 4 and len(model_shape) == 4:
                    if value.shape == model_shape[::-1]:
                        value = value.transpose(3, 2, 1, 0)
            converted_state_dict[new_key] = value
    
    missing_keys = []
    for key in model_state_dict.keys():
        if key not in converted_state_dict:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"  Warning: {len(missing_keys)} keys not found in pretrained weights")
        if len(missing_keys) <= 10:
            for key in missing_keys[:10]:
                print(f"    - {key}")
    
    model.set_state_dict(converted_state_dict)
    return model


def load_pretrained_weights(model, url):
    """Download and load pretrained weights"""
    weight_path = download_weight(url)
    return load_safetensors_weights(model, weight_path)


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        if load_dygraph_pretrain is not None:
            load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
        else:
            load_pretrained_weights(model, model_url)
    elif isinstance(pretrained, str):
        if load_dygraph_pretrain is not None:
            load_dygraph_pretrain(model, pretrained)
        else:
            load_safetensors_weights(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


class GELUTanh(nn.Layer):
    """GELU activation with tanh approximation"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate=True)


class AttentionPoolLatent(nn.Layer):
    """Attention pooling with learnable latent query"""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        expansion_ratio: int = 4,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.latent = self.create_parameter(
            shape=[1, 1, out_features],
            default_initializer=nn.initializer.Constant(0.0)
        )
        
        self.q = nn.Linear(out_features, out_features, bias_attr=qkv_bias)
        self.kv = nn.Linear(in_features, out_features * 2, bias_attr=qkv_bias)
        self.proj = nn.Linear(out_features, out_features)
        
        hidden_features = int(out_features * expansion_ratio)
        self.mlp = MLP(
            in_features=out_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        B, N, C = x.shape
        
        latent = self.latent.expand([B, -1, -1])
        
        q = self.q(latent).reshape([B, 1, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        kv = self.kv(x).reshape([B, N, 2, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        
        out = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, 1, -1])
        out = self.proj(out)
        
        out = out + self.mlp(self.norm(out))
        
        return out.squeeze(1)


def global_pool_nlc(x, pool_type: str = 'avg'):
    """Global pooling for NLC format tensors"""
    if pool_type == 'avg':
        return x.mean(axis=1)
    elif pool_type == 'max':
        return x.max(axis=1)
    elif pool_type == 'map':
        return x
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Layer):
    """Multi-head Self-Attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Layer):
    """MLP with GELU activation"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Layer = nn.GELU,
        drop: float = 0.0,
    ):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Layer):
    """Layer scale for transformer blocks"""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = self.create_parameter(
            shape=[dim],
            default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x):
        return x * self.gamma


class Block(nn.Layer):
    """Transformer Block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Layer = nn.GELU,
        init_values: float = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        if init_values is not None and init_values > 0:
            self.ls1 = LayerScale(dim, init_values)
            self.ls2 = LayerScale(dim, init_values)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SigLIPVisionTransformer(nn.Layer):
    """SigLIP Vision Transformer"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        class_token: bool = False,
        global_pool: str = 'map',
        act_layer: nn.Layer = GELUTanh,
        class_num: int = 1000,
    ):
        super().__init__()
        self.class_token = class_token
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        if class_token:
            self.cls_token = self.create_parameter(
                shape=[1, 1, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02)
            )
        
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches + (1 if class_token else 0), embed_dim],
            default_initializer=nn.initializer.TruncatedNormal(std=0.02)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                in_features=embed_dim,
                out_features=embed_dim,
            )
        
        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following timm's approach"""
        for name, m in self.named_sublayers():
            if isinstance(m, nn.Linear):
                nn.initializer.TruncatedNormal(std=0.02)(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(0)(m.bias)
            elif isinstance(m, nn.Conv2D):
                fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                bound = (1.0 / fan_in) ** 0.5
                nn.initializer.Uniform(-bound, bound)(m.weight)
                if m.bias is not None:
                    nn.initializer.Uniform(-bound, bound)(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        if self.class_token:
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if self.global_pool == 'map':
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x.mean(axis=1)
        elif self.global_pool == 'max':
            x = x.max(axis=1)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_base_patch32_siglip_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch32_siglip_256"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_224(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_224"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_256"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_384"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_512"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_256"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_384"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_512"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_224(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_224"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_378(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=378,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_378"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_384"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_256"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_384"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_512"], use_ssld=use_ssld)
    return model


def vit_giantopt_patch16_siglip_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_giantopt_patch16_siglip_256"], use_ssld=use_ssld)
    return model


def vit_giantopt_patch16_siglip_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_giantopt_patch16_siglip_384"], use_ssld=use_ssld)
    return model


def vit_base_patch32_siglip_gap_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch32_siglip_gap_256"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_gap_224(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_gap_224"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_gap_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_gap_256"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_gap_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_gap_384"], use_ssld=use_ssld)
    return model


def vit_base_patch16_siglip_gap_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_base_patch16_siglip_gap_512"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_gap_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_gap_256"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_gap_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_gap_384"], use_ssld=use_ssld)
    return model


def vit_large_patch16_siglip_gap_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_large_patch16_siglip_gap_512"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_gap_224(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_gap_224"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_gap_378(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=378,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_gap_378"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_gap_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_gap_384"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_gap_448(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=448,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_gap_448"], use_ssld=use_ssld)
    return model


def vit_so400m_patch14_siglip_gap_896(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=896,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=nn.GELU,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch14_siglip_gap_896"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_gap_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_gap_256"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_gap_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_gap_384"], use_ssld=use_ssld)
    return model


def vit_so400m_patch16_siglip_gap_512(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=512,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_so400m_patch16_siglip_gap_512"], use_ssld=use_ssld)
    return model


def vit_giantopt_patch16_siglip_gap_256(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=256,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_giantopt_patch16_siglip_gap_256"], use_ssld=use_ssld)
    return model


def vit_giantopt_patch16_siglip_gap_384(pretrained=False, class_num=1000, use_ssld=False, **kwargs):
    model = SigLIPVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        class_token=False,
        class_num=class_num,
        act_layer=GELUTanh,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["vit_giantopt_patch16_siglip_gap_384"], use_ssld=use_ssld)
    return model


MODEL_URLS["naflexvit_base_patch16_siglip"] = "https://huggingface.co/timm/naflexvit_base_patch16_siglip.v2_webli/resolve/main/model.safetensors"
MODEL_URLS["naflexvit_so400m_patch16_siglip"] = "https://huggingface.co/timm/naflexvit_so400m_patch16_siglip.v2_webli/resolve/main/model.safetensors"
MODEL_URLS["naflexvit_base_patch16_gap"] = "https://huggingface.co/timm/naflexvit_base_patch16_gap.e300_s576_in1k/resolve/main/model.safetensors"
MODEL_URLS["naflexvit_base_patch16_par_gap"] = "https://huggingface.co/timm/naflexvit_base_patch16_par_gap.e300_s576_in1k/resolve/main/model.safetensors"
MODEL_URLS["naflexvit_base_patch16_parfac_gap"] = "https://huggingface.co/timm/naflexvit_base_patch16_parfac_gap.e300_s576_in1k/resolve/main/model.safetensors"


class NaFlexEmbeds(nn.Layer):
    """NaFlex Embedding module with dynamic position embedding support"""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        pos_embed_grid_size: Tuple[int, int] = (16, 16),
        pos_drop_rate: float = 0.0,
        class_token: bool = False,
        reg_tokens: int = 0,
        pos_embed_type: str = 'learned',
        pos_embed_ar_preserving: bool = False,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pos_embed_grid_size = pos_embed_grid_size
        self.has_class_token = class_token
        self.num_reg_tokens = reg_tokens
        self.pos_embed_type = pos_embed_type
        self.pos_embed_ar_preserving = pos_embed_ar_preserving

        self.num_prefix_tokens = (1 if class_token else 0) + reg_tokens

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if class_token:
            self.cls_token = self.create_parameter(
                shape=[1, 1, embed_dim],
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.cls_token = None

        if reg_tokens > 0:
            self.reg_token = self.create_parameter(
                shape=[1, reg_tokens, embed_dim],
                default_initializer=nn.initializer.Constant(0.0)
            )
        else:
            self.reg_token = None

        h, w = pos_embed_grid_size
        if pos_embed_type == 'factorized':
            self.pos_embed_y = self.create_parameter(
                shape=[1, h, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02)
            )
            self.pos_embed_x = self.create_parameter(
                shape=[1, w, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02)
            )
            self.pos_embed = None
        elif pos_embed_type == 'learned':
            self.pos_embed = self.create_parameter(
                shape=[1, h, w, embed_dim],
                default_initializer=nn.initializer.TruncatedNormal(std=0.02)
            )
            self.pos_embed_y = None
            self.pos_embed_x = None
        else:
            self.pos_embed = None
            self.pos_embed_y = None
            self.pos_embed_x = None

        self.pos_drop = nn.Dropout(p=pos_drop_rate)

    def _apply_learned_pos_embed(self, x, grid_size):
        """Apply learned position embedding with interpolation"""
        B, N, C = x.shape
        target_h, target_w = grid_size
        orig_h, orig_w = self.pos_embed_grid_size

        if self.pos_embed_type == 'factorized':
            pos_embed_y = self.pos_embed_y
            pos_embed_x = self.pos_embed_x

            if target_h != orig_h:
                pos_embed_y_nchw = pos_embed_y.transpose([0, 2, 1]).unsqueeze(-1)
                pos_embed_y_interp = F.interpolate(
                    pos_embed_y_nchw,
                    size=target_h,
                    mode='linear',
                    align_corners=False,
                )
                pos_embed_y = pos_embed_y_interp.squeeze(-1).transpose([0, 2, 1])

            if target_w != orig_w:
                pos_embed_x_nchw = pos_embed_x.transpose([0, 2, 1]).unsqueeze(-1)
                pos_embed_x_interp = F.interpolate(
                    pos_embed_x_nchw,
                    size=target_w,
                    mode='linear',
                    align_corners=False,
                )
                pos_embed_x = pos_embed_x_interp.squeeze(-1).transpose([0, 2, 1])

            pos_embed = pos_embed_y + pos_embed_x
            pos_embed_flat = pos_embed.reshape([1, target_h * target_w, C])

        elif self.pos_embed_type == 'learned':
            if target_h == orig_h and target_w == orig_w:
                pos_embed_flat = self.pos_embed.reshape([1, orig_h * orig_w, C])
            else:
                pos_embed_nchw = self.pos_embed.transpose([0, 3, 1, 2])
                pos_embed_interp = F.interpolate(
                    pos_embed_nchw,
                    size=(target_h, target_w),
                    mode='bicubic',
                    align_corners=False,
                )
                pos_embed_flat = pos_embed_interp.flatten(2).transpose([0, 2, 1])
        else:
            pos_embed_flat = paddle.zeros([1, N, C])

        return pos_embed_flat

    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        x = self.proj(x)
        grid_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose([0, 2, 1])

        pos_embed = self._apply_learned_pos_embed(x, grid_size)
        x = x + pos_embed

        prefix_tokens = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token.expand([B, -1, -1]))
        if self.reg_token is not None:
            prefix_tokens.append(self.reg_token.expand([B, -1, -1]))

        if prefix_tokens:
            prefix_tokens = paddle.concat(prefix_tokens, axis=1)
            x = paddle.concat([prefix_tokens, x], axis=1)

        x = self.pos_drop(x)

        return x, grid_size


class NaFlexSigLIPVisionTransformer(nn.Layer):
    """NaFlex SigLIP Vision Transformer with dynamic resolution support"""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        global_pool: str = 'map',
        act_layer: nn.Layer = GELUTanh,
        class_num: int = 1000,
        pos_embed_grid_size: Tuple[int, int] = (16, 16),
        class_token: bool = False,
        reg_tokens: int = 0,
        fc_norm: bool = False,
        init_values: float = None,
        pos_embed_type: str = 'learned',
        pos_embed_ar_preserving: bool = False,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_prefix_tokens = (1 if class_token else 0) + reg_tokens

        self.embeds = NaFlexEmbeds(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            pos_embed_grid_size=pos_embed_grid_size,
            pos_drop_rate=drop_rate,
            class_token=class_token,
            reg_tokens=reg_tokens,
            pos_embed_type=pos_embed_type,
            pos_embed_ar_preserving=pos_embed_ar_preserving,
        )

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                init_values=init_values,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        if fc_norm and global_pool == 'avg':
            self.fc_norm = nn.LayerNorm(embed_dim)
        else:
            self.fc_norm = None

        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                in_features=embed_dim,
                out_features=embed_dim,
            )
        else:
            self.attn_pool = None

        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else nn.Identity()

    def forward_features(self, x):
        x, grid_size = self.embeds(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.global_pool == 'map':
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            if self.num_prefix_tokens > 0:
                x = x[:, self.num_prefix_tokens:]
            x = x.mean(axis=1)
            if self.fc_norm is not None:
                x = self.fc_norm(x)
        elif self.global_pool == 'max':
            if self.num_prefix_tokens > 0:
                x = x[:, self.num_prefix_tokens:]
            x = x.max(axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def naflexvit_base_patch16_siglip(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexSigLIPVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='map',
        act_layer=GELUTanh,
        pos_embed_grid_size=(16, 16),
        class_token=False,
        reg_tokens=0,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_siglip"], use_ssld=use_ssld)
    return model


def naflexvit_so400m_patch16_siglip(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexSigLIPVisionTransformer(
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7361,
        qkv_bias=True,
        global_pool='map',
        act_layer=GELUTanh,
        pos_embed_grid_size=(16, 16),
        class_token=False,
        reg_tokens=0,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_so400m_patch16_siglip"], use_ssld=use_ssld)
    return model


def naflexvit_base_patch16_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexSigLIPVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        act_layer=nn.GELU,
        pos_embed_grid_size=(36, 36),
        class_token=False,
        reg_tokens=4,
        fc_norm=True,
        init_values=1e-5,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_gap"], use_ssld=use_ssld)
    return model


def naflexvit_base_patch16_par_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexSigLIPVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        act_layer=nn.GELU,
        pos_embed_grid_size=(36, 36),
        class_token=False,
        reg_tokens=4,
        fc_norm=True,
        init_values=1e-5,
        pos_embed_ar_preserving=True,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_par_gap"], use_ssld=use_ssld)
    return model


def naflexvit_base_patch16_parfac_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexSigLIPVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        global_pool='avg',
        act_layer=nn.GELU,
        pos_embed_grid_size=(36, 36),
        class_token=False,
        reg_tokens=4,
        fc_norm=True,
        init_values=1e-5,
        pos_embed_type='factorized',
        pos_embed_ar_preserving=True,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_parfac_gap"], use_ssld=use_ssld)
    return model

