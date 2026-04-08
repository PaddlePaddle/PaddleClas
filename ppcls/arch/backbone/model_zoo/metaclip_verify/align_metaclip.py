import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import timm
import os
from functools import partial
from timm.models import load_checkpoint

# --- Paddle Model Definition (Cleaned from paconvert errors) ---
class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (paddle.matmul(q, k.transpose([0, 1, 3, 2]))) * self.scale
        attn = F.softmax(attn, axis=-1); attn = self.attn_drop(attn)
        x = (paddle.matmul(attn, v)).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x); x = self.proj_drop(x)
        return x

class Block(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias_attr=False)
    def forward(self, x):
        x = self.proj(x); x = x.flatten(2).transpose([0, 2, 1])
        return x

class MetaCLIP(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.Constant(0.))
        self.pos_embed = self.create_parameter(shape=[1, num_patches + 1, embed_dim], default_initializer=nn.initializer.Constant(0.))
        # pre_norm=True creates a LayerNorm before the blocks
        self.norm_pre = nn.LayerNorm(embed_dim, epsilon=1e-5)
        self.blocks = nn.LayerList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-5)
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat((cls_token, x), axis=1)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        return x
# ---------------------------------------------------------------------------

def convert_pytorch_to_paddle(torch_model, paddle_model):
    """
    终极权重转换逻辑：解决 Key 错位和参数丢失问题
    """
    torch_state_dict = torch_model.state_dict()
    paddle_state_dict = paddle_model.state_dict()
    new_weight_dict = {}

    print("\n🔍 正在执行深度参数对齐...")
    
    for pd_key in paddle_state_dict.keys():
        # 核心逻辑：尝试多种可能的 Torch Key 变体
        # 1. 完全一致 2. 加上 model. 前缀 3. 去掉前缀
        potential_keys = [pd_key, "model." + pd_key, pd_key.replace("blocks.", "blocks.")]
        
        # 针对 cls_token, pos_embed, patch_embed 的特殊处理
        if "patch_embed" in pd_key: potential_keys.append(pd_key.replace("patch_embed", "patch_embed"))
        
        found_tk = None
        for tk in potential_keys:
            if tk in torch_state_dict:
                found_tk = tk
                break
        
        if found_tk:
            v = torch_state_dict[found_tk].detach().cpu().numpy()
            # Linear 层转置
            if v.ndim == 2 and any(x in found_tk for x in ["qkv", "fc", "proj"]):
                v = v.transpose()
            new_weight_dict[pd_key] = v
        else:
            print(f"❌ 严重警告: Paddle 参数 [{pd_key}] 无法在 PyTorch 中找到对应权重!")

    # 特殊补丁：强制对齐那些可能因为命名差异被漏掉的全局参数
    special_pairs = {
        "cls_token": "cls_token",
        "pos_embed": "pos_embed",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
        "norm_pre.weight": "norm_pre.weight",
        "norm_pre.bias": "norm_pre.bias"
    }
    for pd_k, pt_k in special_pairs.items():
        if pd_k in paddle_state_dict and pd_k not in new_weight_dict:
            if pt_k in torch_state_dict:
                v = torch_state_dict[pt_k].detach().cpu().numpy()
                new_weight_dict[pd_k] = v
                print(f"✅ 已手动修复全局参数: {pd_k}")

    paddle_model.set_dict(new_weight_dict)
    print(f"✨ 权重加载完成! 成功加载 {len(new_weight_dict)}/{len(paddle_state_dict)} 个参数。\n")

def align():
    # 强制使用 CPU，避免因环境配置导致的 CUDNN/GPU 加载错误
    # 昨天的成功可能也是运行在 CPU 上，或者当时的环境变量使得 Paddle 默认选择了 CPU
    paddle.set_device('cpu')

    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    paddle.seed(seed)

    # 1. 准备模型配置 (以 vit_base_patch16 为例)
    model_name = 'vit_base_patch16_clip_224'
    ckpt_path = './weights/metaclip_b16.safetensors'
    
    print(f"🚀 正在初始化 PyTorch 模型: {model_name}")
    # 强制不下载权重，手动加载本地文件
    pt_model = timm.create_model(model_name, pretrained=False, num_classes=0)
    
    if os.path.exists(ckpt_path):
        print(f"📦 正在从本地加载权重: {ckpt_path}")
        load_checkpoint(pt_model, ckpt_path, strict=False)
    else:
        print(f"⚠️ 找不到本地权重 {ckpt_path}，将尝试联网下载...")
        pt_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    pt_model.eval()
    print("✅ PyTorch 模型就绪")

    print("🚀 正在初始化 Paddle 模型 (metaclip_paddle.py)...")
    pd_model = MetaCLIP(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True
    )
    pd_model.eval()

    # 2. 权重转换
    convert_pytorch_to_paddle(pt_model, pd_model)

    # 3. 准备输入数据
    input_data = np.random.randn(1, 3, 224, 224).astype('float32')

    # 4. 前向传播
    with torch.no_grad():
        # timm 在 num_classes=0 时通常只返回 CLS token [1, 768]
        pt_output = pt_model(torch.from_numpy(input_data)).numpy()
    
    with paddle.no_grad():
        pd_model.eval()
        # Paddle 模型目前返回的是全量 tokens [1, 197, 768]
        pd_all_tokens = pd_model(paddle.to_tensor(input_data)).numpy()
        
        # 核心对齐逻辑：如果 PyTorch 只返回了特征向量，我们也只取 Paddle 的第一个 token
        if pt_output.ndim == 2 and pd_all_tokens.ndim == 3:
            print(f"💡 检测到输出维度差异，正在自动对齐特征提取位置...")
            pd_output = pd_all_tokens[:, 0, :] # 只取 CLS Token
        else:
            pd_output = pd_all_tokens

    print(f"📏 PyTorch 输出形状: {pt_output.shape}")
    print(f"📏 Paddle 输出形状: {pd_output.shape}")

    # 5. 对比结果
    diff = np.abs(pt_output - pd_output)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)

    print("\n" + "="*50)
    print(f"{'📊 对齐结果报告':^48}")
    print("="*50)
    print(f"Max Absolute Diff: {max_diff:.8e}")
    print(f"Avg Absolute Diff: {avg_diff:.8e}")
    print("-" * 50)

    if max_diff < 1e-4:
        print("✨ 结论: 前向对齐成功! (Diff < 1e-4)")
    else:
        print("❌ 结论: 前向对齐失败，请检查权重转换或组网逻辑。")
    print("="*50)

if __name__ == "__main__":
    align()
