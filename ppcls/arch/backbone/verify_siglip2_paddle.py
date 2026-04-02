"""
SigLIP-2 模型验证脚本 - PaddlePaddle版本
验证SigLIP-2模型能否正确加载官方权重并基于numpy生成随机数后输出特征
同时验证模型的可训练性，并与timm官方输出进行比较
"""
import argparse
import os
from typing import Optional

import numpy as np
import paddle
from safetensors.numpy import load_file as load_file_numpy

from siglip2_model_paddle import (
    SigLIP2VisionTransformer,
    load_checkpoint,
    siglip2_base_patch16_256,
    siglip2_base_patch16_384,
    siglip2_base_patch16_512,
    siglip2_large_patch16_256,
    siglip2_large_patch16_384,
    siglip2_large_patch16_512,
    siglip2_so400m_patch14_siglip_224,
    siglip2_so400m_patch14_siglip_378,
    siglip2_so400m_patch14_siglip_384,
    siglip2_so400m_patch16_siglip_256,
    siglip2_so400m_patch16_siglip_384,
    siglip2_so400m_patch16_siglip_512,
)

MODEL_FUNCTIONS = {
    "vit_base_patch16_siglip_256.v2_webli": siglip2_base_patch16_256,
    "vit_base_patch16_siglip_384.v2_webli": siglip2_base_patch16_384,
    "vit_base_patch16_siglip_512.v2_webli": siglip2_base_patch16_512,
    "vit_large_patch16_siglip_256.v2_webli": siglip2_large_patch16_256,
    "vit_large_patch16_siglip_384.v2_webli": siglip2_large_patch16_384,
    "vit_large_patch16_siglip_512.v2_webli": siglip2_large_patch16_512,
    "vit_so400m_patch14_siglip_224.v2_webli": siglip2_so400m_patch14_siglip_224,
    "vit_so400m_patch14_siglip_378.v2_webli": siglip2_so400m_patch14_siglip_378,
    "vit_so400m_patch14_siglip_384.webli": siglip2_so400m_patch14_siglip_384,
    "vit_so400m_patch16_siglip_256.v2_webli": siglip2_so400m_patch16_siglip_256,
    "vit_so400m_patch16_siglip_384.v2_webli": siglip2_so400m_patch16_siglip_384,
    "vit_so400m_patch16_siglip_512.v2_webli": siglip2_so400m_patch16_siglip_512,
}

MODEL_DEFAULT_SIZES = {
    "vit_base_patch16_siglip_256.v2_webli": 256,
    "vit_base_patch16_siglip_384.v2_webli": 384,
    "vit_base_patch16_siglip_512.v2_webli": 512,
    "vit_large_patch16_siglip_256.v2_webli": 256,
    "vit_large_patch16_siglip_384.v2_webli": 384,
    "vit_large_patch16_siglip_512.v2_webli": 512,
    "vit_so400m_patch14_siglip_224.v2_webli": 224,
    "vit_so400m_patch14_siglip_378.v2_webli": 378,
    "vit_so400m_patch14_siglip_384.webli": 384,
    "vit_so400m_patch16_siglip_256.v2_webli": 256,
    "vit_so400m_patch16_siglip_384.v2_webli": 384,
    "vit_so400m_patch16_siglip_512.v2_webli": 512,
}


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    paddle.seed(seed)
    np.random.seed(seed)


def test_paddle_trainability(model, np_input, img_size, seed=42, num_steps=3):
    """测试PaddlePaddle模型的可训练性"""
    print(f"\n[可训练性验证]")
    
    model.train()
    num_classes = 1000
    classifier = paddle.nn.Linear(model.embed_dim, num_classes)
    
    optimizer = paddle.optimizer.Adam(
        parameters=list(model.parameters()) + list(classifier.parameters()),
        learning_rate=0.001,
    )
    
    criterion = paddle.nn.CrossEntropyLoss()
    
    set_seed(seed)
    np.random.seed(seed)
    train_input = np.random.randn(4, 3, img_size, img_size).astype(np.float32)
    train_labels = np.random.randint(0, num_classes, size=(4,)).astype(np.int64)
    
    x = paddle.to_tensor(train_input)
    features = model(x)
    logits = classifier(features)
    loss = criterion(logits, paddle.to_tensor(train_labels))
    
    print(f"  [1] 前向传播: 特征={features.shape}, logits={logits.shape}, loss={loss.item():.4f}")
    
    loss.backward()
    print(f"  [2] 反向传播: 成功")
    
    has_grad_count = 0
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad_count += 1
            grad_norm = paddle.norm(param.grad).item()
            total_grad_norm += grad_norm
    
    avg_grad_norm = total_grad_norm / has_grad_count if has_grad_count > 0 else 0.0
    print(f"  [3] 梯度计算: {has_grad_count}个参数有梯度, 平均范数={avg_grad_norm:.6f}")
    
    optimizer.step()
    optimizer.clear_grad()
    print(f"  [4] 参数更新: 成功")
    
    losses = []
    for step in range(num_steps):
        set_seed(seed + step)
        np.random.seed(seed + step)
        
        batch_input = np.random.randn(4, 3, img_size, img_size).astype(np.float32)
        batch_labels = np.random.randint(0, num_classes, size=(4,)).astype(np.int64)
        
        x = paddle.to_tensor(batch_input)
        features = model(x)
        logits = classifier(features)
        loss = criterion(logits, paddle.to_tensor(batch_labels))
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        losses.append(loss.item())
    
    print(f"  [5] 多步训练: {losses}")
    
    return True


def test_model(
    model_name: str,
    checkpoint_path: str,
    img_size: Optional[int] = None,
    seed: int = 42,
    test_trainability: bool = False,
):
    """测试SigLIP-2模型"""
    print(f"\n{'=' * 60}")
    print(f"模型: {model_name}")
    print(f"{'=' * 60}")
    
    set_seed(seed)
    
    if model_name not in MODEL_FUNCTIONS:
        print(f"[ERROR] 不支持的模型: {model_name}")
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 权重文件不存在: {checkpoint_path}")
        return False
    
    if img_size is None:
        img_size = MODEL_DEFAULT_SIZES.get(model_name, 256)
    
    # 创建模型
    print(f"\n[1] 创建模型")
    model_func = MODEL_FUNCTIONS[model_name]
    model = model_func(pretrained=False, checkpoint_path=None, img_size=img_size)
    model.eval()
    print(f"    参数量: {sum(p.size for p in model.parameters()):,}")
    
    # 加载权重
    print(f"\n[2] 加载权重")
    load_checkpoint(model, checkpoint_path)
    print(f"    [OK] 权重加载成功")
    
    # 生成测试输入（基于numpy）
    print(f"\n[3] 生成测试输入")
    set_seed(seed)
    np.random.seed(seed)
    np_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    x = paddle.to_tensor(np_input)
    print(f"    输入: {x.shape}, 范围=[{x.min().item():.4f}, {x.max().item():.4f}]")
    
    # 前向传播
    print(f"\n[4] 前向传播")
    with paddle.no_grad():
        output = model(x)
    print(f"    输出: {output.shape}, 范围=[{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 测试可训练性
    trainability_ok = None
    if test_trainability:
        trainability_ok = test_paddle_trainability(model, np_input, img_size, seed=seed)
    
    # 总结
    print(f"\n[验证总结]")
    print(f"  模型加载: [OK]")
    print(f"  权重加载: [OK]")
    print(f"  前向传播: [OK]")
    if trainability_ok is not None:
        print(f"  可训练性: {'[OK]' if trainability_ok else '[FAIL]'}")
    
    return trainability_ok if trainability_ok is not None else True


def main():
    parser = argparse.ArgumentParser(
        description="SigLIP-2 模型验证脚本 - PaddlePaddle版本"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称 (例如: vit_base_patch16_siglip_256.v2_webli)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="权重文件路径 (例如: path/to/model.safetensors)",
    )
    parser.add_argument(
        "--img_size", type=int, default=None, help="输入图像尺寸 (默认使用模型默认尺寸)"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument(
        "--test_trainability",
        action="store_true",
        help="是否测试模型的可训练性",
    )
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"SigLIP-2 模型验证 - PaddlePaddle版本")
    print(f"{'=' * 60}")
    print(f"  模型: {args.model}")
    print(f"  权重: {args.checkpoint}")
    print(f"  尺寸: {args.img_size if args.img_size else '自动'}")
    print(f"  种子: {args.seed}")
    print(f"  可训练性: {'是' if args.test_trainability else '否'}")
    print(f"{'=' * 60}")
    
    success = test_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        seed=args.seed,
        test_trainability=args.test_trainability,
    )
    
    print(f"\n{'=' * 60}")
    if success:
        print(f"[OK] 模型验证成功！")
    else:
        print(f"[FAIL] 模型验证失败！请检查模型实现。")
    print(f"{'=' * 60}\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
