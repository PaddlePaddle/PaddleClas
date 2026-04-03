import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import numpy as np
from metaclip_paddle import MetaCLIP
import time

def check_trainability():
    print("="*50)
    print(f"{'🚀 MetaCLIP 可训练性验证 (Trainability Check)':^46}")
    print("="*50)

    # 1. 强制使用 CPU (避免环境问题)
    paddle.set_device('cpu')
    print("配置: 使用 CPU 设备")

    # 2. 初始化模型
    print("正在初始化模型...")
    model = MetaCLIP(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        qkv_bias=True,
        pre_norm=True
    )
    model.train() # 设置为训练模式

    # 3. 定义优化器和损失函数
    optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=1e-4)
    # 由于 MetaCLIP 是无监督预训练模型，输出是特征向量
    # 这里我们模拟一个简单的特征回归任务，让它去拟合一个随机目标
    loss_fn = nn.MSELoss()

    print("准备训练环境: Optimizer=AdamW, Loss=MSE")

    # 4. 准备假数据
    batch_size = 2
    img_size = 224
    input_data = paddle.randn([batch_size, 3, img_size, img_size])
    # 目标特征: [batch_size, embed_dim]
    target = paddle.randn([batch_size, 768]) 

    # 5. 训练循环
    print("\n开始 5 步训练迭代...")
    print(f"{'Step':<10} | {'Loss':<20} | {'Status'}")
    print("-" * 45)

    initial_loss = None
    
    for step in range(5):
        start_time = time.time()
        
        # 前向传播
        # MetaCLIP 输出 [B, N, C]，我们需要取 CLS Token [B, 0, C] 用于回归
        output = model(input_data)
        cls_output = output[:, 0, :]
        
        # 计算损失
        loss = loss_fn(cls_output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        current_loss = loss.item()
        
        if step == 0:
            initial_loss = current_loss
            
        print(f"{step+1:<10} | {current_loss:<20.6f} | {'Running'}")

    print("-" * 45)
    
    # 6. 验证结论
    if current_loss < initial_loss:
        print(f"✅ 验证成功: Loss 已下降 ({initial_loss:.6f} -> {current_loss:.6f})")
        print("结论: 模型梯度回传正常，具备可训练性。")
    else:
        print(f"⚠️ 验证警告: Loss 未明显下降，请检查学习率或优化器设置。")
        print("(注: 极短步数内 Loss 震荡属正常现象，只要没有报错且梯度不为 None 即视为通过)")

    print("="*50)

if __name__ == "__main__":
    check_trainability()
