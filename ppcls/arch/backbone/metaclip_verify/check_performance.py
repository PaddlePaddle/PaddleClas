import paddle
import time
from metaclip_paddle import MetaCLIP
import numpy as np

def check_performance():
    print("="*60)
    print(f"{'🚀 MetaCLIP 性能对比验证 (Performance Benchmark)':^56}")
    print("="*60)
    
    # 1. 强制使用 CPU
    paddle.set_device('cpu')
    print("环境: CPU")
    
    # 2. 初始化模型
    print("初始化模型 (Dynamic Mode)...")
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
    model.eval()

    # 3. 准备数据
    batch_size = 1
    img_size = 224
    input_data = paddle.randn([batch_size, 3, img_size, img_size])
    
    # 4. 动态图基准测试 (Warmup)
    print("\n🔍 开始动态图 (Dynamic Graph) 预热...")
    for _ in range(5):
        _ = model(input_data)
        
    print("⚡ 动态图推理测速 (10 次)...")
    start_time = time.time()
    for _ in range(10):
        _ = model(input_data)
    dynamic_time = (time.time() - start_time) / 10
    print(f"  Avg Time: {dynamic_time*1000:.2f} ms / batch")
    
    # 5. 静态图 (JIT) 转换
    print("\n🔨 转换为静态图 (Static Graph / JIT)...")
    jit_model = paddle.jit.to_static(model, input_spec=[input_data])
    
    # 6. 静态图预热
    print("🔍 静态图预热...")
    for _ in range(5):
        _ = jit_model(input_data)
        
    print("⚡ 静态图推理测速 (10 次)...")
    start_time = time.time()
    for _ in range(10):
        _ = jit_model(input_data)
    static_time = (time.time() - start_time) / 10
    print(f"  Avg Time: {static_time*1000:.2f} ms / batch")
    
    # 7. 计算提升比例
    speedup = (dynamic_time - static_time) / dynamic_time * 100
    print("-" * 60)
    print(f"📈 性能对比结果:")
    print(f"  Dynamic: {dynamic_time*1000:.2f} ms")
    print(f"  Static : {static_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}%")
    
    if speedup > 10.0:
        print("✅ 达标: 性能提升 > 10% (满足验收标准)")
    else:
        print("⚠️ 警告: 性能提升不足 10%，可能因 CPU 负载或模型本身已高度优化。")
        print("建议在更稳定的服务器环境或 GPU 上复测。")
        
    print("="*60)

if __name__ == "__main__":
    check_performance()