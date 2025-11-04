"""
GPU检测脚本
用于检测PyTorch是否正确识别和使用GPU
"""

import torch
import sys

print("=" * 60)
print("PyTorch GPU 检测")
print("=" * 60)

# 1. PyTorch版本
print(f"\n1. PyTorch版本: {torch.__version__}")

# 2. CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"2. CUDA是否可用: {'✓ 是' if cuda_available else '✗ 否'}")

if cuda_available:
    # 3. CUDA版本
    print(f"3. CUDA版本: {torch.version.cuda}")
    
    # 4. GPU设备数量
    gpu_count = torch.cuda.device_count()
    print(f"4. 可用GPU数量: {gpu_count}")
    
    # 5. GPU设备信息
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"   - 设备名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   - 总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - 计算能力: {props.major}.{props.minor}")
        print(f"   - 多处理器数量: {props.multi_processor_count}")
    
    # 6. 当前设备
    current_device = torch.cuda.current_device()
    print(f"\n5. 当前使用的GPU: GPU {current_device}")
    
    # 7. 测试GPU计算
    print(f"\n6. 测试GPU计算:")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"   ✓ GPU矩阵运算测试成功")
        print(f"   - 测试张量设备: {z.device}")
        
        # 测量GPU速度
        import time
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # 测量CPU速度
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(100):
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"   - GPU计算时间: {gpu_time:.4f}秒")
        print(f"   - CPU计算时间: {cpu_time:.4f}秒")
        print(f"   - 加速比: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"   ✗ GPU计算测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("✓ GPU可用，DQN训练将使用GPU加速")
    print("=" * 60)
    
else:
    print("\n" + "=" * 60)
    print("✗ GPU不可用，原因可能是:")
    print("  1. 安装了CPU版本的PyTorch")
    print("  2. CUDA驱动未正确安装")
    print("  3. GPU驱动版本与CUDA版本不匹配")
    print("\n安装GPU版PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60)
    sys.exit(1)

