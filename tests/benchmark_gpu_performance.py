#!/usr/bin/env python3
"""
GPU加速性能对比测试
比较CPU和GPU在不同规模网络下的性能差异
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.simulation_config import ConfigManager
from utils.gpu_compute import (
    get_gpu_manager, 
    compute_distance_matrix_gpu,
    compute_statistics_gpu,
    cleanup_gpu_memory,
    get_gpu_memory_info
)


def benchmark_distance_calculation(num_nodes_list=[50, 100, 200, 500]):
    """测试不同规模下的距离计算性能"""
    print("=" * 80)
    print("距离矩阵计算性能对比")
    print("=" * 80)
    print(f"{'节点数':<10} {'CPU时间(s)':<15} {'GPU时间(s)':<15} {'加速比':<10}")
    print("-" * 80)
    
    results = []
    
    for num_nodes in num_nodes_list:
        # 创建测试网络
        config_manager = ConfigManager()
        config_manager.network_config.num_nodes = num_nodes
        config_manager.simulation_config.use_gpu_acceleration = False
        
        network_cpu = config_manager.create_network()
        nodes = network_cpu.nodes
        
        # CPU测试
        gpu_manager_cpu = get_gpu_manager(use_gpu=False)
        start_time = time.time()
        for _ in range(10):  # 重复10次取平均
            _ = compute_distance_matrix_gpu(nodes, gpu_manager_cpu)
        cpu_time = (time.time() - start_time) / 10
        
        # GPU测试
        gpu_manager_gpu = get_gpu_manager(use_gpu=True)
        # 预热GPU
        _ = compute_distance_matrix_gpu(nodes, gpu_manager_gpu)
        
        start_time = time.time()
        for _ in range(10):  # 重复10次取平均
            _ = compute_distance_matrix_gpu(nodes, gpu_manager_gpu)
        gpu_time = (time.time() - start_time) / 10
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{num_nodes:<10} {cpu_time:<15.6f} {gpu_time:<15.6f} {speedup:<10.2f}x")
        
        results.append({
            'num_nodes': num_nodes,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
        
        cleanup_gpu_memory()
    
    print("=" * 80)
    return results


def benchmark_statistics_calculation():
    """测试统计计算性能"""
    print("\n" + "=" * 80)
    print("统计计算性能对比")
    print("=" * 80)
    print(f"{'数据规模':<15} {'CPU时间(s)':<15} {'GPU时间(s)':<15} {'加速比':<10}")
    print("-" * 80)
    
    data_sizes = [1000, 10000, 100000, 1000000]
    
    for size in data_sizes:
        # 生成测试数据
        data = np.random.randn(size) * 1000 + 5000
        
        # CPU测试
        gpu_manager_cpu = get_gpu_manager(use_gpu=False)
        start_time = time.time()
        for _ in range(100):  # 重复100次
            _ = compute_statistics_gpu(data, gpu_manager_cpu)
        cpu_time = (time.time() - start_time) / 100
        
        # GPU测试
        gpu_manager_gpu = get_gpu_manager(use_gpu=True)
        # 预热
        _ = compute_statistics_gpu(data, gpu_manager_gpu)
        
        start_time = time.time()
        for _ in range(100):  # 重复100次
            _ = compute_statistics_gpu(data, gpu_manager_gpu)
        gpu_time = (time.time() - start_time) / 100
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{size:<15} {cpu_time:<15.6f} {gpu_time:<15.6f} {speedup:<10.2f}x")
        
        cleanup_gpu_memory()
    
    print("=" * 80)


def benchmark_full_simulation():
    """测试完整仿真性能"""
    print("\n" + "=" * 80)
    print("完整仿真性能对比 (100步)")
    print("=" * 80)
    
    node_counts = [50, 100, 200]
    
    for num_nodes in node_counts:
        print(f"\n测试 {num_nodes} 节点网络...")
        
        # CPU仿真
        print("  运行CPU仿真...")
        config_manager_cpu = ConfigManager()
        config_manager_cpu.network_config.num_nodes = num_nodes
        config_manager_cpu.simulation_config.time_steps = 100
        config_manager_cpu.simulation_config.use_gpu_acceleration = False
        config_manager_cpu.simulation_config.enable_detailed_plan_log = False
        config_manager_cpu.simulation_config.enable_plots = False
        
        network_cpu = config_manager_cpu.create_network()
        simulation_cpu = config_manager_cpu.create_energy_simulation(network_cpu)
        
        start_time = time.time()
        simulation_cpu.run()
        cpu_time = time.time() - start_time
        
        # GPU仿真
        print("  运行GPU仿真...")
        config_manager_gpu = ConfigManager()
        config_manager_gpu.network_config.num_nodes = num_nodes
        config_manager_gpu.simulation_config.time_steps = 100
        config_manager_gpu.simulation_config.use_gpu_acceleration = True
        config_manager_gpu.simulation_config.enable_detailed_plan_log = False
        config_manager_gpu.simulation_config.enable_plots = False
        
        network_gpu = config_manager_gpu.create_network()
        simulation_gpu = config_manager_gpu.create_energy_simulation(network_gpu)
        
        start_time = time.time()
        simulation_gpu.run()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"\n  结果:")
        print(f"    CPU时间: {cpu_time:.2f}s")
        print(f"    GPU时间: {gpu_time:.2f}s")
        print(f"    加速比:  {speedup:.2f}x")
        
        cleanup_gpu_memory()
    
    print("\n" + "=" * 80)


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print(" " * 25 + "GPU性能测试套件")
    print("=" * 80)
    
    # 显示GPU信息
    gpu_info = get_gpu_memory_info()
    print(f"\nGPU信息:")
    print(f"  可用性: {gpu_info.get('available', False)}")
    if gpu_info.get('available'):
        total_mem = gpu_info.get('total_memory', 0) / 1e9
        print(f"  总显存: {total_mem:.2f} GB")
    print()
    
    # 运行性能测试
    try:
        # 1. 距离计算测试
        distance_results = benchmark_distance_calculation()
        
        # 2. 统计计算测试
        benchmark_statistics_calculation()
        
        # 3. 完整仿真测试（可选，较耗时）
        print("\n是否运行完整仿真测试？(较耗时，约5-10分钟)")
        print("跳过完整仿真测试...")
        # benchmark_full_simulation()
        
        # 总结
        print("\n" + "=" * 80)
        print("性能测试总结")
        print("=" * 80)
        print("\n距离计算加速比:")
        for r in distance_results:
            print(f"  {r['num_nodes']} 节点: {r['speedup']:.2f}x")
        
        print("\n💡 建议:")
        avg_speedup = np.mean([r['speedup'] for r in distance_results])
        if avg_speedup > 2.0:
            print("  ✅ GPU加速效果显著，建议在大规模仿真中启用GPU")
        elif avg_speedup > 1.2:
            print("  ✓ GPU有一定加速效果，对于中大规模网络建议启用")
        else:
            print("  ⚠️ GPU加速效果不明显，可能是因为数据传输开销较大")
            print("     建议只在节点数 > 200 时使用GPU")
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_gpu_memory()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


