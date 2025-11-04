#!/usr/bin/env python3
"""
GPU加速功能测试脚本
测试GPU加速是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.simulation_config import ConfigManager
from utils.gpu_compute import get_gpu_memory_info, cleanup_gpu_memory

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 50)
    print("GPU加速功能测试")
    print("=" * 50)
    
    # 检查GPU内存信息
    gpu_info = get_gpu_memory_info()
    print(f"GPU状态: {gpu_info}")
    
    # 测试配置管理器
    print("\n1. 测试配置管理器...")
    config_manager = ConfigManager()
    
    # 测试CPU模式
    print("\n2. 测试CPU模式...")
    config_manager.simulation_config.use_gpu_acceleration = False
    network_cpu = config_manager.create_network()
    simulation_cpu = config_manager.create_energy_simulation(network_cpu)
    print(f"CPU模式 - 网络节点数: {len(network_cpu.nodes)}")
    print(f"CPU模式 - 仿真GPU状态: {simulation_cpu.use_gpu}")
    
    # 测试GPU模式
    print("\n3. 测试GPU模式...")
    config_manager.simulation_config.use_gpu_acceleration = True
    network_gpu = config_manager.create_network()
    simulation_gpu = config_manager.create_energy_simulation(network_gpu)
    print(f"GPU模式 - 网络节点数: {len(network_gpu.nodes)}")
    print(f"GPU模式 - 仿真GPU状态: {simulation_gpu.use_gpu}")
    
    # 测试距离计算
    print("\n4. 测试距离计算...")
    if len(network_cpu.nodes) >= 2:
        # 使用同一个网络的节点进行测试
        node1, node2 = network_cpu.nodes[0], network_cpu.nodes[1]
        
        # CPU距离计算
        cpu_distance = node1.distance_to(node2)
        print(f"CPU距离计算: {cpu_distance:.4f}")
        
        # GPU距离计算（使用同一个网络）
        gpu_distance = network_cpu.get_distance(node1, node2)
        print(f"GPU距离计算: {gpu_distance:.4f}")
        
        # 验证结果一致性
        if abs(cpu_distance - gpu_distance) < 1e-6:
            print("✅ 距离计算结果一致")
        else:
            print("❌ 距离计算结果不一致")
        
        # 测试GPU网络的GPU距离计算
        node1_gpu, node2_gpu = network_gpu.nodes[0], network_gpu.nodes[1]
        gpu_distance_gpu = network_gpu.get_distance(node1_gpu, node2_gpu)
        print(f"GPU网络距离计算: {gpu_distance_gpu:.4f}")
    
    # 清理GPU内存
    cleanup_gpu_memory()
    
    print("\n" + "=" * 50)
    print("GPU加速功能测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_gpu_availability()
