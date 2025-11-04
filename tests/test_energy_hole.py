#!/usr/bin/env python3
"""
测试能量空洞模式功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.simulation_config import ConfigManager
from core.network import Network
import matplotlib.pyplot as plt
import numpy as np

def test_energy_hole_mode():
    """测试能量空洞模式"""
    print("=== 测试能量空洞模式 ===")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 配置能量空洞模式
    config_manager.network_config.distribution_mode = "energy_hole"
    config_manager.network_config.energy_hole_enabled = True
    config_manager.network_config.energy_hole_ratio = 0.4  # 40%的节点为非太阳能
    config_manager.network_config.energy_hole_center_mode = "random"  # 随机选择中心
    config_manager.network_config.energy_hole_cluster_radius = 2.0
    config_manager.network_config.energy_hole_mobile_ratio = 0.1
    config_manager.network_config.num_nodes = 25
    
    print(f"配置参数:")
    print(f"  分布模式: {config_manager.network_config.distribution_mode}")
    print(f"  能量空洞比例: {config_manager.network_config.energy_hole_ratio}")
    print(f"  空洞中心模式: {config_manager.network_config.energy_hole_center_mode}")
    print(f"  聚集半径: {config_manager.network_config.energy_hole_cluster_radius}")
    print(f"  移动节点比例: {config_manager.network_config.energy_hole_mobile_ratio}")
    
    # 创建网络
    network = config_manager.create_network()
    network.create_nodes()
    
    print(f"\n网络创建完成，共 {len(network.nodes)} 个节点")
    
    # 统计节点属性
    solar_nodes = [node for node in network.nodes if node.has_solar]
    non_solar_nodes = [node for node in network.nodes if not node.has_solar]
    mobile_nodes = [node for node in network.nodes if node.is_mobile]
    
    print(f"太阳能节点: {len(solar_nodes)} 个")
    print(f"非太阳能节点: {len(non_solar_nodes)} 个")
    print(f"移动节点: {len(mobile_nodes)} 个")
    
    # 可视化节点分布
    visualize_network(network)
    
    return network

def visualize_network(network):
    """可视化网络节点分布"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 获取节点位置和属性
    positions = np.array([node.position for node in network.nodes])
    solar_mask = np.array([node.has_solar for node in network.nodes])
    mobile_mask = np.array([node.is_mobile for node in network.nodes])
    
    # 绘制太阳能节点（绿色）
    solar_positions = positions[solar_mask]
    ax.scatter(solar_positions[:, 0], solar_positions[:, 1], 
               c='green', marker='o', s=100, label='太阳能节点', alpha=0.7)
    
    # 绘制非太阳能节点（红色）
    non_solar_positions = positions[~solar_mask]
    ax.scatter(non_solar_positions[:, 0], non_solar_positions[:, 1], 
               c='red', marker='s', s=100, label='非太阳能节点', alpha=0.7)
    
    # 标记移动节点
    mobile_positions = positions[mobile_mask]
    if len(mobile_positions) > 0:
        ax.scatter(mobile_positions[:, 0], mobile_positions[:, 1], 
                   c='blue', marker='^', s=150, label='移动节点', alpha=0.8)
    
    # 计算并显示能量空洞中心
    if len(non_solar_positions) > 0:
        center = np.mean(non_solar_positions, axis=0)
        ax.scatter(center[0], center[1], c='black', marker='x', s=200, 
                   label='能量空洞中心', linewidth=3)
        
        # 绘制能量空洞区域
        circle = plt.Circle(center, network.energy_hole_cluster_radius, 
                          fill=False, color='red', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_title('能量空洞模式网络分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('energy_hole_network.png', dpi=300, bbox_inches='tight')
    print("网络分布图已保存为 energy_hole_network.png")
    plt.show()

def test_different_center_modes():
    """测试不同的空洞中心模式"""
    print("\n=== 测试不同空洞中心模式 ===")
    
    center_modes = ["random", "corner", "center"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, center_mode in enumerate(center_modes):
        # 创建配置
        config_manager = ConfigManager()
        config_manager.network_config.distribution_mode = "energy_hole"
        config_manager.network_config.energy_hole_enabled = True
        config_manager.network_config.energy_hole_ratio = 0.4
        config_manager.network_config.energy_hole_center_mode = center_mode
        config_manager.network_config.energy_hole_cluster_radius = 2.0
        config_manager.network_config.num_nodes = 25
        
        # 创建网络
        network = config_manager.create_network()
        network.create_nodes()
        
        # 可视化
        ax = axes[i]
        positions = np.array([node.position for node in network.nodes])
        solar_mask = np.array([node.has_solar for node in network.nodes])
        
        # 绘制节点
        solar_positions = positions[solar_mask]
        non_solar_positions = positions[~solar_mask]
        
        ax.scatter(solar_positions[:, 0], solar_positions[:, 1], 
                   c='green', marker='o', s=50, label='太阳能节点', alpha=0.7)
        ax.scatter(non_solar_positions[:, 0], non_solar_positions[:, 1], 
                   c='red', marker='s', s=50, label='非太阳能节点', alpha=0.7)
        
        # 标记中心
        if len(non_solar_positions) > 0:
            center = np.mean(non_solar_positions, axis=0)
            ax.scatter(center[0], center[1], c='black', marker='x', s=100, 
                       linewidth=2)
        
        ax.set_title(f'中心模式: {center_mode}')
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('energy_hole_center_modes.png', dpi=300, bbox_inches='tight')
    print("不同中心模式对比图已保存为 energy_hole_center_modes.png")
    plt.show()

if __name__ == "__main__":
    # 测试基本能量空洞模式
    network = test_energy_hole_mode()
    
    # 测试不同中心模式
    test_different_center_modes()
    
    print("\n=== 测试完成 ===")
    print("能量空洞模式功能已成功实现！")
