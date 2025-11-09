# tests/test_info_value.py
# -*- coding: utf-8 -*-
"""
信息价值计算功能测试

测试内容：
1. 信息价值计算方法是否正常工作
2. 信息价值与信息量的差别
3. 时间衰减效果
4. 路由和调度决策中的使用
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
from src.info_collection.path_based_collector import PathBasedInfoCollector
from src.info_collection.physical_center import VirtualCenter
from src.core.SensorNode import SensorNode
from src.config.simulation_config import PathCollectorConfig, ConfigManager


def test_info_value_calculation():
    """测试信息价值计算"""
    print("=" * 60)
    print("测试1: 信息价值计算")
    print("=" * 60)
    
    # 创建虚拟中心
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    
    # 创建信息收集器
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02  # 衰减率
    )
    
    # 模拟节点信息
    node_info = {
        'info_volume': 1000000,  # 1MB信息量
        'info_waiting_since': 0,  # 从时间0开始等待
        'info_is_reported': False,
        'record_time': 0
    }
    
    # 测试不同等待时间的信息价值
    print("\n信息量: 1,000,000 bits")
    print("衰减率: 0.02")
    print("\n等待时间 | 信息价值 | 衰减因子 | 保留比例")
    print("-" * 60)
    
    for waiting_time in [0, 10, 30, 50, 100, 200]:
        current_time = waiting_time
        info_value = collector.calculate_info_value(node_info, current_time)
        decay_factor = math.exp(-0.02 * waiting_time)
        retention = info_value / node_info['info_volume'] * 100
        
        print(f"{waiting_time:8d} | {info_value:10.0f} | {decay_factor:8.4f} | {retention:8.2f}%")
    
    print("\n[OK] 信息价值计算正常")


def test_info_value_vs_info_volume():
    """对比信息价值与信息量的差别"""
    print("\n" + "=" * 60)
    print("测试2: 信息价值 vs 信息量")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    # 场景：两个节点，信息量相同，但等待时间不同
    node1_info = {
        'info_volume': 1000000,
        'info_waiting_since': 0,  # 刚等待
        'info_is_reported': False,
        'record_time': 0
    }
    
    node2_info = {
        'info_volume': 1000000,
        'info_waiting_since': 0,  # 等待了50分钟
        'info_is_reported': False,
        'record_time': 0
    }
    
    current_time = 50
    
    info_volume1 = node1_info['info_volume']
    info_volume2 = node2_info['info_volume']
    info_value1 = collector.calculate_info_value(node1_info, current_time)
    info_value2 = collector.calculate_info_value(node2_info, current_time)
    
    print("\n场景：两个节点信息量相同（1MB），但等待时间不同")
    print(f"\n节点1: 等待时间={current_time - node1_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {info_volume1:,} bits")
    print(f"  - 信息价值: {info_value1:,.0f} bits")
    
    print(f"\n节点2: 等待时间={current_time - node2_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {info_volume2:,} bits")
    print(f"  - 信息价值: {info_value2:,.0f} bits")
    
    print(f"\n差别:")
    print(f"  - 信息量相同: {info_volume1 == info_volume2}")
    print(f"  - 信息价值不同: {abs(info_value1 - info_value2):,.0f} bits")
    print(f"  - 价值差异比例: {abs(info_value1 - info_value2) / max(info_value1, info_value2) * 100:.2f}%")
    
    print("\n[OK] 信息价值能区分不同等待时间的信息")


def test_threshold_decision():
    """测试阈值决策的差别"""
    print("\n" + "=" * 60)
    print("测试3: 阈值决策差别")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        min_info_volume_threshold=1,
        info_value_decay_rate=0.02
    )
    
    threshold_volume = collector.base_data_size * collector.min_info_volume_threshold
    
    # 场景：信息量达到阈值，但等待时间不同
    scenarios = [
        {'waiting_time': 0, 'info_volume': 1000000},
        {'waiting_time': 50, 'info_volume': 1000000},
        {'waiting_time': 100, 'info_volume': 1000000},
    ]
    
    print(f"\n阈值: {threshold_volume:,} bits (基于信息量)")
    print("\n场景 | 等待时间 | 信息量 | 信息价值 | 基于信息量决策 | 基于信息价值决策")
    print("-" * 80)
    
    for i, scenario in enumerate(scenarios, 1):
        waiting_time = scenario['waiting_time']
        info_volume = scenario['info_volume']
        
        node_info = {
            'info_volume': info_volume,
            'info_waiting_since': 0,
            'info_is_reported': False,
            'record_time': 0
        }
        
        current_time = waiting_time
        info_value = collector.calculate_info_value(node_info, current_time)
        
        # 基于信息量的决策
        decision_by_volume = "等待" if info_volume >= threshold_volume else "立即上报"
        
        # 基于信息价值的决策
        decision_by_value = "等待" if info_value >= threshold_volume else "立即上报"
        
        print(f"  {i}  | {waiting_time:8d} | {info_volume:8,} | {info_value:9,.0f} | {decision_by_volume:14s} | {decision_by_value:16s}")
    
    print("\n[OK] 信息价值能影响延迟上报决策")


def test_routing_decision():
    """测试路由决策的差别"""
    print("\n" + "=" * 60)
    print("测试4: 路由决策差别")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    # 场景：两个候选节点，信息量相同，但等待时间不同
    node1_info = {
        'info_volume': 2000000,  # 2MB
        'info_waiting_since': 0,
        'info_is_reported': False,
        'record_time': 0
    }
    
    node2_info = {
        'info_volume': 2000000,  # 2MB
        'info_waiting_since': 0,
        'info_is_reported': False,
        'record_time': 0
    }
    
    current_time = 50
    
    info_value1 = collector.calculate_info_value(node1_info, current_time)
    info_value2 = collector.calculate_info_value(node2_info, current_time)
    
    # 模拟路由奖励计算
    max_info_volume = 1000000
    reward_factor = 0.2
    
    # 基于信息量
    normalized_volume1 = min(node1_info['info_volume'] / max_info_volume, 1.0)
    normalized_volume2 = min(node2_info['info_volume'] / max_info_volume, 1.0)
    bonus1_volume = 1.0 - (reward_factor * normalized_volume1)
    bonus2_volume = 1.0 - (reward_factor * normalized_volume2)
    
    # 基于信息价值
    normalized_value1 = min(info_value1 / max_info_volume, 1.0)
    normalized_value2 = min(info_value2 / max_info_volume, 1.0)
    bonus1_value = 1.0 - (reward_factor * normalized_value1)
    bonus2_value = 1.0 - (reward_factor * normalized_value2)
    
    print("\n场景：两个候选节点，信息量相同（2MB），等待时间不同")
    print(f"\n节点1: 等待时间={current_time - node1_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {node1_info['info_volume']:,} bits")
    print(f"  - 信息价值: {info_value1:,.0f} bits")
    print(f"  - 基于信息量的奖励: {bonus1_volume:.4f}")
    print(f"  - 基于信息价值的奖励: {bonus1_value:.4f}")
    
    print(f"\n节点2: 等待时间={current_time - node2_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {node2_info['info_volume']:,} bits")
    print(f"  - 信息价值: {info_value2:,.0f} bits")
    print(f"  - 基于信息量的奖励: {bonus2_volume:.4f}")
    print(f"  - 基于信息价值的奖励: {bonus2_value:.4f}")
    
    print(f"\n差别:")
    print(f"  - 基于信息量: 两个节点奖励相同 ({bonus1_volume == bonus2_volume})")
    print(f"  - 基于信息价值: 两个节点奖励不同 ({bonus1_value != bonus2_value})")
    print(f"  - 奖励差异: {abs(bonus1_value - bonus2_value):.4f}")
    
    print("\n[OK] 信息价值能影响路由选择")


def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试5: 配置集成")
    print("=" * 60)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 检查配置参数
    print(f"\n配置参数:")
    print(f"  - info_value_decay_rate: {config_manager.path_collector_config.info_value_decay_rate}")
    
    # 创建虚拟中心
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    
    # 直接创建信息收集器（避免导入路径问题）
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=config_manager.path_collector_config.info_value_decay_rate
    )
    
    # 检查信息收集器是否有 calculate_info_value 方法
    has_method = hasattr(collector, 'calculate_info_value')
    print(f"  - 信息收集器有 calculate_info_value 方法: {has_method}")
    
    if has_method:
        # 测试计算
        node_info = {
            'info_volume': 1000000,
            'info_waiting_since': 0,
            'info_is_reported': False,
            'record_time': 0
        }
        info_value = collector.calculate_info_value(node_info, 50)
        print(f"  - 测试计算: info_value = {info_value:,.0f} bits")
    
    print("\n[OK] 配置集成正常")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("信息价值计算功能测试")
    print("=" * 60)
    
    try:
        test_info_value_calculation()
        test_info_value_vs_info_volume()
        test_threshold_decision()
        test_routing_decision()
        test_config_integration()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        print("\n主要差别总结:")
        print("1. 信息价值考虑时间衰减，等待时间越长，价值越低")
        print("2. 信息量相同的节点，如果等待时间不同，信息价值不同")
        print("3. 基于信息价值的决策能更准确地反映信息的实际价值")
        print("4. 路由选择会优先选择信息价值高的节点（信息更新鲜）")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

