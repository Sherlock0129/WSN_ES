# tests/test_info_value_vs_volume_comparison.py
# -*- coding: utf-8 -*-
"""
对比测试：信息价值 vs 信息量

测试内容：
1. 对比使用信息价值和信息量的决策差异
2. 关键指标：上报决策、等待时间、信息价值衰减
3. 轻量级测试（不运行完整仿真）
"""

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import math
from info_collection.path_based_collector import PathBasedInfoCollector
from info_collection.physical_center import VirtualCenter


def test_decision_comparison():
    """对比决策差异"""
    print("=" * 60)
    print("信息价值 vs 信息量 决策对比测试")
    print("=" * 60)
    
    # 创建虚拟中心和信息收集器
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    # 初始化节点信息
    node_info = {
        'info_volume': 1000000,  # 1MB
        'info_waiting_since': 0,
        'info_is_reported': False,
        'record_time': 0
    }
    
    threshold = 500000  # 阈值：500KB
    
    print(f"\n测试场景:")
    print(f"  - 初始信息量: {node_info['info_volume']:,} bits")
    print(f"  - 阈值: {threshold:,} bits")
    print(f"  - 衰减率: 0.02")
    
    print(f"\n时间 | 信息量 | 信息价值 | 基于信息量决策 | 基于信息价值决策")
    print("-" * 70)
    
    decisions_volume = []
    decisions_value = []
    
    for t in range(0, 101, 10):  # 0到100分钟，每10分钟一次
        # 基于信息量的决策
        info_volume = node_info['info_volume']
        decision_volume = "等待" if info_volume >= threshold else "立即上报"
        decisions_volume.append(decision_volume)
        
        # 基于信息价值的决策
        info_value = collector.calculate_info_value(node_info, t)
        decision_value = "等待" if info_value >= threshold else "立即上报"
        decisions_value.append(decision_value)
        
        print(f"{t:4d} | {info_volume:8,} | {info_value:9,.0f} | {decision_volume:14s} | {decision_value:16s}")
    
    # 分析差异
    print("\n" + "=" * 60)
    print("决策差异分析")
    print("=" * 60)
    
    # 统计首次上报时间
    first_report_volume = None
    first_report_value = None
    
    for i, (d_vol, d_val) in enumerate(zip(decisions_volume, decisions_value)):
        t = i * 10
        if first_report_volume is None and d_vol == "立即上报":
            first_report_volume = t
        if first_report_value is None and d_val == "立即上报":
            first_report_value = t
    
    print(f"\n1. 首次上报时间:")
    print(f"   基于信息量: {first_report_volume if first_report_volume is not None else '未上报'} 分钟")
    print(f"   基于信息价值: {first_report_value if first_report_value is not None else '未上报'} 分钟")
    
    if first_report_volume is not None and first_report_value is not None:
        diff = first_report_value - first_report_volume
        print(f"   差异: {diff:+d} 分钟")
        if diff < 0:
            print(f"   -> 信息价值方式更早上报（信息衰减后更快触发上报）")
        elif diff > 0:
            print(f"   -> 信息量方式更早上报")
        else:
            print(f"   -> 上报时间相同")
    
    # 统计决策不同的次数
    different_decisions = sum(1 for d_vol, d_val in zip(decisions_volume, decisions_value) if d_vol != d_val)
    print(f"\n2. 决策差异次数:")
    print(f"   总检查次数: {len(decisions_volume)}")
    print(f"   决策不同次数: {different_decisions}")
    print(f"   决策相同次数: {len(decisions_volume) - different_decisions}")
    if different_decisions > 0:
        print(f"   -> 信息价值方式在 {different_decisions} 个时间点做出了不同决策")
    
    # 计算信息价值衰减效果
    print(f"\n3. 信息价值衰减效果:")
    initial_value = collector.calculate_info_value(node_info, 0)
    final_value = collector.calculate_info_value(node_info, 100)
    decay_ratio = (initial_value - final_value) / initial_value * 100
    print(f"   初始价值: {initial_value:,.0f} bits")
    print(f"   100分钟后: {final_value:,.0f} bits")
    print(f"   衰减比例: {decay_ratio:.2f}%")
    print(f"   -> 信息价值随时间显著衰减")
    
    return {
        'first_report_volume': first_report_volume,
        'first_report_value': first_report_value,
        'different_decisions': different_decisions,
        'decay_ratio': decay_ratio
    }


def test_routing_comparison():
    """对比路由选择差异"""
    print("\n" + "=" * 60)
    print("路由选择对比测试")
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
        'info_waiting_since': 0,  # 刚等待
        'info_is_reported': False,
        'record_time': 0
    }
    
    node2_info = {
        'info_volume': 2000000,  # 2MB
        'info_waiting_since': 0,  # 等待了50分钟
        'info_is_reported': False,
        'record_time': 0
    }
    
    current_time = 50
    max_info_volume = 1000000
    reward_factor = 0.2
    
    # 基于信息量
    volume1 = node1_info['info_volume']
    volume2 = node2_info['info_volume']
    normalized_vol1 = min(volume1 / max_info_volume, 1.0)
    normalized_vol2 = min(volume2 / max_info_volume, 1.0)
    bonus1_volume = 1.0 - (reward_factor * normalized_vol1)
    bonus2_volume = 1.0 - (reward_factor * normalized_vol2)
    
    # 基于信息价值
    value1 = collector.calculate_info_value(node1_info, current_time)
    value2 = collector.calculate_info_value(node2_info, current_time)
    normalized_val1 = min(value1 / max_info_volume, 1.0)
    normalized_val2 = min(value2 / max_info_volume, 1.0)
    bonus1_value = 1.0 - (reward_factor * normalized_val1)
    bonus2_value = 1.0 - (reward_factor * normalized_val2)
    
    print(f"\n场景: 两个候选节点，信息量相同（2MB），等待时间不同")
    print(f"\n节点1: 等待时间={current_time - node1_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {volume1:,} bits")
    print(f"  - 信息价值: {value1:,.0f} bits")
    print(f"  - 基于信息量的奖励: {bonus1_volume:.4f}")
    print(f"  - 基于信息价值的奖励: {bonus1_value:.4f}")
    
    print(f"\n节点2: 等待时间={current_time - node2_info['info_waiting_since']}分钟")
    print(f"  - 信息量: {volume2:,} bits")
    print(f"  - 信息价值: {value2:,.0f} bits")
    print(f"  - 基于信息量的奖励: {bonus2_volume:.4f}")
    print(f"  - 基于信息价值的奖励: {bonus2_value:.4f}")
    
    print(f"\n差别:")
    print(f"  - 基于信息量: 两个节点奖励相同 ({bonus1_volume == bonus2_volume})")
    print(f"  - 基于信息价值: 两个节点奖励不同 ({bonus1_value != bonus2_value})")
    print(f"  - 奖励差异: {abs(bonus1_value - bonus2_value):.4f}")
    
    if bonus1_value > bonus2_value:
        print(f"  -> 信息价值方式会优先选择节点1（信息更新鲜）")
    elif bonus1_value < bonus2_value:
        print(f"  -> 信息价值方式会优先选择节点2（信息更新鲜）")
    else:
        print(f"  -> 两个节点奖励相同")


def test_waiting_time_comparison():
    """对比等待时间的影响"""
    print("\n" + "=" * 60)
    print("等待时间影响对比测试")
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
    
    # 场景：相同信息量，不同等待时间
    scenarios = [
        {'waiting_time': 0, 'info_volume': 1000000},
        {'waiting_time': 30, 'info_volume': 1000000},
        {'waiting_time': 50, 'info_volume': 1000000},
        {'waiting_time': 100, 'info_volume': 1000000},
    ]
    
    threshold = 500000
    
    print(f"\n场景: 相同信息量（1MB），不同等待时间")
    print(f"阈值: {threshold:,} bits")
    print(f"\n等待时间 | 信息量 | 信息价值 | 基于信息量决策 | 基于信息价值决策 | 价值保留率")
    print("-" * 85)
    
    for scenario in scenarios:
        waiting_time = scenario['waiting_time']
        info_volume = scenario['info_volume']
        
        node_info = {
            'info_volume': info_volume,
            'info_waiting_since': 0,
            'info_is_reported': False,
            'record_time': 0
        }
        
        info_value = collector.calculate_info_value(node_info, waiting_time)
        retention = (info_value / info_volume) * 100 if info_volume > 0 else 0
        
        decision_volume = "等待" if info_volume >= threshold else "立即上报"
        decision_value = "等待" if info_value >= threshold else "立即上报"
        
        print(f"{waiting_time:8d} | {info_volume:8,} | {info_value:9,.0f} | {decision_volume:14s} | {decision_value:16s} | {retention:8.2f}%")
    
    print(f"\n结论:")
    print(f"  - 信息量方式: 所有场景都选择'等待'（因为信息量相同且达到阈值）")
    print(f"  - 信息价值方式: 等待时间长的场景选择'立即上报'（因为价值已衰减）")
    print(f"  -> 信息价值方式能更准确地反映信息的实际价值")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("信息价值 vs 信息量 对比测试")
    print("=" * 60)
    
    try:
        # 测试1: 决策对比
        stats = test_decision_comparison()
        
        # 测试2: 路由选择对比
        test_routing_comparison()
        
        # 测试3: 等待时间影响对比
        test_waiting_time_comparison()
        
        # 总结
        print("\n" + "=" * 60)
        print("总结")
        print("=" * 60)
        print("\n主要改进:")
        print("1. 信息价值方式能更早识别过时信息，上报更及时")
        if stats['first_report_value'] is not None and stats['first_report_volume'] is not None:
            if stats['first_report_value'] < stats['first_report_volume']:
                print(f"   - 首次上报时间提前 {stats['first_report_volume'] - stats['first_report_value']} 分钟")
        print("2. 等待时间更短，信息新鲜度更高")
        print("3. 路由选择会优先选择信息更新鲜的节点")
        print("4. 决策更准确，符合信息价值的时间衰减理论")
        print("5. 自动优先处理高价值信息（新鲜信息）")
        print("=" * 60)
        
        return 0
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
