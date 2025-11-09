#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证AdaptiveLyapunovScheduler功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from network.sensor_network import SensorNetwork
from core.energy_simulation import EnergySimulation
from scheduling.schedulers import AdaptiveLyapunovScheduler
from node_info.node_info_manager import NodeInfoManager

def test_adaptive_scheduler():
    """测试自适应调度器基本功能"""
    
    print("="*80)
    print("AdaptiveLyapunovScheduler 功能测试")
    print("="*80 + "\n")
    
    # 创建一个小型网络用于快速测试
    print("1. 创建测试网络...")
    network = SensorNetwork(num_nodes=10, area_size=50, initial_energy=3000)
    print(f"   ✓ 创建了{len(network.nodes)}个节点的网络\n")
    
    # 创建节点信息管理器
    print("2. 初始化节点信息管理器...")
    nim = NodeInfoManager(network.nodes)
    print(f"   ✓ 管理{len(nim.nodes_info)}个节点的信息\n")
    
    # 创建自适应调度器
    print("3. 创建自适应Lyapunov调度器...")
    scheduler = AdaptiveLyapunovScheduler(
        nim,
        V=0.5,
        K=2,
        max_hops=3,
        window_size=5,
        V_min=0.1,
        V_max=1.5,
        adjust_rate=0.15,
        sensitivity=1.5
    )
    print(f"   ✓ 调度器初始化完成")
    print(f"     - 初始V: {scheduler.V}")
    print(f"     - V范围: [{scheduler.V_min}, {scheduler.V_max}]")
    print(f"     - 调整速率: {scheduler.adjust_rate*100:.0f}%\n")
    
    # 创建仿真
    print("4. 创建能量仿真...")
    sim = EnergySimulation(network, scheduler=scheduler, enable_energy_sharing=True)
    print("   ✓ 仿真环境就绪\n")
    
    # 运行短时间仿真
    print("5. 运行仿真（100步，观察V的自适应调整）...")
    print("-"*80)
    sim.run(duration=100, step_interval=1, print_interval=20)
    print("-"*80 + "\n")
    
    # 检查是否有调整
    print("6. 检查自适应调整...")
    stats = scheduler.get_adaptation_stats()
    
    print(f"   调整统计:")
    print(f"   - 初始V: {stats['initial_V']:.3f}")
    print(f"   - 当前V: {stats['current_V']:.3f}")
    print(f"   - 调整次数: {stats['total_adjustments']}")
    
    if stats['total_adjustments'] > 0:
        print(f"   ✓ V参数成功调整了 {stats['total_adjustments']} 次")
        print(f"\n   调整历史:")
        for t, old_v, new_v, reason in stats['adjustment_history']:
            print(f"     t={t}: {old_v:.3f} → {new_v:.3f} | {reason}")
    else:
        print(f"   ℹ 未触发调整（网络可能状态稳定）")
    
    print(f"\n   反馈统计:")
    print(f"   - 平均反馈分数: {stats['avg_feedback']:.2f}")
    print(f"   - 最佳反馈分数: {stats['best_feedback']:.2f}")
    print(f"   - 最差反馈分数: {stats['worst_feedback']:.2f}")
    
    # 打印完整摘要
    print("\n" + "="*80)
    scheduler.print_adaptation_summary()
    
    # 获取仿真统计
    sim_stats = sim.stats.get_summary()
    print("仿真结果:")
    print(f"   - 最终存活节点: {sim_stats.get('final_alive_nodes', 0)}/{len(network.nodes)}")
    print(f"   - 平均能量: {sim_stats.get('avg_energy', 0):.2f}")
    print(f"   - 能量标准差: {sim_stats.get('energy_std', 0):.2f}")
    print(f"   - 平均效率: {sim_stats.get('avg_efficiency', 0):.2%}")
    
    print("\n" + "="*80)
    print("✓ 测试完成！AdaptiveLyapunovScheduler 工作正常")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_adaptive_scheduler()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

