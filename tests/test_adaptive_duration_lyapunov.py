"""
测试自适应传输时长Lyapunov调度器 (AdaptiveDurationLyapunovScheduler)

核心特点：
1. 纯粹的Lyapunov能量优化
2. 自适应选择传输时长（1-5分钟）
3. 不考虑AoI和信息量
4. 得分函数：delivered × Q[r] - V × loss
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.schedulers import LyapunovScheduler, AdaptiveDurationLyapunovScheduler


def test_adaptive_duration_lyapunov():
    """测试自适应传输时长Lyapunov调度器"""
    
    print("=" * 80)
    print("自适应传输时长Lyapunov调度器测试")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    
    # 基础仿真参数
    config.simulation_config.time_steps = 300
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    # 网络参数
    config.network_config.num_nodes = 15
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    # 节点能量参数
    config.node_config.energy_char = 500.0
    config.node_config.capacity = 3.5
    config.node_config.voltage = 3.7
    
    # 调度器参数
    config.scheduler_config.lyapunov_V = 0.5
    config.scheduler_config.K = 2
    config.scheduler_config.max_hops = 3
    
    # 传输时长参数
    config.scheduler_config.duration_min = 1
    config.scheduler_config.duration_max = 5
    
    # 2. 创建网络
    print("\n[1] 创建网络...")
    network = config.create_network()
    print(f"  [OK] 网络节点数: {len(network.nodes)}")
    print(f"  [OK] 物理中心: 节点 0")
    
    # 3. 创建调度器
    print("\n[2] 创建自适应传输时长Lyapunov调度器...")
    from info_collection.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False,
        history_size=1000
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    scheduler_adaptive = AdaptiveDurationLyapunovScheduler(
        node_info_manager=nim,
        V=config.scheduler_config.lyapunov_V,
        K=config.scheduler_config.K,
        max_hops=config.scheduler_config.max_hops,
        min_duration=config.scheduler_config.duration_min,
        max_duration=config.scheduler_config.duration_max
    )
    
    print(f"  [OK] 调度器类型: AdaptiveDurationLyapunovScheduler")
    print(f"  [OK] Lyapunov V: {config.scheduler_config.lyapunov_V}")
    print(f"  [OK] 传输时长范围: {config.scheduler_config.duration_min}-{config.scheduler_config.duration_max} 分钟")
    print(f"  [OK] 得分函数: delivered × Q[r] - V × loss")
    
    # 4. 创建标准Lyapunov调度器用于对比
    print("\n[3] 创建标准Lyapunov调度器（对比）...")
    scheduler_standard = LyapunovScheduler(
        node_info_manager=nim,
        V=config.scheduler_config.lyapunov_V,
        K=config.scheduler_config.K,
        max_hops=config.scheduler_config.max_hops
    )
    print(f"  [OK] 调度器类型: LyapunovScheduler（固定1分钟）")
    
    # 5. 运行一次规划测试
    print("\n[4] 测试自适应传输时长优化...")
    print("-" * 80)
    
    # 模拟节点能量差异
    for i, node in enumerate(network.nodes[1:11], 1):
        if i % 3 == 0:
            node.current_energy = 5000  # 极低能量（需要接收）
        elif i % 3 == 1:
            node.current_energy = 50000  # 高能量（可以捐献）
        else:
            node.current_energy = 20000  # 中等能量
    
    # 更新InfoNode
    for node in network.nodes:
        if node.node_id in nim.latest_info:
            nim.latest_info[node.node_id]['energy'] = node.current_energy
            if node.node_id in nim.info_nodes:
                nim.info_nodes[node.node_id].current_energy = node.current_energy
    
    nim.get_info_nodes()
    
    # 打印节点能量分布
    print("\n节点能量分布：")
    for node in network.nodes[1:11]:
        print(f"  节点 {node.node_id}: {node.current_energy:.0f}J")
    
    # 规划传输（自适应时长）
    plans_adaptive, _ = scheduler_adaptive.plan(network, t=0)
    
    print(f"\n自适应时长Lyapunov调度器规划结果：")
    print(f"  [OK] 规划数量: {len(plans_adaptive)}")
    
    if plans_adaptive:
        print("\n  详细计划：")
        for i, plan in enumerate(plans_adaptive[:5], 1):
            donor_id = plan['donor'].node_id
            receiver_id = plan['receiver'].node_id
            duration = plan.get('duration', 1)
            delivered = plan.get('delivered', 0)
            loss = plan.get('loss', 0)
            score = plan.get('score', 0)
            
            print(f"\n  计划 {i}:")
            print(f"    Donor {donor_id} → Receiver {receiver_id}")
            print(f"    传输时长: {duration} 分钟")
            print(f"    能量传输: {duration * 500:.0f}J (送达: {delivered:.0f}J, 损耗: {loss:.0f}J)")
            print(f"    Lyapunov得分: {score:.2f}")
    
    # 规划传输（标准Lyapunov）
    print("\n" + "-" * 80)
    plans_standard, _ = scheduler_standard.plan(network, t=0)
    
    print(f"\n标准Lyapunov调度器规划结果（对比）：")
    print(f"  [OK] 规划数量: {len(plans_standard)}")
    
    if plans_standard:
        print("\n  详细计划（固定1分钟传输）：")
        for i, plan in enumerate(plans_standard[:5], 1):
            donor_id = plan['donor'].node_id
            receiver_id = plan['receiver'].node_id
            delivered = plan.get('delivered', 0)
            loss = plan.get('loss', 0)
            
            print(f"\n  计划 {i}:")
            print(f"    Donor {donor_id} → Receiver {receiver_id}")
            print(f"    传输时长: 1 分钟（固定）")
            print(f"    能量传输: 500J (送达: {delivered:.0f}J, 损耗: {loss:.0f}J)")
    
    # 6. 对比分析
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    if plans_adaptive:
        durations = [p.get('duration', 1) for p in plans_adaptive]
        total_energy_adaptive = sum([p.get('duration', 1) * 500 for p in plans_adaptive])
        total_delivered_adaptive = sum([p.get('delivered', 0) for p in plans_adaptive])
        total_loss_adaptive = sum([p.get('loss', 0) for p in plans_adaptive])
        avg_duration = np.mean(durations)
        efficiency_adaptive = total_delivered_adaptive / total_energy_adaptive if total_energy_adaptive > 0 else 0
        
        print(f"\n自适应时长Lyapunov调度器：")
        print(f"  - 传输计划数: {len(plans_adaptive)}")
        print(f"  - 平均传输时长: {avg_duration:.2f} 分钟")
        print(f"  - 传输时长分布: {dict(zip(*np.unique(durations, return_counts=True)))}")
        print(f"  - 总传输能量: {total_energy_adaptive:.0f}J")
        print(f"  - 总送达能量: {total_delivered_adaptive:.0f}J")
        print(f"  - 总损耗能量: {total_loss_adaptive:.0f}J")
        print(f"  - 平均效率: {efficiency_adaptive:.2%}")
    
    if plans_standard:
        total_energy_standard = len(plans_standard) * 500
        total_delivered_standard = sum([p.get('delivered', 0) for p in plans_standard])
        total_loss_standard = sum([p.get('loss', 0) for p in plans_standard])
        efficiency_standard = total_delivered_standard / total_energy_standard if total_energy_standard > 0 else 0
        
        print(f"\n标准Lyapunov调度器：")
        print(f"  - 传输计划数: {len(plans_standard)}")
        print(f"  - 平均传输时长: 1.00 分钟（固定）")
        print(f"  - 总传输能量: {total_energy_standard:.0f}J")
        print(f"  - 总送达能量: {total_delivered_standard:.0f}J")
        print(f"  - 总损耗能量: {total_loss_standard:.0f}J")
        print(f"  - 平均效率: {efficiency_standard:.2%}")
    
    # 7. 分析传输时长选择的逻辑
    print("\n" + "=" * 80)
    print("传输时长选择分析")
    print("=" * 80)
    
    if plans_adaptive:
        print("\n时长选择逻辑：")
        print("  - 高效率路径（η > 0.4）：倾向选择更长时长")
        print("  - 低效率路径（η < 0.2）：倾向选择更短时长")
        print("  - Q值大的receiver：倾向选择更长时长（能量缺口大）")
        print("  - 优化目标：最大化 delivered × Q[r] - V × loss")
        
        print("\n实际选择情况：")
        for plan in plans_adaptive[:5]:
            donor_id = plan['donor'].node_id
            receiver_id = plan['receiver'].node_id
            duration = plan.get('duration', 1)
            delivered = plan.get('delivered', 0)
            loss = plan.get('loss', 0)
            energy_sent = duration * 500
            eta = delivered / energy_sent if energy_sent > 0 else 0
            
            print(f"  计划: D{donor_id}→R{receiver_id}, 时长={duration}分钟, η={eta:.2%}")
    
    # 8. 完整仿真测试
    print("\n" + "=" * 80)
    print("完整仿真测试（100分钟）")
    print("=" * 80)
    
    # 重置网络
    network = config.create_network()
    
    # 创建仿真
    simulation = EnergySimulation(
        network=network,
        time_steps=100,
        scheduler=scheduler_adaptive,
        enable_energy_sharing=True,
        passive_mode=True,
        check_interval=10
    )
    
    print("\n开始仿真...")
    simulation.simulate()
    
    print("\n仿真完成！")
    
    # 统计结果
    results = simulation.result_manager.get_results()
    if results:
        print(f"\n仿真统计：")
        print(f"  - 总时间步: {len(results)}")
        
        # 统计传输时长分布
        all_durations = []
        for step_result in results:
            if 'plans' in step_result and step_result['plans']:
                for plan in step_result['plans']:
                    duration = plan.get('duration', 1)
                    all_durations.append(duration)
        
        if all_durations:
            duration_dist = dict(zip(*np.unique(all_durations, return_counts=True)))
            print(f"  - 总传输次数: {len(all_durations)}")
            print(f"  - 传输时长分布: {duration_dist}")
            print(f"  - 平均传输时长: {np.mean(all_durations):.2f} 分钟")
            print(f"  - 总传输能量: {sum([d * 500 for d in all_durations]):.0f}J")
            
            # 按时长统计
            print(f"\n  按时长统计：")
            for dur in sorted(duration_dist.keys()):
                count = duration_dist[dur]
                percentage = count / len(all_durations) * 100
                print(f"    {dur}分钟: {count}次 ({percentage:.1f}%)")
    
    print("\n[SUCCESS] 测试完成！")
    print("\n核心特点总结：")
    print("  1. 纯粹的Lyapunov能量优化，不考虑AoI和信息量")
    print("  2. 自适应选择传输时长（1-5分钟）")
    print("  3. 得分函数：delivered × Q[r] - V × loss")
    print("  4. 根据路径效率和能量缺口智能选择时长")


if __name__ == "__main__":
    test_adaptive_duration_lyapunov()

