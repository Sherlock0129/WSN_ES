"""
测试传输时长感知调度器 (DurationAwareLyapunovScheduler)

功能测试：
1. 传输时长优化（1-5分钟）
2. 能量-AoI-信息量的多目标权衡
3. 信息搭便车机制
4. 与标准Lyapunov调度器对比
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from config.simulation_config import ConfigManager
from core.network import Network
from core.energy_simulation import EnergySimulation
from scheduling.schedulers import LyapunovScheduler, DurationAwareLyapunovScheduler


def test_duration_aware_scheduler():
    """测试传输时长感知调度器"""
    
    print("=" * 80)
    print("传输时长感知调度器测试")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    
    # 基础仿真参数
    config.simulation_config.time_steps = 300  # 测试5小时
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    # 网络参数
    config.network_config.num_nodes = 15
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    # 节点能量参数
    config.node_config.energy_char = 500.0  # 每分钟可传输500J
    config.node_config.capacity = 3.5
    config.node_config.voltage = 3.7
    
    # 调度器参数（传输时长优化）
    config.scheduler_config.scheduler_type = "duration_aware"
    config.scheduler_config.lyapunov_V = 0.5
    config.scheduler_config.K = 2
    config.scheduler_config.max_hops = 3
    
    # 传输时长参数
    config.scheduler_config.duration_min = 1
    config.scheduler_config.duration_max = 5
    config.scheduler_config.duration_w_aoi = 0.1   # AoI惩罚权重
    config.scheduler_config.duration_w_info = 0.05  # 信息量奖励权重
    config.scheduler_config.duration_info_rate = 10000.0  # 信息采集速率（bits/分钟）
    
    # 启用机会主义信息传递
    config.eetor_config.enable_info_aware_routing = True
    config.eetor_config.info_reward_factor = 0.3
    
    config.path_collector_config.enable_opportunistic_info_forwarding = True
    config.path_collector_config.enable_delayed_reporting = True
    config.path_collector_config.max_wait_time = 10
    
    # 2. 创建网络
    print("\n[1] 创建网络...")
    network = config.create_network()
    print(f"  [OK] 网络节点数: {len(network.nodes)}")
    print(f"  [OK] 物理中心: 节点 0")
    
    # 3. 创建调度器（传输时长感知）
    print("\n[2] 创建传输时长感知调度器...")
    from acdr.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False,
        history_size=1000
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    scheduler_duration = DurationAwareLyapunovScheduler(
        node_info_manager=nim,
        V=config.scheduler_config.lyapunov_V,
        K=config.scheduler_config.K,
        max_hops=config.scheduler_config.max_hops,
        min_duration=config.scheduler_config.duration_min,
        max_duration=config.scheduler_config.duration_max,
        w_aoi=config.scheduler_config.duration_w_aoi,
        w_info=config.scheduler_config.duration_w_info,
        info_collection_rate=config.scheduler_config.duration_info_rate
    )
    
    print(f"  [OK] 调度器类型: DurationAwareLyapunovScheduler")
    print(f"  [OK] 传输时长范围: {config.scheduler_config.duration_min}-{config.scheduler_config.duration_max} 分钟")
    print(f"  [OK] AoI权重: {config.scheduler_config.duration_w_aoi}")
    print(f"  [OK] 信息量权重: {config.scheduler_config.duration_w_info}")
    
    # 4. 创建标准Lyapunov调度器用于对比
    print("\n[3] 创建标准Lyapunov调度器（对比）...")
    scheduler_standard = LyapunovScheduler(
        node_info_manager=nim,
        V=config.scheduler_config.lyapunov_V,
        K=config.scheduler_config.K,
        max_hops=config.scheduler_config.max_hops
    )
    print(f"  [OK] 调度器类型: LyapunovScheduler（标准）")
    
    # 5. 运行一次规划测试
    print("\n[4] 测试传输时长优化...")
    print("-" * 80)
    
    # 模拟一些节点能量差异（更明显的差异）
    for i, node in enumerate(network.nodes[1:11], 1):
        if i % 3 == 0:
            node.current_energy = 5000  # 极低能量节点（需要接收）
        elif i % 3 == 1:
            node.current_energy = 50000  # 高能量节点（可以捐献）
        else:
            node.current_energy = 20000  # 中等能量节点
    
    # 更新InfoNode的能量
    for node in network.nodes:
        if node.node_id in nim.latest_info:
            nim.latest_info[node.node_id]['energy'] = node.current_energy
            # 同步到InfoNode
            if node.node_id in nim.info_nodes:
                nim.info_nodes[node.node_id].current_energy = node.current_energy
    
    # 模拟一些节点有未上报信息
    for node_id in [3, 5, 7]:
        if node_id in nim.latest_info:
            nim.latest_info[node_id]['info_volume'] = 50000  # 50kb信息
            nim.latest_info[node_id]['info_is_reported'] = False
            nim.latest_info[node_id]['info_waiting_since'] = 0
    
    # 强制刷新InfoNode
    nim.get_info_nodes()
    
    # 打印节点能量分布用于调试
    print("\n节点能量分布：")
    for node in network.nodes[1:11]:
        print(f"  节点 {node.node_id}: {node.current_energy:.0f}J")
    
    # 规划传输（传输时长感知）
    plans_duration, _ = scheduler_duration.plan(network, t=0)
    
    print(f"\n传输时长感知调度器规划结果：")
    print(f"  [OK] 规划数量: {len(plans_duration)}")
    
    if plans_duration:
        print("\n  详细计划：")
        for i, plan in enumerate(plans_duration[:5], 1):
            donor_id = plan['donor'].node_id
            receiver_id = plan['receiver'].node_id
            duration = plan.get('duration', 1)
            delivered = plan.get('delivered', 0)
            loss = plan.get('loss', 0)
            aoi_cost = plan.get('aoi_cost', 0)
            info_gain = plan.get('info_gain', 0)
            score = plan.get('score', 0)
            
            print(f"\n  计划 {i}:")
            print(f"    Donor {donor_id} → Receiver {receiver_id}")
            print(f"    传输时长: {duration} 分钟")
            print(f"    能量传输: {duration * 500:.0f}J (送达: {delivered:.0f}J, 损耗: {loss:.0f}J)")
            print(f"    AoI代价: {aoi_cost:.1f} 分钟")
            print(f"    信息收益: {info_gain:.0f} bits")
            print(f"    综合得分: {score:.2f}")
    
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
    
    if plans_duration:
        durations = [p.get('duration', 1) for p in plans_duration]
        total_energy_duration = sum([p.get('duration', 1) * 500 for p in plans_duration])
        avg_duration = np.mean(durations)
        
        print(f"\n传输时长感知调度器：")
        print(f"  - 传输计划数: {len(plans_duration)}")
        print(f"  - 平均传输时长: {avg_duration:.2f} 分钟")
        print(f"  - 传输时长分布: {dict(zip(*np.unique(durations, return_counts=True)))}")
        print(f"  - 总传输能量: {total_energy_duration:.0f}J")
    
    if plans_standard:
        total_energy_standard = len(plans_standard) * 500
        print(f"\n标准Lyapunov调度器：")
        print(f"  - 传输计划数: {len(plans_standard)}")
        print(f"  - 平均传输时长: 1.00 分钟（固定）")
        print(f"  - 总传输能量: {total_energy_standard:.0f}J")
    
    # 7. 完整仿真测试
    print("\n" + "=" * 80)
    print("完整仿真测试（100分钟）")
    print("=" * 80)
    
    # 重置网络
    network = config.create_network()
    
    # 创建仿真
    simulation = EnergySimulation(
        network=network,
        time_steps=100,
        scheduler=scheduler_duration,
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
    
    print("\n[SUCCESS] 测试完成！")


if __name__ == "__main__":
    test_duration_aware_scheduler()

