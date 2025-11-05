
"""
DQN推理模式使用示例（已修复传能过于频繁的问题）
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.dqn_scheduler import DQNScheduler
from info_collection.physical_center import NodeInfoManager


def run_dqn_inference(model_path="dqn_model.pth", time_steps=10080):
    """
    使用训练好的DQN模型运行推理（已修复传能过频问题）
    
    修复要点：
    1. training_mode=False（关闭训练模式）
    2. epsilon=0.0（无探索）
    3. passive_mode=True（控制传输频率）
    """
    print("=" * 80)
    print("DQN推理模式（已修复传能过频问题）")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    config.simulation_config.time_steps = time_steps
    config.simulation_config.enable_energy_sharing = True
    config.network_config.num_nodes = 15
    config.network_config.enable_physical_center = True
    
    # 2. 创建网络
    print("\n[1] 创建网络...")
    network = config.create_network()
    print(f"  ✓ 节点数: {len(network.nodes)}")
    
    # 3. 创建节点信息管理器
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    # 4. 创建DQN调度器（推理模式）
    print("\n[2] 创建DQN调度器（推理模式）...")
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,
        training_mode=False,      # ← 关键修复1: 关闭训练模式
        epsilon_start=0.0,        # ← 关键修复2: 无探索
        epsilon_end=0.0
    )
    print("  ✓ 训练模式: False")
    print("  ✓ 探索率: 0.0 (无探索)")
    
    # 5. 初始化并加载模型
    print("\n[3] 加载训练好的模型...")
    scheduler.plan(network, 0)  # 初始化agent
    scheduler.load_model(model_path)
    
    # 6. 强制设置epsilon为0（双重保险）
    scheduler.agent.epsilon = 0.0
    print(f"  ✓ 模型已加载: {model_path}")
    print(f"  ✓ 当前epsilon: {scheduler.agent.epsilon}")
    
    # 7. 运行仿真（使用被动模式）
    print("\n[4] 运行仿真...")
    print(f"  - 仿真步数: {time_steps}")
    print(f"  - 被动模式: True (控制传输频率)")  # ← 关键修复3
    print(f"  - 检查间隔: 10分钟")
    
    simulation = EnergySimulation(
        network=network,
        time_steps=time_steps,
        scheduler=scheduler,
        enable_energy_sharing=True,
        passive_mode=True,        # ← 关键修复3: 启用被动模式
        check_interval=10         # ← 每10分钟检查一次，而非每分钟
    )
    
    print("-" * 80)
    simulation.simulate()
    print("-" * 80)
    
    # 8. 统计结果
    print("\n[5] 仿真统计:")
    results = simulation.result_manager.get_results()
    
    # 统计传输次数和时长
    total_transfers = 0
    total_energy_sent = 0
    durations = []
    
    for result in results:
        if 'plans' in result and result['plans']:
            total_transfers += len(result['plans'])
            for plan in result['plans']:
                duration = plan.get('duration', 1)
                durations.append(duration)
                total_energy_sent += duration * 500  # E_char=500
    
    import numpy as np
    print(f"  - 总传输次数: {total_transfers}")
    if durations:
        print(f"  - 平均传输时长: {np.mean(durations):.2f} 分钟")
        print(f"  - 时长范围: {min(durations)}-{max(durations)} 分钟")
        print(f"  - 总传输能量: {total_energy_sent:.0f}J")
    
    # 能量统计
    final_energies = [node.current_energy for node in network.nodes[1:]]
    alive_nodes = sum(1 for e in final_energies if e > 0)
    print(f"\n  - 存活节点: {alive_nodes}/{len(final_energies)}")
    print(f"  - 平均能量: {np.mean(final_energies):.0f}J")
    print(f"  - 能量标准差: {np.std(final_energies):.0f}J")
    
    print("\n" + "=" * 80)
    print("✅ 推理完成！")
    print("=" * 80)
    
    if alive_nodes < len(final_energies):
        print("\n⚠️ 警告: 仍有节点死亡")
        print("建议:")
        print("1. 增加check_interval (如15或20分钟)")
        print("2. 检查训练质量（可能需要更多训练回合）")
        print("3. 调整K值（减少并发传输）")
    else:
        print("\n✅ 所有节点存活，DQN工作正常！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN推理模式（已修复）')
    parser.add_argument('--model', type=str, default='dqn_model.pth',
                       help='模型路径')
    parser.add_argument('--steps', type=int, default=10080,
                       help='仿真步数（默认10080=7天）')
    
    args = parser.parse_args()
    
    run_dqn_inference(model_path=args.model, time_steps=args.steps)
