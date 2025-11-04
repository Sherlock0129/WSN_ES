"""
使用DQN调度器运行完整仿真

快速开始：
1. 训练DQN：python run_dqn_simulation.py --train
2. 测试DQN：python run_dqn_simulation.py --test
3. 完整运行：python run_dqn_simulation.py --full
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import argparse
import numpy as np
from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.dqn_scheduler import DQNScheduler
from acdr.physical_center import NodeInfoManager


def create_dqn_config(test_mode=False):
    """创建DQN仿真配置"""
    config = ConfigManager()
    
    # 基础配置
    if test_mode:
        config.simulation_config.time_steps = 200  # 测试模式：200步
    else:
        config.simulation_config.time_steps = 100  # 训练模式：100步/回合
    
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    # 网络配置
    config.network_config.num_nodes = 15
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    # 节点配置
    config.node_config.energy_char = 500.0
    config.node_config.capacity = 3.5
    config.node_config.voltage = 3.7
    
    return config


def train_dqn(episodes=50, save_path="dqn_model.pth"):
    """
    训练DQN调度器
    
    :param episodes: 训练回合数
    :param save_path: 模型保存路径
    """
    print("=" * 80)
    print("DQN深度强化学习调度器训练")
    print("=" * 80)
    
    # 创建配置
    config = create_dqn_config(test_mode=False)
    
    # 创建节点信息管理器
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    
    # 创建DQN调度器（训练模式）
    print("\n[1] 创建DQN调度器...")
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,          # 10个离散动作（1-10分钟）
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        training_mode=True      # 训练模式
    )
    
    print(f"  [OK] 动作空间: 10个离散动作（1-10分钟）")
    print(f"  [OK] 初始探索率: {scheduler.epsilon_start}")
    print(f"  [OK] 学习率: {scheduler.lr}")
    
    # 训练循环
    print(f"\n[2] 开始训练（{episodes}回合）...")
    print("-" * 80)
    
    for episode in range(episodes):
        # 创建新网络
        network = config.create_network()
        nim.initialize_node_info(network.nodes, initial_time=0)
        scheduler.nim = nim
        
        # 重置调度器状态
        scheduler.prev_state = None
        scheduler.prev_action = None
        scheduler.episode_count = episode + 1
        
        # 运行仿真（静默模式）
        simulation = EnergySimulation(
            network=network,
            time_steps=config.simulation_config.time_steps,
            scheduler=scheduler,
            enable_energy_sharing=True,
            passive_mode=False
        )
        
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        simulation.simulate()
        sys.stdout = old_stdout
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            stats = scheduler.get_training_stats()
            print(f"  回合 {episode + 1}/{episodes}: "
                  f"Loss={stats['avg_loss']:.4f}, "
                  f"ε={stats['epsilon']:.4f}, "
                  f"Buffer={stats['buffer_size']}")
    
    print("-" * 80)
    print("\n[3] 训练完成！")
    
    # 保存模型
    scheduler.save_model(save_path)
    print(f"\n[SUCCESS] 模型已保存到: {save_path}")
    
    return scheduler


def test_dqn(model_path="dqn_model.pth"):
    """
    测试训练好的DQN调度器
    
    :param model_path: 模型路径
    """
    print("\n" + "=" * 80)
    print("DQN调度器测试")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        print("请先运行训练：python run_dqn_simulation.py --train")
        return
    
    # 创建配置
    config = create_dqn_config(test_mode=True)
    
    # 创建网络
    print("\n[1] 创建测试网络...")
    network = config.create_network()
    print(f"  [OK] 网络节点数: {len(network.nodes)}")
    
    # 创建节点信息管理器
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    # 创建DQN调度器（测试模式）
    print("\n[2] 创建DQN调度器（测试模式）...")
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,
        training_mode=False  # 测试模式（不探索）
    )
    
    # 初始化agent并加载模型
    scheduler.plan(network, 0)  # 初始化
    scheduler.load_model(model_path)
    print(f"  [OK] 模型已加载: {model_path}")
    
    # 运行仿真
    print("\n[3] 运行仿真（200步）...")
    print("-" * 80)
    
    simulation = EnergySimulation(
        network=network,
        time_steps=200,
        scheduler=scheduler,
        enable_energy_sharing=True,
        passive_mode=False
    )
    
    simulation.simulate()
    
    print("-" * 80)
    
    # 统计结果
    print("\n[4] 仿真统计:")
    results = simulation.result_manager.get_results()
    
    # 统计传输时长分布
    durations = []
    for result in results:
        if 'plans' in result and result['plans']:
            for plan in result['plans']:
                durations.append(plan.get('duration', 1))
    
    if durations:
        duration_dist = dict(zip(*np.unique(durations, return_counts=True)))
        
        print(f"\n  传输统计:")
        print(f"  - 总传输次数: {len(durations)}")
        print(f"  - 平均传输时长: {np.mean(durations):.2f} 分钟")
        print(f"  - 总传输能量: {sum([d * 500 for d in durations]):.0f}J")
        
        print(f"\n  动作使用频率:")
        for dur in sorted(duration_dist.keys()):
            count = duration_dist[dur]
            percentage = count / len(durations) * 100
            bar = '█' * int(percentage / 2)
            print(f"    {dur:2d}分钟: {bar} {count:3d}次 ({percentage:5.1f}%)")
    
    # 能量统计
    final_energies = [node.current_energy for node in network.nodes[1:]]
    print(f"\n  能量统计:")
    print(f"  - 平均能量: {np.mean(final_energies):.0f}J")
    print(f"  - 能量标准差: {np.std(final_energies):.0f}J")
    print(f"  - 能量CV: {np.std(final_energies)/np.mean(final_energies):.4f}")
    
    print("\n[SUCCESS] 测试完成！")


def run_full_simulation(model_path="dqn_model.pth", time_steps=1000):
    """
    运行完整的长时间仿真
    
    :param model_path: 模型路径
    :param time_steps: 仿真步数
    """
    print("\n" + "=" * 80)
    print(f"完整仿真（{time_steps}步）")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        print("请先运行训练：python run_dqn_simulation.py --train")
        return
    
    # 创建配置
    config = ConfigManager()
    config.simulation_config.time_steps = time_steps
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    config.network_config.num_nodes = 20  # 更大的网络
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    # 创建网络
    print("\n[1] 创建网络...")
    network = config.create_network()
    print(f"  [OK] 网络节点数: {len(network.nodes)}")
    
    # 创建节点信息管理器
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    # 创建DQN调度器
    print("\n[2] 创建DQN调度器...")
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,
        training_mode=False
    )
    
    scheduler.plan(network, 0)
    scheduler.load_model(model_path)
    print(f"  [OK] 模型已加载: {model_path}")
    
    # 运行仿真
    print(f"\n[3] 运行仿真（{time_steps}步）...")
    print("-" * 80)
    
    simulation = EnergySimulation(
        network=network,
        time_steps=time_steps,
        scheduler=scheduler,
        enable_energy_sharing=True,
        passive_mode=True,  # 启用智能被动传能
        check_interval=10
    )
    
    simulation.simulate()
    
    print("-" * 80)
    print("\n[SUCCESS] 完整仿真完成！")
    print(f"\n结果保存在: data/ 目录")


def main():
    parser = argparse.ArgumentParser(
        description='使用DQN调度器运行WSN仿真',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练DQN（推荐50回合，约3小时）
  python run_dqn_simulation.py --train --episodes 50
  
  # 快速训练（20回合，约1小时）
  python run_dqn_simulation.py --train --episodes 20
  
  # 测试DQN
  python run_dqn_simulation.py --test
  
  # 运行完整仿真（7天）
  python run_dqn_simulation.py --full --steps 10080
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='训练DQN调度器')
    parser.add_argument('--test', action='store_true',
                       help='测试训练好的DQN调度器')
    parser.add_argument('--full', action='store_true',
                       help='运行完整长时间仿真')
    parser.add_argument('--episodes', type=int, default=50,
                       help='训练回合数（默认50）')
    parser.add_argument('--steps', type=int, default=1000,
                       help='完整仿真步数（默认1000）')
    parser.add_argument('--model', type=str, default='dqn_model.pth',
                       help='模型文件路径（默认dqn_model.pth）')
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助
    if not (args.train or args.test or args.full):
        parser.print_help()
        return
    
    # 执行相应操作
    if args.train:
        train_dqn(episodes=args.episodes, save_path=args.model)
    
    if args.test:
        test_dqn(model_path=args.model)
    
    if args.full:
        run_full_simulation(model_path=args.model, time_steps=args.steps)


if __name__ == "__main__":
    main()

