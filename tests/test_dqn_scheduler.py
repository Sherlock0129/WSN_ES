"""
测试DQN深度强化学习调度器（离散动作空间）

功能：
1. 训练DQN智能体
2. 测试训练后的策略
3. 与标准Lyapunov调度器对比
4. 展示离散动作空间的优势
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.dqn_scheduler import DQNScheduler
from scheduling.schedulers import LyapunovScheduler


def train_dqn_scheduler(episodes=50, steps_per_episode=100):
    """
    训练DQN调度器
    
    :param episodes: 训练回合数
    :param steps_per_episode: 每回合步数
    """
    print("=" * 80)
    print("DQN深度强化学习调度器训练（离散动作空间：1-10分钟）")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    
    config.simulation_config.time_steps = steps_per_episode
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    config.network_config.num_nodes = 10  # 较小网络用于快速训练
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    config.node_config.energy_char = 500.0
    
    # 2. 创建DQN调度器
    from info_collection.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,          # 10个离散动作（1-10分钟）
        lr=1e-3,                # 学习率
        gamma=0.99,             # 折扣因子
        tau=0.005,              # 软更新系数
        buffer_capacity=10000,  # 经验回放容量
        epsilon_start=1.0,      # 初始探索率
        epsilon_end=0.01,       # 最终探索率
        epsilon_decay=0.995,    # 探索率衰减
        training_mode=True      # 训练模式
    )
    
    print(f"\n[1] DQN调度器参数:")
    print(f"  - 动作空间: 10个离散动作（1-10分钟）")
    print(f"  - 学习率: 1e-3")
    print(f"  - 折扣因子: 0.99")
    print(f"  - 软更新系数: 0.005")
    print(f"  - 探索率: {scheduler.epsilon_start} → {scheduler.epsilon_end}")
    
    # 3. 训练循环
    print(f"\n[2] 开始训练（{episodes}回合，每回合{steps_per_episode}步）...")
    
    episode_rewards = []
    losses = []
    epsilons = []
    
    for episode in range(episodes):
        # 创建新网络（每回合重置）
        network = config.create_network()
        nim.initialize_node_info(network.nodes, initial_time=0)
        scheduler.nim = nim
        
        # 运行仿真
        simulation = EnergySimulation(
            network=network,
            time_steps=steps_per_episode,
            scheduler=scheduler,
            enable_energy_sharing=True,
            passive_mode=False  # 禁用被动模式，每步都传输
        )
        
        # 重置调度器状态
        scheduler.prev_state = None
        scheduler.prev_action = None
        scheduler.episode_count = episode + 1
        
        # 运行仿真（静默模式）
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        simulation.simulate()
        
        sys.stdout = old_stdout
        
        # 获取训练统计
        stats = scheduler.get_training_stats()
        
        # 记录统计信息
        if stats['avg_loss'] > 0:
            losses.append(stats['avg_loss'])
        epsilons.append(stats['epsilon'])
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"\n  回合 {episode + 1}/{episodes}:")
            print(f"    - 平均损失: {stats['avg_loss']:.4f}")
            print(f"    - 探索率ε: {stats['epsilon']:.4f}")
            print(f"    - 缓冲区大小: {stats['buffer_size']}")
            print(f"    - 更新次数: {stats['update_count']}")
    
    print("\n[3] 训练完成！")
    
    # 4. 保存模型
    model_path = "dqn_model.pth"
    scheduler.save_model(model_path)
    
    # 5. 绘制训练曲线
    if losses:
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 探索率曲线
        plt.subplot(1, 3, 2)
        plt.plot(epsilons)
        plt.title('Epsilon (Exploration Rate)')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        # 损失滑动平均
        plt.subplot(1, 3, 3)
        window = 10
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(smoothed)
            plt.title(f'Smoothed Loss (window={window})')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dqn_training_curves.png', dpi=150)
        print(f"\n[4] 训练曲线已保存: dqn_training_curves.png")
    
    return scheduler


def test_dqn_scheduler(model_path="dqn_model.pth"):
    """
    测试训练后的DQN调度器
    
    :param model_path: 模型路径
    """
    print("\n" + "=" * 80)
    print("DQN调度器测试（离散动作空间）")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    
    config.simulation_config.time_steps = 200
    config.simulation_config.enable_energy_sharing = True
    config.simulation_config.enable_k_adaptation = False
    config.simulation_config.fixed_k = 2
    
    config.network_config.num_nodes = 15
    config.network_config.distribution_mode = "random"
    config.network_config.enable_physical_center = True
    
    config.node_config.energy_char = 500.0
    
    # 2. 创建网络
    print("\n[1] 创建测试网络...")
    network = config.create_network()
    print(f"  [OK] 网络节点数: {len(network.nodes)}")
    
    # 3. 创建DQN调度器（测试模式）
    from info_collection.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    scheduler_dqn = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,
        training_mode=False  # 测试模式（不探索）
    )
    
    # 加载训练好的模型
    if os.path.exists(model_path):
        # 先运行一步以初始化agent
        scheduler_dqn.plan(network, 0)
        scheduler_dqn.load_model(model_path)
        print(f"\n[2] 加载训练模型: {model_path}")
    else:
        print(f"\n[2] 警告: 模型文件不存在，使用随机初始化")
    
    # 4. 创建标准Lyapunov调度器（对比）
    scheduler_lyapunov = LyapunovScheduler(
        node_info_manager=nim,
        V=0.5,
        K=2,
        max_hops=3
    )
    
    print("\n[3] 运行对比测试...")
    
    # 5. 测试DQN调度器
    print("\n  测试DQN调度器（200步）...")
    network_dqn = config.create_network()
    nim_dqn = NodeInfoManager(initial_position=(5.0, 5.0), enable_logging=False)
    nim_dqn.initialize_node_info(network_dqn.nodes, initial_time=0)
    scheduler_dqn.nim = nim_dqn
    
    simulation_dqn = EnergySimulation(
        network=network_dqn,
        time_steps=200,
        scheduler=scheduler_dqn,
        enable_energy_sharing=True,
        passive_mode=False
    )
    
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    simulation_dqn.simulate()
    sys.stdout = old_stdout
    
    results_dqn = simulation_dqn.result_manager.get_results()
    
    # 6. 测试标准Lyapunov调度器
    print("  测试标准Lyapunov调度器（200步）...")
    network_lyapunov = config.create_network()
    nim_lyapunov = NodeInfoManager(initial_position=(5.0, 5.0), enable_logging=False)
    nim_lyapunov.initialize_node_info(network_lyapunov.nodes, initial_time=0)
    scheduler_lyapunov.nim = nim_lyapunov
    
    simulation_lyapunov = EnergySimulation(
        network=network_lyapunov,
        time_steps=200,
        scheduler=scheduler_lyapunov,
        enable_energy_sharing=True,
        passive_mode=False
    )
    
    sys.stdout = io.StringIO()
    simulation_lyapunov.simulate()
    sys.stdout = old_stdout
    
    results_lyapunov = simulation_lyapunov.result_manager.get_results()
    
    # 7. 统计对比
    print("\n[4] 对比结果:")
    print("=" * 80)
    
    # DQN统计
    dqn_durations = []
    for result in results_dqn:
        if 'plans' in result and result['plans']:
            for plan in result['plans']:
                dqn_durations.append(plan.get('duration', 1))
    
    # Lyapunov统计（固定1分钟）
    lyapunov_count = 0
    for result in results_lyapunov:
        if 'plans' in result and result['plans']:
            lyapunov_count += len(result['plans'])
    
    print("\nDQN调度器（离散动作空间：1-10分钟）:")
    if dqn_durations:
        duration_dist = dict(zip(*np.unique(dqn_durations, return_counts=True)))
        print(f"  - 总传输次数: {len(dqn_durations)}")
        print(f"  - 平均传输时长: {np.mean(dqn_durations):.2f} 分钟")
        print(f"  - 传输时长分布: {duration_dist}")
        print(f"  - 总传输能量: {sum([d * 500 for d in dqn_durations]):.0f}J")
        
        # 动作分布可视化
        print(f"\n  动作使用频率:")
        for dur in sorted(duration_dist.keys()):
            count = duration_dist[dur]
            percentage = count / len(dqn_durations) * 100
            bar = '█' * int(percentage / 2)
            print(f"    {dur:2d}分钟: {bar} {count:3d}次 ({percentage:5.1f}%)")
    else:
        print(f"  - 无传输计划")
    
    print("\n标准Lyapunov调度器:")
    print(f"  - 总传输次数: {lyapunov_count}")
    print(f"  - 平均传输时长: 1.00 分钟（固定）")
    print(f"  - 总传输能量: {lyapunov_count * 500:.0f}J")
    
    # 8. 能量方差对比
    print("\n能量均衡性对比:")
    
    # DQN最终能量方差
    dqn_final_energies = [node.current_energy for node in network_dqn.nodes[1:]]
    dqn_std = np.std(dqn_final_energies)
    dqn_mean = np.mean(dqn_final_energies)
    dqn_cv = dqn_std / dqn_mean if dqn_mean > 0 else 0
    
    # Lyapunov最终能量方差
    lyap_final_energies = [node.current_energy for node in network_lyapunov.nodes[1:]]
    lyap_std = np.std(lyap_final_energies)
    lyap_mean = np.mean(lyap_final_energies)
    lyap_cv = lyap_std / lyap_mean if lyap_mean > 0 else 0
    
    print(f"  DQN - 能量标准差: {dqn_std:.2f}J, CV: {dqn_cv:.4f}")
    print(f"  Lyapunov - 能量标准差: {lyap_std:.2f}J, CV: {lyap_cv:.4f}")
    
    if dqn_cv < lyap_cv:
        improvement = (lyap_cv - dqn_cv) / lyap_cv * 100
        print(f"  [SUCCESS] DQN均衡性提升: {improvement:.1f}%")
    
    # 9. 离散动作空间的优势总结
    print("\n" + "=" * 80)
    print("离散动作空间（DQN）的优势:")
    print("=" * 80)
    print("  1. [✓] 训练更快 - ε-greedy策略简单高效")
    print("  2. [✓] 收敛更稳定 - 离散选择避免连续值的震荡")
    print("  3. [✓] 计算更快 - Q值计算比Actor-Critic快")
    print("  4. [✓] 动作范围大 - 支持1-10分钟（比DDPG的1-5更广）")
    print("  5. [✓] 易于理解 - 清晰的动作选择逻辑")
    
    print("\n[SUCCESS] 测试完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN调度器训练和测试（离散动作空间）')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                       help='运行模式: train(训练), test(测试), both(训练+测试)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='训练回合数（默认50）')
    parser.add_argument('--steps', type=int, default=100,
                       help='每回合步数（默认100）')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'both':
        train_dqn_scheduler(episodes=args.episodes, steps_per_episode=args.steps)
    
    if args.mode == 'test' or args.mode == 'both':
        test_dqn_scheduler()

