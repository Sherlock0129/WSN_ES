"""
测试DDPG深度强化学习调度器

功能：
1. 训练DDPG智能体
2. 测试训练后的策略
3. 与标准Lyapunov调度器对比
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from config.simulation_config import ConfigManager
from core.network import Network
from core.energy_simulation import EnergySimulation
from scheduling.ddpg_scheduler import DDPGScheduler
from scheduling.schedulers import LyapunovScheduler


def train_ddpg_scheduler(episodes=50, steps_per_episode=100):
    """
    训练DDPG调度器
    
    :param episodes: 训练回合数
    :param steps_per_episode: 每回合步数
    """
    print("=" * 80)
    print("DDPG深度强化学习调度器训练")
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
    
    # 2. 创建DDPG调度器
    from acdr.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    
    scheduler = DDPGScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        buffer_capacity=10000,
        training_mode=True  # 训练模式
    )
    
    print(f"\n[1] DDPG调度器参数:")
    print(f"  - Actor学习率: 1e-4")
    print(f"  - Critic学习率: 1e-3")
    print(f"  - 折扣因子: 0.99")
    print(f"  - 软更新系数: 0.001")
    print(f"  - 经验回放容量: 10000")
    
    # 3. 训练循环
    print(f"\n[2] 开始训练（{episodes}回合，每回合{steps_per_episode}步）...")
    
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(episodes):
        # 创建新网络（每回合重置）
        network = config.create_network()
        nim.initialize_node_info(network.nodes, initial_time=0)
        scheduler.nim = nim
        
        # 重置DDPG噪声
        if scheduler.agent:
            scheduler.agent.noise.reset()
        
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
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        simulation.simulate()
        
        sys.stdout = old_stdout
        
        # 获取训练统计
        stats = scheduler.get_training_stats()
        
        # 记录损失
        if stats['avg_actor_loss'] > 0:
            actor_losses.append(stats['avg_actor_loss'])
        if stats['avg_critic_loss'] > 0:
            critic_losses.append(stats['avg_critic_loss'])
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"\n  回合 {episode + 1}/{episodes}:")
            print(f"    - 平均Actor损失: {stats['avg_actor_loss']:.4f}")
            print(f"    - 平均Critic损失: {stats['avg_critic_loss']:.4f}")
            print(f"    - 缓冲区大小: {stats['buffer_size']}")
    
    print("\n[3] 训练完成！")
    
    # 4. 保存模型
    model_path = "ddpg_model.pth"
    scheduler.save_model(model_path)
    
    # 5. 绘制训练曲线
    if actor_losses:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ddpg_training_curves.png')
        print(f"\n[4] 训练曲线已保存: ddpg_training_curves.png")
    
    return scheduler


def test_ddpg_scheduler(model_path="ddpg_model.pth"):
    """
    测试训练后的DDPG调度器
    
    :param model_path: 模型路径
    """
    print("\n" + "=" * 80)
    print("DDPG调度器测试")
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
    
    # 3. 创建DDPG调度器（测试模式）
    from acdr.physical_center import NodeInfoManager
    
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    scheduler_ddpg = DDPGScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        training_mode=False  # 测试模式（不添加噪声）
    )
    
    # 加载训练好的模型
    if os.path.exists(model_path):
        # 先运行一步以初始化agent
        scheduler_ddpg.plan(network, 0)
        scheduler_ddpg.load_model(model_path)
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
    
    # 5. 测试DDPG调度器
    print("\n  测试DDPG调度器（200步）...")
    network_ddpg = config.create_network()
    nim_ddpg = NodeInfoManager(initial_position=(5.0, 5.0), enable_logging=False)
    nim_ddpg.initialize_node_info(network_ddpg.nodes, initial_time=0)
    scheduler_ddpg.nim = nim_ddpg
    
    simulation_ddpg = EnergySimulation(
        network=network_ddpg,
        time_steps=200,
        scheduler=scheduler_ddpg,
        enable_energy_sharing=True,
        passive_mode=False
    )
    
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    simulation_ddpg.simulate()
    sys.stdout = old_stdout
    
    results_ddpg = simulation_ddpg.result_manager.get_results()
    
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
    
    # DDPG统计
    ddpg_durations = []
    for result in results_ddpg:
        if 'plans' in result and result['plans']:
            for plan in result['plans']:
                ddpg_durations.append(plan.get('duration', 1))
    
    # Lyapunov统计（固定1分钟）
    lyapunov_count = 0
    for result in results_lyapunov:
        if 'plans' in result and result['plans']:
            lyapunov_count += len(result['plans'])
    
    print("\nDDPG调度器:")
    if ddpg_durations:
        print(f"  - 总传输次数: {len(ddpg_durations)}")
        print(f"  - 平均传输时长: {np.mean(ddpg_durations):.2f} 分钟")
        print(f"  - 传输时长分布: {dict(zip(*np.unique(ddpg_durations, return_counts=True)))}")
        print(f"  - 总传输能量: {sum([d * 500 for d in ddpg_durations]):.0f}J")
    else:
        print(f"  - 无传输计划")
    
    print("\n标准Lyapunov调度器:")
    print(f"  - 总传输次数: {lyapunov_count}")
    print(f"  - 平均传输时长: 1.00 分钟（固定）")
    print(f"  - 总传输能量: {lyapunov_count * 500:.0f}J")
    
    # 8. 能量方差对比
    print("\n能量均衡性对比:")
    
    # DDPG最终能量方差
    ddpg_final_energies = [node.current_energy for node in network_ddpg.nodes[1:]]
    ddpg_std = np.std(ddpg_final_energies)
    ddpg_mean = np.mean(ddpg_final_energies)
    ddpg_cv = ddpg_std / ddpg_mean if ddpg_mean > 0 else 0
    
    # Lyapunov最终能量方差
    lyap_final_energies = [node.current_energy for node in network_lyapunov.nodes[1:]]
    lyap_std = np.std(lyap_final_energies)
    lyap_mean = np.mean(lyap_final_energies)
    lyap_cv = lyap_std / lyap_mean if lyap_mean > 0 else 0
    
    print(f"  DDPG - 能量标准差: {ddpg_std:.2f}J, CV: {ddpg_cv:.4f}")
    print(f"  Lyapunov - 能量标准差: {lyap_std:.2f}J, CV: {lyap_cv:.4f}")
    
    if ddpg_cv < lyap_cv:
        improvement = (lyap_cv - ddpg_cv) / lyap_cv * 100
        print(f"  [SUCCESS] DDPG均衡性提升: {improvement:.1f}%")
    
    print("\n[SUCCESS] 测试完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DDPG调度器训练和测试')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                       help='运行模式: train(训练), test(测试), both(训练+测试)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='训练回合数（默认50）')
    parser.add_argument('--steps', type=int, default=100,
                       help='每回合步数（默认100）')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'both':
        train_ddpg_scheduler(episodes=args.episodes, steps_per_episode=args.steps)
    
    if args.mode == 'test' or args.mode == 'both':
        test_ddpg_scheduler()

