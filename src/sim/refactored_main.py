"""
重构后的主仿真程序
使用新的配置管理和接口系统
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import ConfigManager, load_config
from utils.logger import logger, get_detailed_plan_logger
from scheduling.schedulers import (
    LyapunovScheduler, ClusterScheduler, PredictionScheduler,
    PowerControlScheduler, BaselineHeuristic,
    AdaptiveDurationLyapunovScheduler
)
# 尝试导入深度学习调度器（可能需要PyTorch）
try:
    from scheduling.dqn_scheduler import DQNScheduler
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
    logger.warning("DQN调度器不可用（需要安装PyTorch）")

try:
    from scheduling.ddpg_scheduler import DDPGScheduler
    DDPG_AVAILABLE = True
except ImportError:
    DDPG_AVAILABLE = False
    logger.warning("DDPG调度器不可用（需要安装PyTorch）")
from scheduling.passive_transfer import compare_passive_modes as _compare_passive_modes
from utils.error_handling import error_handler, handle_exceptions
from viz.plotter import plot_node_distribution, plot_energy_over_time, plot_center_node_energy
from sim.parallel_executor import ParallelSimulationExecutor


def create_scheduler(config_manager: ConfigManager, network):
    """
    根据配置创建调度器
    
    :param config_manager: 配置管理器
    :param network: 网络对象
    :return: 调度器实例
    """
    scheduler_type = config_manager.scheduler_config.scheduler_type
    scheduler_params = config_manager.get_scheduler_params()
    
    # 获取节点信息管理器（从ADCR、PathCollector或PeriodicCollector）
    node_info_manager = None
    if hasattr(network, 'adcr_link') and network.adcr_link is not None:
        node_info_manager = network.adcr_link.vc
        logger.info("调度器使用ADCR的节点信息管理器")
    elif hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        node_info_manager = network.path_info_collector.vc
        logger.info("调度器使用PathCollector的节点信息管理器")
    elif hasattr(network, 'periodic_info_collector') and network.periodic_info_collector is not None:
        node_info_manager = network.periodic_info_collector.vc
        logger.info("调度器使用PeriodicCollector的节点信息管理器")
    else:
        # 如果没有信息管理器，创建一个独立的
        physical_center = network.get_physical_center() if hasattr(network, 'get_physical_center') else None
        if physical_center:
            initial_pos = tuple(physical_center.position)
        else:
            nodes = network.nodes
            initial_pos = (
                sum(n.position[0] for n in nodes) / len(nodes),
                sum(n.position[1] for n in nodes) / len(nodes)
            )
        node_info_manager = config_manager.create_virtual_center(initial_position=initial_pos, enable_logging=False)
        node_info_manager.initialize_node_info(network.nodes, initial_time=0)
        logger.info("调度器创建了独立的节点信息管理器")
    
    # 添加node_info_manager到参数中
    scheduler_params['node_info_manager'] = node_info_manager
    
    # 检查是否启用深度学习调度器
    sched_config = config_manager.scheduler_config
    
    # 优先检查DQN/DDPG开关（覆盖scheduler_type）
    if sched_config.enable_dqn:
        if not DQN_AVAILABLE:
            raise ImportError("DQN调度器已启用但PyTorch未安装。请安装: pip install torch")
        
        logger.info("=" * 60)
        logger.info("使用DQN深度强化学习调度器（离散动作空间：1-10分钟）")
        logger.info(f"  - 训练模式: {sched_config.dqn_training_mode}")
        logger.info(f"  - 模型路径: {sched_config.dqn_model_path}")
        logger.info(f"  - 动作空间: {sched_config.dqn_action_dim}个离散动作")
        logger.info("=" * 60)
        
        scheduler = DQNScheduler(
            node_info_manager=node_info_manager,
            K=scheduler_params.get('K', 2),
            max_hops=scheduler_params.get('max_hops', 5),
            action_dim=sched_config.dqn_action_dim,
            lr=sched_config.dqn_lr,
            gamma=sched_config.dqn_gamma,
            tau=sched_config.dqn_tau,
            buffer_capacity=sched_config.dqn_buffer_capacity,
            epsilon_start=sched_config.dqn_epsilon_start,
            epsilon_end=sched_config.dqn_epsilon_end,
            epsilon_decay=sched_config.dqn_epsilon_decay,
            training_mode=sched_config.dqn_training_mode
        )
        
        # 如果是测试模式，加载已训练模型
        if not sched_config.dqn_training_mode and os.path.exists(sched_config.dqn_model_path):
            # 先初始化agent
            dummy_network = network if network else None
            if dummy_network:
                scheduler.plan(dummy_network, 0)
            scheduler.load_model(sched_config.dqn_model_path)
            logger.info(f"DQN模型已加载: {sched_config.dqn_model_path}")
        elif not sched_config.dqn_training_mode:
            logger.warning(f"⚠ DQN模型文件不存在: {sched_config.dqn_model_path}")
            logger.warning("  将使用随机初始化的网络（性能可能较差）")
        
        return scheduler
    
    elif sched_config.enable_ddpg:
        if not DDPG_AVAILABLE:
            raise ImportError("DDPG调度器已启用但PyTorch未安装。请安装: pip install torch")
        
        logger.info("=" * 60)
        logger.info("使用DDPG深度强化学习调度器（连续动作空间：自主探索）")
        logger.info(f"  - 训练模式: {sched_config.ddpg_training_mode}")
        logger.info(f"  - 模型路径: {sched_config.ddpg_model_path}")
        logger.info(f"  - 动作范围: [{sched_config.ddpg_action_min:.1f}, {sched_config.ddpg_action_max:.1f}] 分钟")
        logger.info("=" * 60)
        
        scheduler = DDPGScheduler(
            node_info_manager=node_info_manager,
            K=scheduler_params.get('K', 2),
            max_hops=scheduler_params.get('max_hops', 5),
            action_dim=sched_config.ddpg_action_dim,
            actor_lr=sched_config.ddpg_actor_lr,
            critic_lr=sched_config.ddpg_critic_lr,
            gamma=sched_config.ddpg_gamma,
            tau=sched_config.ddpg_tau,
            buffer_capacity=sched_config.ddpg_buffer_capacity,
            training_mode=sched_config.ddpg_training_mode,
            action_min=sched_config.ddpg_action_min,
            action_max=sched_config.ddpg_action_max
        )
        
        # 如果是测试模式，加载已训练模型
        if not sched_config.ddpg_training_mode and os.path.exists(sched_config.ddpg_model_path):
            dummy_network = network if network else None
            if dummy_network:
                scheduler.plan(dummy_network, 0)
            scheduler.load_model(sched_config.ddpg_model_path)
            logger.info(f"DDPG模型已加载: {sched_config.ddpg_model_path}")
        elif not sched_config.ddpg_training_mode:
            logger.warning(f"⚠ DDPG模型文件不存在: {sched_config.ddpg_model_path}")
            logger.warning("  将使用随机初始化的网络（性能可能较差）")
        
        return scheduler
    
    # 使用传统调度器
    logger.info("=" * 60)
    logger.info(f"创建调度器: {scheduler_type}")
    logger.info("=" * 60)
    
    if scheduler_type == "ThresholdScheduler":
        from scheduling.schedulers import ThresholdScheduler
        scheduler = ThresholdScheduler(**scheduler_params)
        logger.info("使用极简阈值法调度器 (ThresholdScheduler)")
    elif scheduler_type == "LyapunovScheduler":

        scheduler = LyapunovScheduler(**scheduler_params)
        logger.info("使用标准 Lyapunov 调度器")
    elif scheduler_type == "AdaptiveLyapunovScheduler":
        from scheduling.schedulers import AdaptiveLyapunovScheduler
        scheduler = AdaptiveLyapunovScheduler(**scheduler_params)
        logger.info("使用自适应参数 Lyapunov 调度器 (AdaptiveLyapunovScheduler)")
        logger.info(f"  - 初始V: {scheduler_params.get('V', 0.5)}")
        logger.info(f"  - V范围: [{scheduler_params.get('V_min', 0.1)}, {scheduler_params.get('V_max', 2.0)}]")
        logger.info(f"  - 调整速率: {scheduler_params.get('adjust_rate', 0.1)*100:.0f}%")
        logger.info(f"  - 反馈窗口: {scheduler_params.get('window_size', 10)}")
        logger.info(f"  - 特性: V参数自动调整，基于4维反馈（均衡、效率、存活率、总能量）")
    elif scheduler_type == "AdaptiveDurationLyapunovScheduler":
        scheduler = AdaptiveDurationLyapunovScheduler(**scheduler_params)
        logger.info("使用自适应时长 Lyapunov 调度器")
    elif scheduler_type == "ClusterScheduler":
        scheduler = ClusterScheduler(**scheduler_params)
        logger.info("使用聚类调度器")
    elif scheduler_type == "PredictionScheduler":
        scheduler = PredictionScheduler(**scheduler_params)
        logger.info("使用预测调度器")
    elif scheduler_type == "PowerControlScheduler":
        scheduler = PowerControlScheduler(**scheduler_params)
        logger.info("使用功率控制调度器")
    elif scheduler_type == "BaselineHeuristic":
        scheduler = BaselineHeuristic(**scheduler_params)
        logger.info("使用基线启发式调度器")
    elif scheduler_type == "duration_aware" or scheduler_type == "DurationAwareLyapunovScheduler":
        from scheduling.schedulers import DurationAwareLyapunovScheduler
        scheduler = DurationAwareLyapunovScheduler(**scheduler_params)
        logger.info("使用传输时长感知 Lyapunov 调度器 (DurationAwareLyapunovScheduler)")
        logger.info(f"  - 时长范围: {scheduler_params.get('min_duration', 1)}-{scheduler_params.get('max_duration', 5)} 分钟")
        logger.info(f"  - AoI权重: {scheduler_params.get('w_aoi', 0.1)}")
        logger.info(f"  - 信息量权重: {scheduler_params.get('w_info', 0.05)}")
        logger.info(f"  - 节点锁定: 启用（当duration > 1时）")
    elif scheduler_type == "AdaptiveDurationAwareLyapunovScheduler":
        from scheduling.schedulers import AdaptiveDurationAwareLyapunovScheduler
        scheduler = AdaptiveDurationAwareLyapunovScheduler(**scheduler_params)
        logger.info("使用自适应传输时长感知 Lyapunov 调度器 (AdaptiveDurationAwareLyapunovScheduler)")
        logger.info(f"  - 初始V: {scheduler_params.get('V', 0.5)}")
        logger.info(f"  - V范围: [{scheduler_params.get('V_min', 0.1)}, {scheduler_params.get('V_max', 2.0)}]")
        logger.info(f"  - 调整速率: {scheduler_params.get('adjust_rate', 0.1)*100:.0f}%")
        logger.info(f"  - 反馈窗口: {scheduler_params.get('window_size', 10)}")
        logger.info(f"  - 时长范围: {scheduler_params.get('min_duration', 1)}-{scheduler_params.get('max_duration', 5)} 分钟")
        logger.info(f"  - AoI权重: {scheduler_params.get('w_aoi', 0.02)}")
        logger.info(f"  - 信息量权重: {scheduler_params.get('w_info', 0.1)}")
        logger.info(f"  - 特性: V参数自适应调整 + 传输时长优化 + 节点锁定机制")
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    logger.info("=" * 60)
    return scheduler


def run_simulation(config_file: str = None):
    """运行仿真"""
    
    # 1. 加载配置
    logger.info("开始加载配置...")
    if config_file and os.path.exists(config_file):
        config_manager = load_config(config_file)
        logger.info(f"从 {config_file} 加载配置")
    else:
        config_manager = ConfigManager()
        logger.info("使用默认配置")
    
    # 1.1 显示当前使用的调度器类型（用于调试）
    scheduler_type = config_manager.scheduler_config.scheduler_type
    logger.info(f"当前调度器类型: {scheduler_type}")
    logger.info("提示: 如果调度器类型不正确，请在代码中显式设置:")
    logger.info(f"  config_manager.scheduler_config.scheduler_type = \"你的调度器类型\"")
    
    # 1.1 检查是否是DQN/DDPG训练模式
    sched_config = config_manager.scheduler_config
    is_dqn_training = sched_config.enable_dqn and sched_config.dqn_training_mode
    is_ddpg_training = sched_config.enable_ddpg and sched_config.ddpg_training_mode
    
    if is_dqn_training or is_ddpg_training:
        # 使用训练循环
        return _run_training_loop(config_manager, config_file)
    
    # 否则使用标准单次运行
    return _run_single_simulation(config_manager, config_file)


def _run_training_loop(config_manager: ConfigManager, config_file: str = None):
    """DQN/DDPG训练循环"""
    sched_config = config_manager.scheduler_config
    
    if sched_config.enable_dqn:
        mode_name = "DQN"
        training_episodes = sched_config.dqn_training_episodes
        save_interval = sched_config.dqn_save_interval
        model_path = sched_config.dqn_model_path
        action_info = f"离散动作空间：{sched_config.dqn_action_dim}个选项"
    else:
        mode_name = "DDPG"
        training_episodes = sched_config.ddpg_training_episodes
        save_interval = sched_config.ddpg_save_interval
        model_path = sched_config.ddpg_model_path
        action_info = f"连续动作空间：[{sched_config.ddpg_action_min:.1f}, {sched_config.ddpg_action_max:.1f}]分钟"
    
    logger.info("=" * 80)
    logger.info(f"{mode_name} 训练模式 - 自主探索最优传输时长")
    logger.info("=" * 80)
    logger.info(f"训练回合数: {training_episodes}")
    logger.info(f"每回合步数: {config_manager.simulation_config.time_steps}")
    logger.info(f"模型保存间隔: 每{save_interval}回合")
    logger.info(f"模型保存路径: {model_path}")
    logger.info(f"动作空间: {action_info}")
    logger.info("=" * 80)
    
    # 训练循环

    scheduler = None
    for episode in range(training_episodes):
        logger.info(f"\n{'='*70}")
        logger.info(f"训练回合 {episode + 1}/{training_episodes}")
        logger.info(f"{'='*70}")
        
        # 创建新网络
        network = config_manager.create_network()
        
        # 创建或复用调度器
        if episode == 0:
            scheduler = create_scheduler(config_manager, network)
            if hasattr(scheduler, 'episode_count'):
                scheduler.episode_count = 0
        else:
            # 复用调度器，重置状态
            if hasattr(scheduler, 'nim'):
                scheduler.nim.initialize_node_info(network.nodes, initial_time=0)
            if hasattr(scheduler, 'prev_state'):
                scheduler.prev_state = None
                scheduler.prev_action = None
        
        if hasattr(scheduler, 'episode_count'):
            scheduler.episode_count = episode + 1
        
        # 运行仿真（使用config_manager创建，确保所有参数正确传递）
        simulation = config_manager.create_energy_simulation(network, scheduler)
        simulation.training_mode = True  # 标记为训练模式
        
        # 静默运行
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        simulation.simulate()
        sys.stdout = old_stdout
        
        # 显示训练统计
        if hasattr(scheduler, 'get_training_stats'):
            stats = scheduler.get_training_stats()
            logger.info(f"训练统计:")
            
            # DQN统计
            if 'avg_loss' in stats:
                logger.info(f"  - 平均损失: {stats['avg_loss']:.4f}")
                logger.info(f"  - 探索率ε: {stats.get('epsilon', 0):.4f}")
                logger.info(f"  - 缓冲区: {stats.get('buffer_size', 0)}")
                logger.info(f"  - 更新次数: {stats.get('update_count', 0)}")
            
            # DDPG统计
            if 'avg_actor_loss' in stats:
                logger.info(f"  - Actor损失: {stats['avg_actor_loss']:.4f}")
                logger.info(f"  - Critic损失: {stats['avg_critic_loss']:.4f}")
                logger.info(f"  - 缓冲区: {stats.get('buffer_size', 0)}")
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            if hasattr(scheduler, 'save_model'):
                scheduler.save_model(model_path)
                logger.info(f"模型已保存（回合{episode + 1}）")
    
    # 训练完成，保存最终模型
    if scheduler and hasattr(scheduler, 'save_model'):
        scheduler.save_model(model_path)
        logger.info("\n" + "="*80)
        logger.info(f"{mode_name} 训练完成！")
        logger.info(f"最终模型已保存到: {model_path}")
        logger.info("="*80)
        logger.info(f"\n使用训练好的模型:")
        logger.info(f"1. 在配置中设置 dqn_training_mode = False")
        logger.info(f"2. 重新运行: python src/sim/refactored_main.py")


def _run_single_simulation(config_manager: ConfigManager, config_file: str = None):
    """运行单次仿真（标准模式）"""
    
    sim_config = config_manager.simulation_config
    if sim_config.passive_mode:
        logger.info("=" * 50)
        logger.info("智能被动传能模式已启用")
        logger.info(f"  - 检查间隔: {sim_config.check_interval} 分钟")
        logger.info(f"  - 临界比例: {sim_config.critical_ratio * 100:.1f}%")
        logger.info(f"  - 能量方差阈值: {sim_config.energy_variance_threshold:.2f}")
        logger.info(f"  - 冷却期: {sim_config.cooldown_period} 分钟")
        logger.info(f"  - 预测窗口: {sim_config.predictive_window} 分钟")
        logger.info("=" * 50)
    else:
        logger.info("=" * 50)
        logger.info("传统主动传能模式 (每60分钟定时触发)")
        logger.info("=" * 50)
    
    # 2. 创建网络
    logger.info("创建网络...")
    with handle_exceptions("网络创建", recoverable=False):
        network = config_manager.create_network()
        logger.info(f"网络创建完成: {network.num_nodes} 个节点")
    
    # 2.5. 创建ADCR链路层（可选）
    if config_manager.simulation_config.enable_adcr_link_layer:
        logger.info("创建ADCR链路层...")
        with handle_exceptions("ADCR链路层创建", recoverable=True):
            network.adcr_link = config_manager.create_adcr_link_layer(network)
            logger.info("ADCR链路层创建完成")
    else:
        logger.info("ADCR链路层已禁用")
        network.adcr_link = None
    
    # 2.6. 创建信息收集器（路径 / 定期）
    vc = None

    def ensure_virtual_center():
        nonlocal vc
        if network.adcr_link is not None:
            vc = network.adcr_link.vc
            return vc
        if vc is None:
            physical_center = network.get_physical_center()
            if physical_center:
                initial_pos = tuple(physical_center.position)
            else:
                nodes = network.get_regular_nodes() if hasattr(network, 'get_regular_nodes') else network.nodes
                initial_pos = (
                    sum(n.position[0] for n in nodes) / len(nodes),
                    sum(n.position[1] for n in nodes) / len(nodes)
                )
            vc = config_manager.create_virtual_center(initial_position=initial_pos, enable_logging=True)
            vc.initialize_node_info(network.nodes, initial_time=0)
            logger.info(f"创建独立虚拟中心，位置: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
        return vc

    if config_manager.path_collector_config.enable_path_collector:
        logger.info("创建路径信息收集器...")
        with handle_exceptions("路径信息收集器创建", recoverable=True):
            physical_center = network.get_physical_center()
            network.path_info_collector = config_manager.create_path_collector(ensure_virtual_center(), physical_center)
            logger.info(f"路径信息收集器创建完成 (物理中心: {'ID=' + str(physical_center.node_id) if physical_center else '未启用'})")
    else:
        logger.info("路径信息收集器已禁用")
        network.path_info_collector = None

    if config_manager.periodic_collector_config.enable_periodic_collector:
        logger.info("创建定期上报信息收集器...")
        with handle_exceptions("定期信息收集器创建", recoverable=True):
            physical_center = network.get_physical_center()
            network.periodic_info_collector = config_manager.create_periodic_collector(ensure_virtual_center(), physical_center)
            logger.info(f"定期信息收集器创建完成 (物理中心: {'ID=' + str(physical_center.node_id) if physical_center else '未启用'})")
    else:
        network.periodic_info_collector = None
    
    # 3. 创建调度器
    logger.info("创建调度器...")
    with handle_exceptions("调度器创建", recoverable=False):
        scheduler = create_scheduler(config_manager, network)
        logger.info(f"调度器创建完成: {scheduler.get_name()}")
    
    # 3.5 设置EETOR配置（确保路由算法使用正确的配置）
    from routing.energy_transfer_routing import set_eetor_config
    set_eetor_config(config_manager.eetor_config)
    if config_manager.eetor_config.enable_info_aware_routing:
        logger.info("信息感知路由已启用")
        logger.info(f"  - info_reward_factor: {config_manager.eetor_config.info_reward_factor}")
    
    # 4. 运行仿真
    logger.info("开始仿真...")
    with handle_exceptions("仿真运行", recoverable=False):
        simulation = config_manager.create_energy_simulation(network, scheduler)
        
        # 设置虚拟中心归档路径
        if hasattr(network, 'adcr_link') and network.adcr_link is not None:
            # ADCR的虚拟中心
            network.adcr_link.set_archive_path(simulation.session_dir)
            logger.info("虚拟中心归档路径已设置（ADCR）")
        else:
            if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
                import os
                archive_path = os.path.join(simulation.session_dir, "virtual_center_node_info.csv")
                network.path_info_collector.vc.archive_path = archive_path
                logger.info(f"虚拟中心归档路径已设置（PathCollector）: {archive_path}")
            if hasattr(network, 'periodic_info_collector') and network.periodic_info_collector is not None:
                import os
                archive_path = os.path.join(simulation.session_dir, "virtual_center_node_info.csv")
                network.periodic_info_collector.vc.archive_path = archive_path
                logger.info(f"虚拟中心归档路径已设置（PeriodicCollector）: {archive_path}")
        
        simulation.simulate()
        logger.info("仿真完成")
        
        # 如果使用自适应调度器，打印自适应摘要
        from scheduling.schedulers import AdaptiveLyapunovScheduler
        if isinstance(scheduler, AdaptiveLyapunovScheduler):
            logger.info("\n" + "=" * 80)
            scheduler.print_adaptation_summary()
            logger.info("=" * 80)
        
        # 强制刷新虚拟中心归档
        if hasattr(network, 'adcr_link') and network.adcr_link is not None:
            network.adcr_link.vc.force_flush_archive()
            logger.info("虚拟中心归档已保存（ADCR）")
        else:
            if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
                network.path_info_collector.vc.force_flush_archive()
                logger.info("虚拟中心归档已保存（PathCollector）")
            if hasattr(network, 'periodic_info_collector') and network.periodic_info_collector is not None:
                network.periodic_info_collector.vc.force_flush_archive()
                logger.info("虚拟中心归档已保存（PeriodicCollector）")
    
    # 5. 生成可视化
    if config_manager.simulation_config.enable_visualization:
        logger.info("生成可视化图表...")
        with handle_exceptions("可视化生成", recoverable=True):
            # 绘制节点分布图
            plot_node_distribution(network.nodes, session_dir=simulation.session_dir)
            
            # 绘制能量随时间变化图
            plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=simulation.session_dir)
            
            # 绘制物理中心节点能量变化图
            plot_center_node_energy(network.nodes, simulation.result_manager.get_results(), session_dir=simulation.session_dir)
            
            # 绘制K值变化图
            simulation.plot_K_history()
            
            # 绘制ADCR聚类和路径图
            if hasattr(network, 'adcr_link') and network.adcr_link is not None:
                from viz.plotter import plot_adcr_clusters_and_paths
                plot_adcr_clusters_and_paths(network.adcr_link, session_dir=simulation.session_dir)
            
            # 如果使用DurationAwareLyapunovScheduler，生成专门的可视化
            from scheduling.schedulers import DurationAwareLyapunovScheduler
            if isinstance(scheduler, DurationAwareLyapunovScheduler):
                logger.info("生成传输时长感知的专门可视化...")
                try:
                    from viz.duration_aware_plotter import (
                        plot_duration_statistics,
                        plot_energy_transfer_timeline,
                        plot_energy_transfer_with_duration,
                        create_energy_flow_animation
                    )
                    
                    # 统计图
                    plot_duration_statistics(simulation, simulation.session_dir)
                    logger.info("传输时长统计图生成完成")
                    
                    # 为关键时间点生成时间线图和路径图
                    if hasattr(simulation, 'plans_by_time'):
                        # 选择有传输的时间点
                        transfer_times = [t for t, data in simulation.plans_by_time.items() 
                                        if data.get('plans')]
                        
                        # 选择前3个和最后1个时间点
                        sample_times = transfer_times[:3] + transfer_times[-1:] if transfer_times else []
                        
                        for t in sample_times:
                            plans = simulation.plans_by_time[t].get('plans', [])
                            if plans:
                                plot_energy_transfer_timeline(plans, network, t, simulation.session_dir)
                                plot_energy_transfer_with_duration(plans, network, t, simulation.session_dir)
                        
                        logger.info(f"为{len(sample_times)}个时间点生成了时间线和路径图")
                    
                    # 创建动画（可选，比较耗时）
                    # create_energy_flow_animation(simulation)
                    # logger.info("能量传输动画生成完成")
                    
                except Exception as e:
                    logger.warning(f"生成传输时长可视化时出错: {e}")
            
            logger.info("可视化图表生成完成")
    
    # 6. 输出统计信息
    logger.info("输出统计信息...")
    stats = simulation.print_statistics()
    
    # 7. 导出配置参数
    logger.info("导出配置参数...")
    with handle_exceptions("配置参数导出", recoverable=True):
        config_file = config_manager.export_config_to_session(simulation.session_dir)
        if config_file:
            logger.info(f"配置参数已导出到: {config_file}")
        else:
            logger.warning("配置参数导出失败")
    
    # 8. 保存详细计划日志
    logger.info("保存详细计划日志...")
    with handle_exceptions("详细计划日志保存", recoverable=True):
        # 创建详细计划日志记录器，使用仿真的会话目录
        plan_logger = get_detailed_plan_logger(simulation.session_dir)
        plan_logger.save_simulation_plans(simulation)
        logger.info("详细计划日志保存完成")
    
    # 9. 输出路径信息收集器统计（如果启用）
    if network.path_info_collector is not None:
        logger.info("=" * 60)
        logger.info("路径信息收集器统计:")
        logger.info("=" * 60)
        network.path_info_collector.print_statistics()
    
    # 10. 保存结果
    logger.info("保存结果...")
    with handle_exceptions("结果保存", recoverable=True):
        # 使用EnergySimulation的默认路径（按日期组织）
        simulation.save_results()
        logger.info(f"结果已保存到: {simulation.session_dir}")
    
    # 11. 输出错误摘要
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        logger.warning(f"仿真过程中发生 {error_summary['total_errors']} 个错误")
        logger.info(f"错误类型统计: {error_summary['error_counts']}")
    else:
        logger.info("仿真过程中未发生错误")
    
    return simulation, stats


def compare_passive_modes():
    """比较智能被动传能 vs 传统主动传能的性能（调用scheduling模块）"""
    return _compare_passive_modes(ConfigManager, create_scheduler, logger)


def compare_schedulers():
    """比较不同调度器的性能"""
    logger.info("开始调度器性能比较...")
    
    schedulers_to_test = [
        "LyapunovScheduler",
        "ClusterScheduler", 
        "PredictionScheduler",
        "BaselineHeuristic"
    ]
    
    results = {}
    
    for scheduler_type in schedulers_to_test:
        logger.info(f"测试调度器: {scheduler_type}")
        
        # 创建配置
        config_manager = ConfigManager()
        config_manager.scheduler_config.scheduler_type = scheduler_type
        
        try:
            # 运行仿真
            simulation, stats = run_simulation()
            results[scheduler_type] = {
                'avg_variance': stats['avg_variance'],
                'total_loss_energy': stats['total_loss_energy'],
                'energy_efficiency': stats['total_received_energy'] / stats['total_sent_energy'] if stats['total_sent_energy'] > 0 else 0
            }
            logger.info(f"{scheduler_type} 测试完成")
            
        except Exception as e:
            logger.error(f"{scheduler_type} 测试失败: {str(e)}")
            results[scheduler_type] = None
    
    # 输出比较结果
    logger.info("调度器性能比较结果:")
    for scheduler_type, result in results.items():
        if result:
            logger.info(f"{scheduler_type}:")
            logger.info(f"  平均方差: {result['avg_variance']:.4f}")
            logger.info(f"  总能量损失: {result['total_loss_energy']:.4f} J")
            logger.info(f"  能量效率: {result['energy_efficiency']:.2%}")
        else:
            logger.info(f"{scheduler_type}: 测试失败")
    
    return results


def run_parallel_simulation(config_file: str = None):
    """运行并行仿真"""
    logger.info("=" * 50)
    logger.info("无线传感器网络能量传输仿真系统 - 并行模式")
    logger.info("=" * 50)
    
    # 1. 加载配置
    if config_file and os.path.exists(config_file):
        config_manager = ConfigManager(config_file)
        logger.info(f"从 {config_file} 加载配置")
    else:
        config_manager = ConfigManager()
        logger.info("使用默认配置")
    
    # 检查并行配置
    if not config_manager.parallel_config.enabled:
        logger.error("并行模式未在配置中启用")
        logger.info("请在配置文件中设置 parallel.enabled = true")
        return
    
    # 2. 运行并行仿真
    logger.info("开始并行仿真...")
    with handle_exceptions("并行仿真运行", recoverable=False):
        executor = ParallelSimulationExecutor(config_manager)
        results = executor.run_parallel_simulations()
        
        successful_runs = [r for r in results if r["status"] == "success"]
        failed_runs = [r for r in results if r["status"] == "failed"]
        
        logger.info(f"并行仿真完成: 成功 {len(successful_runs)}/{len(results)} 次运行")
        if failed_runs:
            logger.warning(f"失败 {len(failed_runs)} 次运行")
    
    logger.info("并行仿真程序执行完成")


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("无线传感器网络能量传输仿真系统")
    logger.info("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            # 比较调度器性能
            compare_schedulers()
        elif sys.argv[1] == "compare-passive":
            # 比较智能被动传能 vs 传统主动传能
            compare_passive_modes()
        elif sys.argv[1] == "config":
            # 使用指定配置文件
            config_file = sys.argv[2] if len(sys.argv) > 2 else "src/config/default_config.json"
            run_simulation(config_file)
        elif sys.argv[1] == "parallel":
            # 并行仿真模式
            config_file = sys.argv[2] if len(sys.argv) > 2 else "src/config/default_config.json"
            run_parallel_simulation(config_file)
        elif sys.argv[1] == "passive":
            # 智能被动传能配置示例
            config_file = "src/config/智能被动传能示例.json"
            if os.path.exists(config_file):
                run_simulation(config_file)
            else:
                logger.warning(f"配置文件不存在: {config_file}")
                logger.info("将使用默认智能被动传能配置运行...")
                run_simulation()
        elif sys.argv[1] == "help" or sys.argv[1] == "-h" or sys.argv[1] == "--help":
            print_help()
        else:
            logger.error(f"未知参数: {sys.argv[1]}")
            print_help()
    else:
        # 使用默认配置运行仿真
        run_simulation()
    
    logger.info("程序执行完成")


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 70)
    print("无线传感器网络能量传输仿真系统 - 使用说明")
    print("=" * 70)
    print("\n基本用法:")
    print("  python src/sim/refactored_main.py [选项] [参数]")
    print("\n可用选项:")
    print("  (无参数)           - 使用默认配置运行仿真（启用智能被动传能）")
    print("  config <file>      - 使用指定的JSON配置文件运行仿真")
    print("  passive            - 使用智能被动传能示例配置运行")
    print("  compare            - 比较不同调度器的性能")
    print("  compare-passive    - 比较智能被动传能 vs 传统主动传能")
    print("  parallel <file>    - 运行并行仿真")
    print("  help / -h / --help - 显示此帮助信息")
    print("\n示例:")
    print("  # 默认运行（智能被动传能）")
    print("  python src/sim/refactored_main.py")
    print()
    print("  # 使用智能被动传能示例配置")
    print("  python src/sim/refactored_main.py passive")
    print()
    print("  # 使用自定义配置文件")
    print("  python src/sim/refactored_main.py config my_config.json")
    print()
    print("  # 比较传能模式性能")
    print("  python src/sim/refactored_main.py compare-passive")
    print()
    print("  # 比较调度器性能")
    print("  python src/sim/refactored_main.py compare")
    print("\n智能被动传能说明:")
    print("  系统默认启用智能被动传能模式，具有以下特点：")
    print("  - 基于多维度综合决策触发能量传输")
    print("  - 支持低能量节点比例、能量方差、预测性触发")
    print("  - 冷却期机制防止频繁触发")
    print("  - 相比定时触发可节省30-50%的传能次数")
    print("\n配置文件:")
    print("  默认配置: 自动启用智能被动传能")
    print("  示例配置: src/config/智能被动传能示例.json")
    print("  详细文档: docs/智能被动传能系统说明.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
