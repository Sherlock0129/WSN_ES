"""
支持DQN训练的主仿真程序

使用方法：
1. 在 simulation_config.py 中设置：
   enable_dqn = True
   dqn_training_mode = True
   dqn_training_episodes = 50  # 训练回合数

2. 运行：python src/sim/refactored_main_with_training.py
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import ConfigManager, load_config
from sim.refactored_main import create_scheduler, run_simulation as _original_run
from utils.logger import logger


def run_simulation_with_training(config_file: str = None):
    """运行仿真（支持DQN训练循环）"""
    
    # 加载配置
    if config_file and os.path.exists(config_file):
        config_manager = load_config(config_file)
        logger.info(f"从 {config_file} 加载配置")
    else:
        config_manager = ConfigManager()
        logger.info("使用默认配置")
    
    sched_config = config_manager.scheduler_config
    
    # 检查是否是DQN训练模式
    if sched_config.enable_dqn and sched_config.dqn_training_mode:
        logger.info("=" * 80)
        logger.info("DQN训练模式")
        logger.info("=" * 80)
        
        # 获取训练参数
        training_episodes = getattr(sched_config, 'dqn_training_episodes', 50)
        save_interval = getattr(sched_config, 'dqn_save_interval', 10)
        
        logger.info(f"训练回合数: {training_episodes}")
        logger.info(f"模型保存间隔: 每{save_interval}回合")
        logger.info(f"模型保存路径: {sched_config.dqn_model_path}")
        
        # 训练循环
        from core.energy_simulation import EnergySimulation
        
        for episode in range(training_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"训练回合 {episode + 1}/{training_episodes}")
            logger.info(f"{'='*60}")
            
            # 创建新网络
            network = config_manager.create_network()
            
            # 创建或复用调度器
            if episode == 0:
                scheduler = create_scheduler(config_manager, network)
                scheduler.episode_count = 0
            else:
                # 复用调度器，重置状态
                scheduler.nim.initialize_node_info(network.nodes, initial_time=0)
                scheduler.prev_state = None
                scheduler.prev_action = None
            
            scheduler.episode_count = episode + 1
            
            # 运行仿真
            simulation = EnergySimulation(
                network=network,
                time_steps=config_manager.simulation_config.time_steps,
                scheduler=scheduler,
                enable_energy_sharing=config_manager.simulation_config.enable_energy_sharing,
                passive_mode=config_manager.simulation_config.passive_mode
            )
            
            # 静默运行
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            simulation.simulate()
            sys.stdout = old_stdout
            
            # 显示训练统计
            stats = scheduler.get_training_stats()
            logger.info(f"训练统计:")
            logger.info(f"  - 平均损失: {stats['avg_loss']:.4f}")
            logger.info(f"  - 探索率ε: {stats['epsilon']:.4f}")
            logger.info(f"  - 缓冲区: {stats['buffer_size']}")
            
            # 定期保存模型
            if (episode + 1) % save_interval == 0:
                scheduler.save_model(sched_config.dqn_model_path)
                logger.info(f"✓ 模型已保存（回合{episode + 1}）")
        
        # 训练完成，保存最终模型
        scheduler.save_model(sched_config.dqn_model_path)
        logger.info("\n" + "="*80)
        logger.info("DQN训练完成！")
        logger.info(f"最终模型已保存到: {sched_config.dqn_model_path}")
        logger.info("="*80)
        
    else:
        # 非DQN训练模式，使用原始运行逻辑
        _original_run(config_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行WSN仿真（支持DQN训练）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（YAML格式）')
    
    args = parser.parse_args()
    
    run_simulation_with_training(config_file=args.config)

