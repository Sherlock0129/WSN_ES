"""
重构后的主仿真程序
使用新的配置管理和接口系统
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import ConfigManager, load_config
from scheduling.schedulers import (
    LyapunovScheduler, ClusterScheduler, PredictionScheduler,
    PowerControlScheduler, BaselineHeuristic
)
from utils.error_handling import logger, error_handler, handle_exceptions
from viz.plotter import plot_node_distribution, plot_energy_over_time


def create_scheduler(config_manager: ConfigManager):
    """根据配置创建调度器"""
    scheduler_type = config_manager.scheduler_config.scheduler_type
    scheduler_params = config_manager.get_scheduler_params()
    
    logger.info(f"创建调度器: {scheduler_type}")
    
    if scheduler_type == "LyapunovScheduler":
        return LyapunovScheduler(**scheduler_params)
    elif scheduler_type == "ClusterScheduler":
        return ClusterScheduler(**scheduler_params)
    elif scheduler_type == "PredictionScheduler":
        return PredictionScheduler(**scheduler_params)
    elif scheduler_type == "PowerControlScheduler":
        return PowerControlScheduler(**scheduler_params)
    elif scheduler_type == "BaselineHeuristic":
        return BaselineHeuristic(**scheduler_params)
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")


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
    
    # 2. 创建网络
    logger.info("创建网络...")
    with handle_exceptions("网络创建", recoverable=False):
        network = config_manager.create_network()
        logger.info(f"网络创建完成: {network.num_nodes} 个节点")
    
    # 2.5. 创建ADCR链路层
    logger.info("创建ADCR链路层...")
    with handle_exceptions("ADCR链路层创建", recoverable=True):
        network.adcr_link = config_manager.create_adcr_link_layer(network)
        logger.info("ADCR链路层创建完成")
    
    # 3. 创建调度器
    logger.info("创建调度器...")
    with handle_exceptions("调度器创建", recoverable=False):
        scheduler = create_scheduler(config_manager)
        logger.info(f"调度器创建完成: {scheduler.get_name()}")
    
    # 4. 运行仿真
    logger.info("开始仿真...")
    with handle_exceptions("仿真运行", recoverable=False):
        simulation = config_manager.create_energy_simulation(network, scheduler)
        simulation.simulate()
        logger.info("仿真完成")
    
    # 5. 生成可视化
    logger.info("生成可视化图表...")
    with handle_exceptions("可视化生成", recoverable=True):
        # 绘制节点分布图
        plot_node_distribution(network.nodes)
        
        # 绘制能量随时间变化图
        plot_energy_over_time(network.nodes, simulation.results)
        
        # 绘制K值变化图
        simulation.plot_K_history()
        
        logger.info("可视化图表生成完成")
    
    # 6. 输出统计信息
    logger.info("输出统计信息...")
    stats = simulation.print_statistics()
    
    # 7. 保存结果
    logger.info("保存结果...")
    with handle_exceptions("结果保存", recoverable=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"data/simulation_results_{timestamp}.csv"
        simulation.save_results(results_file)
        logger.info(f"结果已保存到: {results_file}")
    
    # 8. 输出错误摘要
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        logger.warning(f"仿真过程中发生 {error_summary['total_errors']} 个错误")
        logger.info(f"错误类型统计: {error_summary['error_counts']}")
    else:
        logger.info("仿真过程中未发生错误")
    
    return simulation, stats


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
        elif sys.argv[1] == "config":
            # 使用指定配置文件
            config_file = sys.argv[2] if len(sys.argv) > 2 else "src/config/default_config.json"
            run_simulation(config_file)
        else:
            logger.error(f"未知参数: {sys.argv[1]}")
            logger.info("使用方法:")
            logger.info("  python main.py                    # 使用默认配置运行仿真")
            logger.info("  python main.py config <file>      # 使用指定配置文件")
            logger.info("  python main.py compare            # 比较调度器性能")
    else:
        # 使用默认配置运行仿真
        run_simulation()
    
    logger.info("程序执行完成")


if __name__ == "__main__":
    main()
