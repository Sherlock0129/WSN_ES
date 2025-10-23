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
from utils.logger import logger, get_detailed_plan_logger
from utils.error_handling import error_handler, handle_exceptions
from viz.plotter import plot_node_distribution, plot_energy_over_time
from sim.parallel_executor import ParallelSimulationExecutor


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
    
    # 1.5 显示智能被动传能配置信息
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
        plot_node_distribution(network.nodes, session_dir=simulation.session_dir)
        
        # 绘制能量随时间变化图
        plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=simulation.session_dir)
        
        # 绘制K值变化图
        simulation.plot_K_history()
        
        # 绘制ADCR聚类和路径图
        if hasattr(network, 'adcr_link') and network.adcr_link is not None:
            from viz.plotter import plot_adcr_clusters_and_paths
            plot_adcr_clusters_and_paths(network.adcr_link, session_dir=simulation.session_dir)
        
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
    
    # 9. 保存结果
    logger.info("保存结果...")
    with handle_exceptions("结果保存", recoverable=True):
        # 使用EnergySimulation的默认路径（按日期组织）
        simulation.save_results()
        logger.info(f"结果已保存到: {simulation.session_dir}")
    
    # 10. 输出错误摘要
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        logger.warning(f"仿真过程中发生 {error_summary['total_errors']} 个错误")
        logger.info(f"错误类型统计: {error_summary['error_counts']}")
    else:
        logger.info("仿真过程中未发生错误")
    
    return simulation, stats


def compare_passive_modes():
    """比较智能被动传能 vs 传统主动传能的性能"""
    logger.info("=" * 60)
    logger.info("开始比较智能被动传能 vs 传统主动传能")
    logger.info("=" * 60)
    
    modes_to_test = [
        ("智能被动传能(默认)", {"passive_mode": True, "check_interval": 10, "critical_ratio": 0.2}),
        ("智能被动传能(快速)", {"passive_mode": True, "check_interval": 5, "critical_ratio": 0.15}),
        ("智能被动传能(节能)", {"passive_mode": True, "check_interval": 20, "critical_ratio": 0.3}),
        ("传统主动传能(60分钟)", {"passive_mode": False}),
    ]
    
    results = {}
    
    for mode_name, mode_config in modes_to_test:
        logger.info(f"\n测试模式: {mode_name}")
        logger.info("-" * 60)
        
        # 创建配置
        config_manager = ConfigManager()
        config_manager.simulation_config.enable_energy_sharing = True
        config_manager.simulation_config.time_steps = 10080  # 测试1天
        
        # 应用模式配置
        for key, value in mode_config.items():
            setattr(config_manager.simulation_config, key, value)
        
        try:
            # 创建网络和调度器
            network = config_manager.create_network()
            scheduler = create_scheduler(config_manager)
            
            # 运行仿真
            simulation = config_manager.create_energy_simulation(network, scheduler)
            simulation.simulate()
            
            # 收集统计信息
            stats = simulation.print_statistics()
            results[mode_name] = {
                'avg_variance': stats['avg_variance'],
                'total_loss_energy': stats['total_loss_energy'],
                'energy_efficiency': stats['total_received_energy'] / stats['total_sent_energy'] if stats['total_sent_energy'] > 0 else 0,
                'transfer_count': len([t for t in range(config_manager.simulation_config.time_steps) 
                                      if t in simulation.plans_by_time])
            }
            logger.info(f"✓ {mode_name} 测试完成")
            
        except Exception as e:
            logger.error(f"✗ {mode_name} 测试失败: {str(e)}")
            results[mode_name] = None
    
    # 输出比较结果
    logger.info("\n" + "=" * 60)
    logger.info("传能模式性能比较结果:")
    logger.info("=" * 60)
    for mode_name, result in results.items():
        if result:
            logger.info(f"\n{mode_name}:")
            logger.info(f"  传能次数: {result['transfer_count']} 次")
            logger.info(f"  平均方差: {result['avg_variance']:.4f}")
            logger.info(f"  总能量损失: {result['total_loss_energy']:.4f} J")
            logger.info(f"  能量效率: {result['energy_efficiency']:.2%}")
        else:
            logger.info(f"\n{mode_name}: 测试失败")
    
    return results


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
