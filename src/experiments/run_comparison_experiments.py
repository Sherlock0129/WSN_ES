#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照实验执行脚本
执行四类对照实验：
1. 智能被动传能
2. 信息价值
3. 机会主义上报机制
4. AdaptiveDurationAware
"""

import os
import sys
import shutil
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import ConfigManager
from utils.output_manager import OutputManager


def run_simulation_with_config(config_manager: ConfigManager):
    """
    运行仿真并返回session_dir
    
    Args:
        config_manager: 配置管理器
        
    Returns:
        session_dir: 会话目录路径
    """
    from sim.refactored_main import create_scheduler
    from routing.energy_transfer_routing import set_eetor_config
    from viz.plotter import plot_node_distribution, plot_energy_over_time, plot_center_node_energy
    
    # 创建网络
    network = config_manager.create_network()
    
    # 创建ADCR链路层（如果需要）
    if config_manager.simulation_config.enable_adcr_link_layer:
        network.adcr_link = config_manager.create_adcr_link_layer(network)
    else:
        network.adcr_link = None
    
    # 创建路径信息收集器（如果需要）
    if config_manager.path_collector_config.enable_path_collector:
        from info_collection.physical_center import VirtualCenter
        physical_center = network.get_physical_center()
        if physical_center:
            initial_pos = tuple(physical_center.position)
        else:
            nodes = network.get_regular_nodes() if hasattr(network, 'get_regular_nodes') else network.nodes
            initial_pos = (sum(n.position[0] for n in nodes) / len(nodes),
                          sum(n.position[1] for n in nodes) / len(nodes))
        vc = VirtualCenter(initial_position=initial_pos, enable_logging=True)
        vc.initialize_node_info(network.nodes, initial_time=0)
        network.path_info_collector = config_manager.create_path_collector(vc, physical_center)
    else:
        network.path_info_collector = None
    
    # 创建调度器
    scheduler = create_scheduler(config_manager, network)
    
    # 设置调度器的 path_collector（如果存在）
    if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        scheduler.path_collector = network.path_info_collector
    
    # 设置EETOR配置
    set_eetor_config(config_manager.eetor_config)
    
    # 运行仿真
    simulation = config_manager.create_energy_simulation(network, scheduler)
    session_dir = simulation.session_dir
    
    # 设置虚拟中心归档路径
    if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        archive_path = os.path.join(session_dir, "virtual_center_node_info.csv")
        network.path_info_collector.vc.archive_path = archive_path
    
    simulation.simulate()

    # 保存详细的计划日志
    from utils.logger import get_detailed_plan_logger
    plan_logger = get_detailed_plan_logger(session_dir)
    plan_logger.save_simulation_plans(simulation)
    
    # 生成可视化
    plot_node_distribution(network.nodes, session_dir=session_dir)
    plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
    plot_center_node_energy(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
    simulation.plot_K_history()
    
    return session_dir


def copy_and_rename_result(session_dir: str, exp_name: str, desc: str):
    """
    复制结果图片并重命名文件夹
    
    Args:
        session_dir: 原始会话目录
        exp_name: 实验名称（用于文件命名）
        desc: 实验描述（中文）
    """
    if not os.path.exists(session_dir):
        print(f"警告：会话目录不存在: {session_dir}")
        return
    
    # 复制center_node_energy_over_time.png到备选图目录
    src_image = os.path.join(session_dir, "center_node_energy_over_time.png")
    if os.path.exists(src_image):
        dest_dir = "data/备选图"
        os.makedirs(dest_dir, exist_ok=True)
        dest_image = os.path.join(dest_dir, f"center_node_energy_over_time_{exp_name}.png")
        shutil.copy2(src_image, dest_image)
        print(f"已复制图片: {dest_image}")
    else:
        print(f"警告：图片不存在: {src_image}")
    
    # 重命名会话目录
    parent_dir = os.path.dirname(session_dir)
    new_dir_name = f"{os.path.basename(session_dir)}_{exp_name}"
    new_dir_path = os.path.join(parent_dir, new_dir_name)
    
    if os.path.exists(new_dir_path):
        print(f"警告：目标目录已存在: {new_dir_path}")
    else:
        os.rename(session_dir, new_dir_path)
        print(f"已重命名目录: {new_dir_path}")


# ==================== 实验1：智能被动传能 ====================

def run_exp1_baseline_passive():
    """实验1-基准：智能被动传能（默认配置）"""
    print("\n" + "="*80)
    print("实验1-基准：智能被动传能")
    print("="*80)
    
    config_manager = ConfigManager()
    # 确保使用智能被动模式（默认配置）
    config_manager.simulation_config.passive_mode = True
    config_manager.simulation_config.check_interval = 1
    
    print(f"配置:")
    print(f"  - passive_mode = {config_manager.simulation_config.passive_mode}")
    print(f"  - check_interval = {config_manager.simulation_config.check_interval}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp1_baseline_passive", "智能被动传能")
    
    print("实验1-基准 完成\n")
    return session_dir


def run_exp1_active_60min():
    """实验1-对照：60分钟主动传能"""
    print("\n" + "="*80)
    print("实验1-对照：60分钟主动传能")
    print("="*80)
    
    config_manager = ConfigManager()
    # 禁用智能被动模式，使用定时主动传能
    config_manager.simulation_config.passive_mode = False
    # 注意：passive_mode=False时，系统会每60分钟触发一次（在PassiveTransferManager中实现）
    
    print(f"配置:")
    print(f"  - passive_mode = {config_manager.simulation_config.passive_mode}")
    print(f"  - 触发方式: 每60分钟定时触发")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp1_active_60min", "60分钟主动传能")
    
    print("实验1-对照 完成\n")
    return session_dir


# ==================== 实验2：信息价值 ====================

def run_exp2_baseline_with_info():
    """实验2-基准：启用信息价值（默认配置）"""
    print("\n" + "="*80)
    print("实验2-基准：启用信息价值")
    print("="*80)
    
    config_manager = ConfigManager()
    # 确保信息感知路由和调度器信息奖励都启用（默认配置）
    config_manager.eetor_config.enable_info_aware_routing = True
    # 调度器的信息奖励通过duration_w_info控制（默认0.1）
    
    print(f"配置:")
    print(f"  - enable_info_aware_routing = {config_manager.eetor_config.enable_info_aware_routing}")
    print(f"  - duration_w_info = {config_manager.scheduler_config.duration_w_info}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp2_baseline_with_info", "启用信息价值")
    
    print("实验2-基准 完成\n")
    return session_dir


def run_exp2_no_info_reward():
    """实验2-对照1：去掉调度器信息价值奖励"""
    print("\n" + "="*80)
    print("实验2-对照1：去掉调度器信息价值奖励")
    print("="*80)
    
    config_manager = ConfigManager()
    # 禁用调度器中的信息奖励（将权重设为0）
    config_manager.scheduler_config.duration_w_info = 0.0
    # 保持路由中的信息感知（用于对比）
    config_manager.eetor_config.enable_info_aware_routing = True
    
    print(f"配置:")
    print(f"  - duration_w_info = {config_manager.scheduler_config.duration_w_info} (已禁用)")
    print(f"  - enable_info_aware_routing = {config_manager.eetor_config.enable_info_aware_routing} (保持启用)")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp2_no_info_reward", "去掉调度器信息价值奖励")
    
    print("实验2-对照1 完成\n")
    return session_dir


def run_exp2_no_info_routing():
    """实验2-对照2：去掉路由信息感知"""
    print("\n" + "="*80)
    print("实验2-对照2：去掉路由信息感知")
    print("="*80)
    
    config_manager = ConfigManager()
    # 禁用路由中的信息感知
    config_manager.eetor_config.enable_info_aware_routing = False
    # 保持调度器中的信息奖励（用于对比）
    config_manager.scheduler_config.duration_w_info = 0.1
    
    print(f"配置:")
    print(f"  - enable_info_aware_routing = {config_manager.eetor_config.enable_info_aware_routing} (已禁用)")
    print(f"  - duration_w_info = {config_manager.scheduler_config.duration_w_info} (保持启用)")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp2_no_info_routing", "去掉路由信息感知")
    
    print("实验2-对照2 完成\n")
    return session_dir


def run_exp2_no_info_both():
    """实验2-对照3：同时去掉路由和调度的信息价值"""
    print("\n" + "="*80)
    print("实验2-对照3：同时去掉路由和调度的信息价值")
    print("="*80)
    
    config_manager = ConfigManager()
    # 同时禁用路由和调度中的信息价值
    config_manager.eetor_config.enable_info_aware_routing = False
    config_manager.scheduler_config.duration_w_info = 0.0
    
    print(f"配置:")
    print(f"  - enable_info_aware_routing = {config_manager.eetor_config.enable_info_aware_routing} (已禁用)")
    print(f"  - duration_w_info = {config_manager.scheduler_config.duration_w_info} (已禁用)")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp2_no_info_both", "同时去掉路由和调度的信息价值")
    
    print("实验2-对照3 完成\n")
    return session_dir


# ==================== 实验3：机会主义上报机制 ====================

def run_exp3_baseline_opportunistic():
    """实验3-基准：机会主义上报机制（默认配置）"""
    print("\n" + "="*80)
    print("实验3-基准：机会主义上报机制")
    print("="*80)
    
    config_manager = ConfigManager()
    # 确保使用PathCollector和机会主义上报（默认配置）
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = True
    config_manager.path_collector_config.enable_delayed_reporting = True
    # 禁用ADCR
    config_manager.simulation_config.enable_adcr_link_layer = False
    
    print(f"配置:")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_opportunistic_info_forwarding = {config_manager.path_collector_config.enable_opportunistic_info_forwarding}")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting}")
    print(f"  - enable_adcr_link_layer = {config_manager.simulation_config.enable_adcr_link_layer}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp3_baseline_opportunistic", "机会主义上报机制")
    
    print("实验3-基准 完成\n")
    return session_dir


def run_exp3_adcr():
    """实验3-对照1：使用ADCR协议"""
    print("\n" + "="*80)
    print("实验3-对照1：使用ADCR协议")
    print("="*80)
    
    config_manager = ConfigManager()
    # 启用ADCR，禁用PathCollector
    config_manager.simulation_config.enable_adcr_link_layer = True
    config_manager.path_collector_config.enable_path_collector = False
    
    print(f"配置:")
    print(f"  - enable_adcr_link_layer = {config_manager.simulation_config.enable_adcr_link_layer}")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - ADCR round_period = {config_manager.adcr_config.round_period} 分钟")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp3_adcr", "使用ADCR协议")
    
    print("实验3-对照1 完成\n")
    return session_dir


def run_exp3_direct_report():
    """实验3-对照2：每个人直接给中心发（禁用机会主义，立即上报）"""
    print("\n" + "="*80)
    print("实验3-对照2：每个人直接给中心发")
    print("="*80)
    
    config_manager = ConfigManager()
    # 使用PathCollector，但禁用机会主义和延迟上报（立即上报）
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = False
    config_manager.path_collector_config.enable_delayed_reporting = False
    # 禁用ADCR
    config_manager.simulation_config.enable_adcr_link_layer = False
    
    print(f"配置:")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_opportunistic_info_forwarding = {config_manager.path_collector_config.enable_opportunistic_info_forwarding} (已禁用)")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting} (已禁用)")
    print(f"  - 模式: 立即上报，无机会主义")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp3_direct_report", "每个人直接给中心发")
    
    print("实验3-对照2 完成\n")
    return session_dir


# ==================== 实验4：AdaptiveDurationAware ====================

def run_exp4_baseline_adaptive_duration():
    """实验4-基准：AdaptiveDurationAwareLyapunovScheduler（默认配置）"""
    print("\n" + "="*80)
    print("实验4-基准：AdaptiveDurationAwareLyapunovScheduler")
    print("="*80)
    
    config_manager = ConfigManager()
    # 确保使用AdaptiveDurationAwareLyapunovScheduler（默认配置）
    config_manager.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"
    
    print(f"配置:")
    print(f"  - scheduler_type = {config_manager.scheduler_config.scheduler_type}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp4_baseline_adaptive_duration", "AdaptiveDurationAwareLyapunovScheduler")
    
    print("实验4-基准 完成\n")
    return session_dir


def run_exp4_traditional_lyapunov():
    """实验4-对照1：传统LyapunovScheduler"""
    print("\n" + "="*80)
    print("实验4-对照1：传统LyapunovScheduler")
    print("="*80)
    
    config_manager = ConfigManager()
    # 使用传统LyapunovScheduler（无自适应，无时长优化）
    config_manager.scheduler_config.scheduler_type = "LyapunovScheduler"
    
    print(f"配置:")
    print(f"  - scheduler_type = {config_manager.scheduler_config.scheduler_type}")
    print(f"  - 特点: 无自适应参数调整，无传输时长优化")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp4_traditional_lyapunov", "传统LyapunovScheduler")
    
    print("实验4-对照1 完成\n")
    return session_dir


def run_exp4_duration_aware_only():
    """实验4-对照2：单纯DurationAwareLyapunovScheduler（无自适应）"""
    print("\n" + "="*80)
    print("实验4-对照2：DurationAwareLyapunovScheduler")
    print("="*80)
    
    config_manager = ConfigManager()
    # 使用DurationAwareLyapunovScheduler（有时长优化，但无自适应参数调整）
    config_manager.scheduler_config.scheduler_type = "DurationAwareLyapunovScheduler"
    
    print(f"配置:")
    print(f"  - scheduler_type = {config_manager.scheduler_config.scheduler_type}")
    print(f"  - 特点: 有传输时长优化，但无自适应参数调整")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp4_duration_aware_only", "DurationAwareLyapunovScheduler")
    
    print("实验4-对照2 完成\n")
    return session_dir


# ==================== 主函数 ====================

def main():
    """主函数：只运行E2的基准和no_info_reward实验"""
    print("="*80)
    print("运行 E2 对照实验 (基准 vs. 无信息奖励)")
    print("="*80)

    # 实验2：信息价值
    print("\n" + "="*80)
    print("开始实验2：信息价值")
    print("="*80)
    print("运行 E2-基准（启用信息价值）...")
    run_exp2_baseline_with_info()
    
    print("运行 E2-对照1（去掉调度器信息价值奖励）...")
    run_exp2_no_info_reward()

    print("\n" + "="*80)
    print("E2部分实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()

