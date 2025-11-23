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
    
    # 验证信息收集器配置的互斥性
    config_manager.validate_info_collector_config()
    
    physical_center = network.get_physical_center()
    nodes_for_init = (network.get_regular_nodes() if hasattr(network, 'get_regular_nodes')
                      else network.nodes)
    vc = None

    def ensure_virtual_center():
        nonlocal vc
        if vc is None:
            if physical_center:
                initial_pos = tuple(physical_center.position)
            else:
                initial_pos = (
                    sum(n.position[0] for n in nodes_for_init) / len(nodes_for_init),
                    sum(n.position[1] for n in nodes_for_init) / len(nodes_for_init)
                )
            vc = config_manager.create_virtual_center(
                initial_position=initial_pos,
                enable_logging=True
            )
            vc.initialize_node_info(network.nodes, initial_time=0)
        return vc

    if config_manager.path_collector_config.enable_path_collector:
        path_vc = ensure_virtual_center()
        network.path_info_collector = config_manager.create_path_collector(path_vc, physical_center)
    else:
        network.path_info_collector = None

    if config_manager.periodic_collector_config.enable_periodic_collector:
        periodic_vc = ensure_virtual_center()
        network.periodic_info_collector = config_manager.create_periodic_collector(periodic_vc, physical_center)
    else:
        network.periodic_info_collector = None
    
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
    archive_path = os.path.join(session_dir, "virtual_center_node_info.csv")
    if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        network.path_info_collector.vc.archive_path = archive_path
    if hasattr(network, 'periodic_info_collector') and network.periodic_info_collector is not None:
        network.periodic_info_collector.vc.archive_path = archive_path
    
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
    """实验3-对照2：每个人直接给中心发（使用定期上报模式，每个节点每隔360分钟向中心发送一次信息）"""
    print("\n" + "="*80)
    print("实验3-对照2：每个人直接给中心发（定期上报模式）")
    print("="*80)
    
    config_manager = ConfigManager()
    # 使用定期上报收集器，禁用路径收集器和ADCR
    config_manager.periodic_collector_config.enable_periodic_collector = True
    config_manager.periodic_collector_config.report_interval = 360  # 360分钟（6小时）上报一次
    config_manager.path_collector_config.enable_path_collector = False
    config_manager.simulation_config.enable_adcr_link_layer = False
    
    print(f"配置:")
    print(f"  - enable_periodic_collector = {config_manager.periodic_collector_config.enable_periodic_collector}")
    print(f"  - report_interval = {config_manager.periodic_collector_config.report_interval} 分钟（{config_manager.periodic_collector_config.report_interval // 60} 小时）")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector} (已禁用)")
    print(f"  - enable_adcr_link_layer = {config_manager.simulation_config.enable_adcr_link_layer} (已禁用)")
    print(f"  - 模式: 定期上报，每个节点每隔{config_manager.periodic_collector_config.report_interval}分钟（{config_manager.periodic_collector_config.report_interval // 60}小时）向中心发送一次信息，不依赖能量传输路径")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp3_direct_report", "每个人直接给中心发")
    
    print("实验3-对照2 完成\n")
    return session_dir


def run_exp3_periodic_report():
    """实验3-对照3：定期上报模式（每个节点每隔60分钟向中心发送一次信息）"""
    print("\n" + "="*80)
    print("实验3-对照3：定期上报模式")
    print("="*80)
    
    config_manager = ConfigManager()
    # 使用定期上报收集器，禁用路径收集器和ADCR
    config_manager.periodic_collector_config.enable_periodic_collector = True
    config_manager.periodic_collector_config.report_interval = 60  # 60分钟上报一次
    config_manager.path_collector_config.enable_path_collector = False
    config_manager.simulation_config.enable_adcr_link_layer = False
    
    print(f"配置:")
    print(f"  - enable_periodic_collector = {config_manager.periodic_collector_config.enable_periodic_collector}")
    print(f"  - report_interval = {config_manager.periodic_collector_config.report_interval} 分钟")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector} (已禁用)")
    print(f"  - enable_adcr_link_layer = {config_manager.simulation_config.enable_adcr_link_layer} (已禁用)")
    print(f"  - 模式: 定期上报，不依赖能量传输路径")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp3_periodic_report", "定期上报模式")
    
    print("实验3-对照3 完成\n")
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


# ==================== 实验5：AOEI动态/静态上限 ====================

def run_exp5_dynamic_aoei_cap():
    """实验5-基准：启用AOEI自适应等待上限"""
    print("\n" + "="*80)
    print("实验5-基准：AOEI动态上限（自适应等待）")
    print("="*80)
    
    config_manager = ConfigManager()
    config_manager.info_config.force_report_on_stale = True
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = True
    config_manager.path_collector_config.enable_adaptive_wait_time = True
    # 基于信息量缩放动态上限，保持延迟上报机制
    config_manager.path_collector_config.enable_delayed_reporting = True
    config_manager.path_collector_config.max_wait_time = 100  # 更严格的等待上限
    
    print("配置:")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_opportunistic_info_forwarding = {config_manager.path_collector_config.enable_opportunistic_info_forwarding}")
    print(f"  - enable_adaptive_wait_time = {config_manager.path_collector_config.enable_adaptive_wait_time}")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting}")
    print(f"  - max_wait_time = {config_manager.path_collector_config.max_wait_time} 分钟")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp5_dynamic_aoei_cap", "AOEI动态上限")
    
    print("实验5-基准 完成\n")
    return session_dir


def run_exp5_static_aoei_cap():
    """实验5-对照：禁用AOEI自适应等待上限（静态阈值）"""
    print("\n" + "="*80)
    print("实验5-对照：AOEI静态上限（固定等待）")
    print("="*80)
    
    config_manager = ConfigManager()
    config_manager.info_config.force_report_on_stale = True
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = True
    config_manager.path_collector_config.enable_adaptive_wait_time = False
    config_manager.path_collector_config.enable_delayed_reporting = True
    # 统一使用静态上限，保持其他参数一致
    config_manager.path_collector_config.max_wait_time = 200
    
    print("配置:")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_opportunistic_info_forwarding = {config_manager.path_collector_config.enable_opportunistic_info_forwarding}")
    print(f"  - enable_adaptive_wait_time = {config_manager.path_collector_config.enable_adaptive_wait_time} (已禁用)")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting}")
    print(f"  - max_wait_time = {config_manager.path_collector_config.max_wait_time} 分钟（固定阈值）")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp5_static_aoei_cap", "AOEI静态上限")
    
    print("实验5-对照 完成\n")
    return session_dir


# ==================== 实验6：ALDP vs 传统Lyapunov ====================

def run_exp6_aldp_scheduler():
    """实验6-基准：AdaptiveDurationAwareLyapunovScheduler（ALDP）"""
    print("\n" + "="*80)
    print("实验6-基准：ALDP（AdaptiveDurationAwareLyapunovScheduler）")
    print("="*80)
    
    config_manager = ConfigManager()
    config_manager.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"
    
    print("配置:")
    print(f"  - scheduler_type = {config_manager.scheduler_config.scheduler_type}")
    print(f"  - duration range = {config_manager.scheduler_config.duration_min}~{config_manager.scheduler_config.duration_max} 分钟")
    print(f"  - V初始值 = {config_manager.scheduler_config.adaptive_lyapunov_v}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp6_aldp", "ALDP调度器")
    
    print("实验6-基准 完成\n")
    return session_dir


def run_exp6_traditional_lyapunov():
    """实验6-对照：传统LyapunovScheduler"""
    print("\n" + "="*80)
    print("实验6-对照：传统LyapunovScheduler")
    print("="*80)
    
    config_manager = ConfigManager()
    config_manager.scheduler_config.scheduler_type = "LyapunovScheduler"
    
    print("配置:")
    print(f"  - scheduler_type = {config_manager.scheduler_config.scheduler_type}")
    print(f"  - V = {config_manager.scheduler_config.lyapunov_v}")
    print(f"  - K = {config_manager.scheduler_config.lyapunov_k}")
    
    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp6_traditional_lyapunov", "传统Lyapunov")
    
    print("实验6-对照 完成\n")
    return session_dir


# ==================== 实验7：能量估算影响 ====================

def run_exp7_with_estimation():
    """实验7-基准：启用虚拟中心能量估算"""
    print("\n" + "="*80)
    print("实验7-基准：启用能量估算")
    print("="*80)

    config_manager = ConfigManager()
    config_manager.info_config.enable_energy_estimation = True
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.periodic_collector_config.enable_periodic_collector = False

    print("配置:")
    print(f"  - enable_energy_estimation = {config_manager.info_config.enable_energy_estimation}")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_periodic_collector = {config_manager.periodic_collector_config.enable_periodic_collector}")

    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp7_with_estimation", "启用能量估算")

    print("实验7-基准 完成\n")
    return session_dir


def run_exp7_without_estimation():
    """实验7-对照：禁用能量估算，仅依赖上报信息"""
    print("\n" + "="*80)
    print("实验7-对照：关闭能量估算，路径+强制上报")
    print("="*80)

    config_manager = ConfigManager()
    config_manager.info_config.enable_energy_estimation = False
    config_manager.path_collector_config.enable_path_collector = True
    config_manager.path_collector_config.enable_delayed_reporting = False  # 路径信息立即上报
    config_manager.periodic_collector_config.enable_periodic_collector = True
    config_manager.periodic_collector_config.report_interval = 120  # 每2小时强制上报一次

    print("配置:")
    print(f"  - enable_energy_estimation = {config_manager.info_config.enable_energy_estimation} (已禁用)")
    print(f"  - enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting}")
    print(f"  - enable_periodic_collector = {config_manager.periodic_collector_config.enable_periodic_collector}")
    print(f"  - report_interval = {config_manager.periodic_collector_config.report_interval} 分钟")

    session_dir = run_simulation_with_config(config_manager)
    copy_and_rename_result(session_dir, "exp7_without_estimation", "禁用能量估算")

    print("实验7-对照 完成\n")
    return session_dir


# ==================== 主函数 ====================

def main():
    """主函数：仅运行实验5（动态/静态 AOEI 上限对比）"""
    print("="*80)
    print("运行实验5：AOEI动态上限 vs 静态上限")
    print("="*80)
    run_exp5_dynamic_aoei_cap()
    run_exp5_static_aoei_cap()


if __name__ == "__main__":
    main()

