#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验执行脚本
逐个执行消融实验，修改配置参数，运行仿真，并处理结果文件
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
    
    # 生成可视化
    plot_node_distribution(network.nodes, session_dir=session_dir)
    plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
    plot_center_node_energy(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
    simulation.plot_K_history()
    
    return session_dir


def copy_and_rename_result(session_dir: str, ablation_name: str, ablation_desc: str):
    """
    复制center_node_energy_over_time.png到备选图文件夹，并重命名
    同时重命名原始文件夹，标出是什么消融实验
    
    Args:
        session_dir: 原始会话目录路径
        ablation_name: 消融实验名称（如 "ablation_1"）
        ablation_desc: 消融实验描述（如 "去除AOEI价格信号"）
    """
    # 源文件路径
    source_file = os.path.join(session_dir, "center_node_energy_over_time.png")
    
    # 目标目录
    target_dir = "data/备选图"
    os.makedirs(target_dir, exist_ok=True)
    
    # 目标文件名（使用拼音或英文，避免中文编码问题）
    # 将中文描述转换为拼音简写
    desc_map = {
        "去除AOEI价格信号": "no_AOEI",
        "去除InfoNode数字孪生": "no_InfoNode",
        "去除去重动态等待": "no_dedup_wait",
        "去除EETOR路由约束": "no_EETOR",
        "去除公平保护": "no_fairness"
    }
    desc_key = desc_map.get(ablation_desc, ablation_desc.replace("去除", "no_").replace("动态", "").replace(" ", "_"))
    target_filename = f"center_node_energy_over_time_{ablation_name}_{desc_key}.png"
    target_file = os.path.join(target_dir, target_filename)
    
    # 复制文件
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        print(f"已复制图片到: {target_file}")
    else:
        print(f"警告: 源文件不存在: {source_file}")
    
    # 重命名原始文件夹（使用英文标识，避免中文编码问题）
    parent_dir = os.path.dirname(session_dir)
    old_folder_name = os.path.basename(session_dir)
    desc_map = {
        "去除AOEI价格信号": "no_AOEI",
        "去除InfoNode数字孪生": "no_InfoNode",
        "去除去重动态等待": "no_dedup_wait",
        "去除EETOR路由约束": "no_EETOR",
        "去除公平保护": "no_fairness"
    }
    desc_key = desc_map.get(ablation_desc, ablation_desc.replace("去除", "no_").replace("动态", "").replace(" ", "_"))
    new_folder_name = f"{old_folder_name}_{ablation_name}_{desc_key}"
    new_folder_path = os.path.join(parent_dir, new_folder_name)
    
    if os.path.exists(session_dir):
        os.rename(session_dir, new_folder_path)
        print(f"已重命名文件夹: {new_folder_name}")
        return new_folder_path
    else:
        print(f"警告: 文件夹不存在: {session_dir}")
        return session_dir


def run_ablation_1():
    """Ablation-1: 去除AOEI价格信号 - 替换为固定阈值触发"""
    print("\n" + "="*80)
    print("开始 Ablation-1: 去除AOEI价格信号")
    print("="*80)
    
    # 修改配置
    config_manager = ConfigManager()
    config_manager.simulation_config.passive_mode = False  # 禁用被动模式，改为定时触发
    
    print(f"配置修改: passive_mode = {config_manager.simulation_config.passive_mode}")
    
    # 运行仿真
    session_dir = run_simulation_with_config(config_manager)
    
    # 处理结果
    copy_and_rename_result(session_dir, "ablation_1", "去除AOEI价格信号")
    
    print("Ablation-1 完成\n")
    return session_dir


def run_ablation_2():
    """Ablation-2: 去除InfoNode数字孪生 - 禁用PathCollector"""
    print("\n" + "="*80)
    print("开始 Ablation-2: 去除InfoNode数字孪生")
    print("="*80)
    
    # 修改配置
    config_manager = ConfigManager()
    config_manager.path_collector_config.enable_path_collector = False  # 禁用PathCollector
    
    print(f"配置修改: enable_path_collector = {config_manager.path_collector_config.enable_path_collector}")
    
    # 运行仿真
    session_dir = run_simulation_with_config(config_manager)
    
    # 处理结果
    copy_and_rename_result(session_dir, "ablation_2", "去除InfoNode数字孪生")
    
    print("Ablation-2 完成\n")
    return session_dir


def run_ablation_3():
    """Ablation-3: 去除去重/动态等待 - 禁用信息去重机制"""
    print("\n" + "="*80)
    print("开始 Ablation-3: 去除去重/动态等待")
    print("="*80)
    
    # 修改配置
    config_manager = ConfigManager()
    config_manager.path_collector_config.enable_info_volume_accumulation = False  # 禁用信息量累积
    config_manager.path_collector_config.enable_delayed_reporting = False  # 禁用延迟上报
    config_manager.path_collector_config.enable_adaptive_wait_time = False  # 禁用自适应等待时间
    
    print(f"配置修改:")
    print(f"  - enable_info_volume_accumulation = {config_manager.path_collector_config.enable_info_volume_accumulation}")
    print(f"  - enable_delayed_reporting = {config_manager.path_collector_config.enable_delayed_reporting}")
    print(f"  - enable_adaptive_wait_time = {config_manager.path_collector_config.enable_adaptive_wait_time}")
    
    # 运行仿真
    session_dir = run_simulation_with_config(config_manager)
    
    # 处理结果
    copy_and_rename_result(session_dir, "ablation_3", "去除去重动态等待")
    
    print("Ablation-3 完成\n")
    return session_dir


def run_ablation_4():
    """Ablation-4: 去除EETOR路由约束 - 允许任意低效路径"""
    print("\n" + "="*80)
    print("开始 Ablation-4: 去除EETOR路由约束")
    print("="*80)
    
    # 修改配置
    config_manager = ConfigManager()
    config_manager.eetor_config.min_efficiency = 0.0  # 移除效率阈值
    config_manager.network_config.max_hops = 100  # 移除跳数限制（设置为很大的值）
    
    print(f"配置修改:")
    print(f"  - min_efficiency = {config_manager.eetor_config.min_efficiency}")
    print(f"  - max_hops = {config_manager.network_config.max_hops}")
    
    # 运行仿真
    session_dir = run_simulation_with_config(config_manager)
    
    # 处理结果
    copy_and_rename_result(session_dir, "ablation_4", "去除EETOR路由约束")
    
    print("Ablation-4 完成\n")
    return session_dir


def run_ablation_6():
    """Ablation-6: 去除公平保护 - 禁用弱势权重"""
    print("\n" + "="*80)
    print("开始 Ablation-6: 去除公平保护")
    print("="*80)
    
    # 修改配置
    config_manager = ConfigManager()
    config_manager.simulation_config.critical_ratio = 0.0  # 禁用公平保护
    
    print(f"配置修改: critical_ratio = {config_manager.simulation_config.critical_ratio}")
    
    # 运行仿真
    session_dir = run_simulation_with_config(config_manager)
    
    # 处理结果
    copy_and_rename_result(session_dir, "ablation_6", "去除公平保护")
    
    print("Ablation-6 完成\n")
    return session_dir


def main():
    """主函数：执行所有消融实验"""
    print("="*80)
    print("消融实验执行脚本")
    print("="*80)
    print("将执行以下消融实验:")
    print("1. Ablation-1: 去除AOEI价格信号")
    print("2. Ablation-2: 去除InfoNode数字孪生")
    print("3. Ablation-3: 去除去重/动态等待")
    print("4. Ablation-4: 去除EETOR路由约束")
    print("5. Ablation-6: 去除公平保护")
    print("="*80)
    
    results = {}
    
    try:
        # 执行各个消融实验
        results['ablation_1'] = run_ablation_1()
        results['ablation_2'] = run_ablation_2()
        results['ablation_3'] = run_ablation_3()
        results['ablation_4'] = run_ablation_4()
        results['ablation_6'] = run_ablation_6()
        
        print("\n" + "="*80)
        print("所有消融实验完成！")
        print("="*80)
        print("结果文件已复制到: data/备选图/")
        print("原始文件夹已重命名，包含消融实验标识")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

