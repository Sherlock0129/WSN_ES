#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Related-Work 对照实验脚本（仅反面 3+1）

统一基于 config/simulation_config_2.py 的基准配置运行四个实验：
1) 阶段1：阈值法 ThresholdScheduler（最近优先，固定 E_char，duration=1）
2) 阶段2：传统 Lyapunov（V=0.2）
3) 阶段3：传统 Lyapunov（V=1.0）
4) 阶段4：强化学习 DQN（推理，使用 ../sim/dqn_model.pth）

说明：不包含“正面（我们的方案）”实验，该部分由外部单独运行。
"""

import os
import sys
import shutil

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用第二套基准配置
from config.simulation_config_2 import ConfigManager

from utils.output_manager import OutputManager


def run_simulation_with_config(config_manager: ConfigManager):
    """
    使用给定配置运行仿真，返回会话目录。
    关键点：统一使用 ADCR 进行信息上传；其余流程参考 refactored_main。
    """
    from sim.refactored_main import create_scheduler
    from routing.energy_transfer_routing import set_eetor_config
    from viz.plotter import plot_node_distribution, plot_energy_over_time, plot_center_node_energy

    # 强制启用 ADCR 信息上传；禁用 PathCollector/PeriodicCollector
    config_manager.simulation_config.enable_adcr_link_layer = True
    config_manager.path_collector_config.enable_path_collector = False
    config_manager.periodic_collector_config.enable_periodic_collector = False
    # 主动模式：定时传能间隔（分钟）
    config_manager.simulation_config.active_transfer_interval = 20

    # 创建网络
    network = config_manager.create_network()

    # ADCR 链路层（强制启用）
    network.adcr_link = config_manager.create_adcr_link_layer(network)

    # 校验信息收集器互斥（此时两者均为 False，不会冲突）
    config_manager.validate_info_collector_config()

    # 路径/定期信息收集器（禁用）
    network.path_info_collector = None
    network.periodic_info_collector = None

    # 创建调度器
    scheduler = create_scheduler(config_manager, network)

    # 传入 path_collector（此处无）
    if hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        scheduler.path_collector = network.path_info_collector

    # 设置 EETOR 配置
    set_eetor_config(config_manager.eetor_config)

    # 运行仿真
    simulation = config_manager.create_energy_simulation(network, scheduler)
    session_dir = simulation.session_dir

    # 为 ADCR 的 VC 设置归档路径（参考 refactored_main）
    if hasattr(network, 'adcr_link') and network.adcr_link is not None:
        network.adcr_link.set_archive_path(session_dir)

    simulation.simulate()

    # 保存详细计划日志
    from utils.logger import get_detailed_plan_logger
    plan_logger = get_detailed_plan_logger(session_dir)
    plan_logger.save_simulation_plans(simulation)

    # 可视化
    try:
        plot_node_distribution(network.nodes, session_dir=session_dir)
        plot_energy_over_time(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
        plot_center_node_energy(network.nodes, simulation.result_manager.get_results(), session_dir=session_dir)
        simulation.plot_K_history()
    except Exception:
        pass

    # 仿真结束后强制刷新 VC 归档
    if hasattr(network, 'adcr_link') and network.adcr_link is not None:
        network.adcr_link.vc.force_flush_archive()

    return session_dir


def copy_and_rename_result(session_dir: str, exp_name: str):
    """复制关键图并重命名会话目录。"""
    if not os.path.exists(session_dir):
        print(f"警告：会话目录不存在: {session_dir}")
        return

    # 复制中心节点能量图
    src_image = os.path.join(session_dir, "center_node_energy_over_time.png")
    if os.path.exists(src_image):
        dest_dir = "data/备选图"
        os.makedirs(dest_dir, exist_ok=True)
        dest_image = os.path.join(dest_dir, f"center_node_energy_over_time_{exp_name}.png")
        shutil.copy2(src_image, dest_image)
        print(f"已复制图片: {dest_image}")

    # 重命名目录
    parent_dir = os.path.dirname(session_dir)
    new_dir_name = f"{os.path.basename(session_dir)}_{exp_name}"
    new_dir_path = os.path.join(parent_dir, new_dir_name)
    if not os.path.exists(new_dir_path):
        os.rename(session_dir, new_dir_path)
        print(f"已重命名目录: {new_dir_path}")
    else:
        print(f"警告：目标目录已存在: {new_dir_path}")


# ==================== 四个反面实验 ====================

def run_rw_stage2_threshold():
    print("\n" + "="*80)
    print("RelatedWork 阶段2：阈值法 ThresholdScheduler")
    print("="*80)

    cfg = ConfigManager()
    cfg.scheduler_config.scheduler_type = "ThresholdScheduler"

    print(f"配置: scheduler_type = {cfg.scheduler_config.scheduler_type}")
    session_dir = run_simulation_with_config(cfg)
    copy_and_rename_result(session_dir, "rw_stage2_threshold")
    return session_dir


def run_rw_stage3_lyapunov_v02():
    print("\n" + "="*80)
    print("RelatedWork 阶段3：Lyapunov V=0.2")
    print("="*80)

    cfg = ConfigManager()
    cfg.scheduler_config.scheduler_type = "LyapunovScheduler"
    cfg.scheduler_config.lyapunov_v = 0.2

    print(f"配置: scheduler_type = {cfg.scheduler_config.scheduler_type}, V = {cfg.scheduler_config.lyapunov_v}")
    session_dir = run_simulation_with_config(cfg)
    copy_and_rename_result(session_dir, "rw_stage2-3_lyapunov_v02")
    return session_dir


def run_rw_stage3_lyapunov_v10():
    print("\n" + "="*80)
    print("RelatedWork 阶段3：Lyapunov V=1.0（偏保守）")
    print("="*80)

    cfg = ConfigManager()
    cfg.scheduler_config.scheduler_type = "LyapunovScheduler"
    cfg.scheduler_config.lyapunov_v = 1.0

    print(f"配置: scheduler_type = {cfg.scheduler_config.scheduler_type}, V = {cfg.scheduler_config.lyapunov_v}")
    session_dir = run_simulation_with_config(cfg)
    copy_and_rename_result(session_dir, "rw_stage3_lyapunov_v10")
    return session_dir


def run_rw_stage4_dqn():
    print("\n" + "="*80)
    print("RelatedWork 阶段4：DQN 推理（使用已训练模型）")
    print("="*80)

    cfg = ConfigManager()
    # 启用 DQN，将覆盖 scheduler_type
    cfg.scheduler_config.enable_dqn = True
    cfg.scheduler_config.dqn_training_mode = False

    # 模型路径：相对本脚本 ../sim/dqn_model.pth
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(script_dir, "..", "sim", "dqn_model.pth"))
    cfg.scheduler_config.dqn_model_path = model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 DQN 模型: {model_path}（请放置模型或修改路径）")

    print(f"配置: enable_dqn = {cfg.scheduler_config.enable_dqn}, training_mode = {cfg.scheduler_config.dqn_training_mode}")
    print(f"模型: {cfg.scheduler_config.dqn_model_path}")

    session_dir = run_simulation_with_config(cfg)
    copy_and_rename_result(session_dir, "rw_stage4_dqn")
    return session_dir


# ==================== 主函数 ====================

def main():
    print("="*80)
    print("运行 Related-Work 对照实验（4 项，基于 simulation_config_2.py）")
    print("="*80)

    run_rw_stage2_threshold()
    run_rw_stage3_lyapunov_v02()
    run_rw_stage4_dqn()

    print("\n" + "="*80)
    print("Related-Work 对照实验全部完成！")
    print("="*80)


if __name__ == "__main__":
    main()
