#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的主入口文件 - 支持单次仿真和并行仿真
"""

import os
from typing import Optional

from wsn_et.viz import plotter
from wsn_et.experiments.experiment_manager import create_simulation_runner, create_default_config

def save_all_plans_to_file(simulation, output_file="all_plans.txt"):
    """保存所有计划到文件（适配新的 DataCollector 架构）"""
    data_collector = simulation.get_data_collector()
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== 整个模拟的能量传输计划记录 ===\n\n")
        
        # 从 DataCollector 获取决策记录
        decision_records = data_collector.full_decision_history
        
        for record in decision_records:
            t = record["time_step"]
            plans = record["plans"]
            node_energies = record["node_energies"]
            
            if not plans:
                continue
                
            f.write(f"t={t}\n")
            
            if node_energies:
                f.write("  [NODE_ENERGIES]\n")
                for node_id in sorted(node_energies.keys()):
                    energy = node_energies[node_id]
                    f.write(f"    Node {node_id}: {energy:.2f}J\n")
            
            for i, plan in enumerate(plans):
                receiver_id = plan.get("receiver_id")
                donor_id = plan.get("donor_id")
                path_ids = plan.get("path_ids", [])
                distance = plan.get("distance", 0.0)
                energy_sent = plan.get("energy_sent", 0.0)
                
                path_str = "->".join(str(n) for n in path_ids) if path_ids else ""
                line = f"  [SELECTED] d={donor_id}, r={receiver_id}, path={path_str}, dist={distance:.2f}, energy_sent={energy_sent:.2f}\n"
                f.write(line)
            
            f.write("\n")
        
        f.write("=== 记录结束 ===\n")
    
    print(f"已保存 {len(decision_records)} 个时间步的plans到 {output_file}")


def run_single_simulation(config, output_dir: str = "adcr"):
    """运行单次仿真"""
    print("=== 单次仿真模式 ===")
    
    # 创建仿真运行器
    runner = create_simulation_runner(config)
    
    # 运行仿真
    result = runner.run_single_simulation(output_dir=output_dir)
    
    if result["status"] == "failed":
        print(f"仿真失败: {result['error']}")
        return None
    
    simulation = result["simulation"]
    network = result["network"]
    
    print("仿真完成！")
    
    # 保存计划文件
    save_all_plans_to_file(simulation)
    
    # 绘制图表
    try:
        network.adcr_link.plot_clusters_and_paths(output_dir=output_dir)
        
        # 获取特定时间步的计划
        data_collector = simulation.get_data_collector()
        decision_records = data_collector.full_decision_history
        
        if decision_records:
            # 选择中间时间步的计划
            mid_index = len(decision_records) // 2
            record = decision_records[mid_index]
            t = record["time_step"]
            plans = record["plans"]
            
            # 保存特定时间步的计划
            out_path = f"plans_t{t}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for plan in plans:
                    receiver_id = plan.get("receiver_id")
                    donor_id = plan.get("donor_id")
                    path_ids = plan.get("path_ids", [])
                    path_str = "->".join(str(n) for n in path_ids) if path_ids else ""
                    line = f"d={donor_id}, r={receiver_id}, path={path_str}\n"
                    f.write(line)
            
            # 绘制能量路径
            plotter.plot_energy_paths_at_time(network, plans, t)
        
        # 绘制其他图表
        plotter.plot_node_distribution(network.nodes)
        
        # 从 DataCollector 获取状态记录用于绘制
        status_records = data_collector.status_records
        if status_records:
            # 转换为原来的格式用于兼容
            results = []
            for record in status_records:
                step_result = []
                for node_data in record["nodes"]:
                    step_result.append({
                        "node_id": node_data["node_id"],
                        "current_energy": node_data["current_energy"],
                        "received_energy": node_data["received_energy"],
                        "transferred_energy": node_data["transferred_energy"],
                        "energy_history": node_data["energy_history"]
                    })
                results.append(step_result)
            
            plotter.plot_energy_over_time(network.nodes, results)
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")
    
    return result


def run_parallel_simulation(config, num_runs: int = 10, max_workers: int = 4):
    """运行并行仿真"""
    print("=== 并行仿真模式 ===")
    
    # 创建仿真运行器
    runner = create_simulation_runner(config)
    
    # 运行并行仿真
    results = runner.run_parallel_simulations(
        num_runs=num_runs,
        max_workers=max_workers,
        use_same_seed=True,
        weight_config={
            "mode": "single",
            "vary_param": "w_l",
            "start": 0.5,
            "step": 0.2,
            "w_b_fixed": 0.7,
            "w_d_fixed": 0.5,
            "w_l_fixed": 1.0
        }
    )
    
    print("\n所有并行运行完成！")
    return results


def main():
    """主函数"""
    # 固定参数配置
    MODE = "single"  # 运行模式: "single" 或 "parallel"
    TIME_STEPS = 10080  # 仿真时间步数
    NUM_NODES = 25  # 网络节点数
    OUTPUT_DIR = "adcr"  # 输出目录
    
    # 并行仿真参数（仅在 MODE="parallel" 时使用）
    NUM_RUNS = 10  # 并行运行次数
    MAX_WORKERS = 4  # 最大工作进程数
    
    # 创建配置
    config = create_default_config()
    config.update_simulation_config(time_steps=TIME_STEPS)
    config.update_network_config(num_nodes=NUM_NODES)
    config.update_output_config(output_dir=OUTPUT_DIR)
    
    print("=" * 60)
    print("WSN能量传输仿真系统")
    print("=" * 60)
    print(f"运行模式: {MODE}")
    print(f"时间步数: {TIME_STEPS}")
    print(f"网络节点数: {NUM_NODES}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    if MODE == "parallel":
        print(f"并行运行次数: {NUM_RUNS}")
        print(f"最大工作进程: {MAX_WORKERS}")
    
    print("=" * 60)
    
    # 运行仿真
    if MODE == "single":
        result = run_single_simulation(config, OUTPUT_DIR)
        if result:
            print("单次仿真完成！")
    elif MODE == "parallel":
        results = run_parallel_simulation(config, NUM_RUNS, MAX_WORKERS)
        print("并行仿真完成！")


if __name__ == "__main__":
    main()

