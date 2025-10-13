#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行仿真启动文件
用于运行多次独立的仿真实验，支持固定种子配置用于对比实验
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import json
from datetime import datetime
import random
import numpy as np

# 导入原有模块
from src.network import Network
from src.schedulers import LyapunovScheduler
from src.energy_simulation import EnergySimulation
from src.plotter import (
    plot_node_distribution, 
    plot_energy_over_time, 
    plot_energy_distribution,
    plot_energy_histogram,
    plot_energy_transfer_history
)


def run_single_simulation(run_id, network_config, scheduler_config, output_base_dir="data", use_same_seed=True, reward_weights=None):
    """
    单次仿真函数 - 完全独立运行，支持动态权重参数
    
    Args:
        run_id: 运行ID (0, 1, 2, ...)
        network_config: 网络配置
        scheduler_config: 调度器配置
        output_base_dir: 输出基础目录
        use_same_seed: 是否使用相同种子（用于对比实验）
        reward_weights: 奖励函数权重参数
            {
                "w_b": 0.5,  # 均衡改进权重
                "w_d": 0.8,  # 有效送达量权重  
                "w_l": 1.5   # 损耗惩罚权重
            }
    
    Returns:
        dict: 运行结果信息
    """
    try:
        # 创建独立的输出目录
        output_dir = os.path.join(output_base_dir, f"run_{run_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 种子设置
        if use_same_seed:
            # 对比实验：所有运行使用相同种子
            base_seed = 42
            random.seed(base_seed)
            np.random.seed(base_seed)
            seed_info = f"固定种子 {base_seed}"
        else:
            # 独立实验：每次运行使用不同种子
            seed = 42 + run_id
            random.seed(seed)
            np.random.seed(seed)
            seed_info = f"种子 {seed}"
        
        print(f"运行 {run_id}: {seed_info}")
        
        # 创建网络和调度器
        network = Network(num_nodes=25, network_config=network_config)
        scheduler = LyapunovScheduler(V=0.5, K=1, max_hops=3)
        simulation = EnergySimulation(network, time_steps=10080, scheduler=scheduler)
        
        # 🔥 关键步骤：设置奖励函数权重
        if reward_weights:
            simulation.w_b = reward_weights.get("w_b", 0.8)  # 默认0.8
            simulation.w_d = reward_weights.get("w_d", 0.8)  # 默认0.8
            simulation.w_l = reward_weights.get("w_l", 1.5)  # 默认1.5
            
            print(f"  设置权重: w_b={simulation.w_b}, w_d={simulation.w_d}, w_l={simulation.w_l}")
        
        # 运行仿真
        start_time = time.time()
        simulation.simulate()
        end_time = time.time()
        
        # # 保存结果到独立目录（包含权重信息）
        # save_simulation_results(simulation, output_dir, run_id, reward_weights)

        # 计算统计信息（与main.py保持一致）
        final_energies = [node.current_energy for node in network.nodes]
        avg_energy = np.mean(final_energies)
        std_energy = np.std(final_energies)
        
        # 计算与main.py相同的统计数据
        # 1. 所有时间点方差的平均值
        avg_variance = np.mean(simulation.energy_Standards) if simulation.energy_Standards else 0
        
        # 2. 总发送能量 = 所有节点传输的能量总和
        total_sent_energy = sum(sum(node.transferred_history) for node in network.nodes)
        
        # 3. 总接收能量 = 所有节点接收的能量总和
        total_received_energy = sum(sum(node.received_history) for node in network.nodes)
        
        # 4. 总损失能量 = 总发送能量 - 总接收能量
        total_loss_energy = total_sent_energy - total_received_energy
        
        # 5. 能量传输效率
        energy_transfer_efficiency = (total_received_energy / total_sent_energy * 100 if total_sent_energy > 0 else 0)
        
        return {
            "run_id": run_id,
            "status": "success",
            "duration": end_time - start_time,
            "output_dir": output_dir,
            "final_energies": final_energies,
            "avg_energy": avg_energy,
            "std_energy": std_energy,
            "seed_info": seed_info,
            "reward_weights": reward_weights,
            "num_nodes": len(network.nodes),
            "time_steps": simulation.time_steps,
            # 与main.py相同的统计数据
            "avg_variance": avg_variance,
            "total_sent_energy": total_sent_energy,
            "total_received_energy": total_received_energy,
            "total_loss_energy": total_loss_energy,
            "energy_transfer_efficiency": energy_transfer_efficiency
        }
        
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "output_dir": output_dir if 'output_dir' in locals() else None,
            "seed_info": seed_info if 'seed_info' in locals() else "unknown",
            "reward_weights": reward_weights if 'reward_weights' in locals() else None
        }


# def save_simulation_results(simulation, output_dir, run_id, reward_weights=None):
#     """
#     保存仿真结果到指定目录
#
#     Args:
#         simulation: EnergySimulation对象
#         output_dir: 输出目录
#         run_id: 运行ID
#         reward_weights: 奖励函数权重参数
#     """
#     try:
#         # 保存节点分布图
#         plot_node_distribution(simulation.network.nodes, output_dir=output_dir)
#
#         # 保存能量变化图
#         plot_energy_over_time(simulation.network.nodes, simulation.results, output_dir=output_dir)
#
#         # 保存最终能量分布图
#         plot_energy_distribution(simulation.network.nodes, simulation.time_steps, output_dir=output_dir)
#
#         # 保存最终能量直方图
#         plot_energy_histogram(simulation.network.nodes, simulation.time_steps, output_dir=output_dir)
#
#         # 保存能量传输历史
#         plot_energy_transfer_history(simulation.network.nodes, output_dir=output_dir)
#
#         # 保存仿真数据（包含权重信息）
#         save_simulation_data(simulation, output_dir, run_id, reward_weights)
#
#         print(f"运行 {run_id}: 结果已保存到 {output_dir}")
#
#     except Exception as e:
#         print(f"运行 {run_id}: 保存结果时出错 - {e}")


def save_simulation_data(simulation, output_dir, run_id, reward_weights=None):
    """
    保存仿真数据到JSON文件
    
    Args:
        simulation: EnergySimulation对象
        output_dir: 输出目录
        run_id: 运行ID
        reward_weights: 奖励函数权重参数
    """
    try:
        # 收集节点数据
        nodes_data = []
        for node in simulation.network.nodes:
            nodes_data.append({
                "node_id": node.node_id,
                "position": node.position,
                "has_solar": node.has_solar,
                "is_mobile": getattr(node, "is_mobile", False),
                "final_energy": node.current_energy,
                "initial_energy": node.initial_energy,
                "received_energy": sum(getattr(node, "received_history", []) or [0]),
                "transferred_energy": sum(getattr(node, "transferred_history", []) or [0])
            })
        
        # 计算与main.py相同的统计数据
        # 1. 所有时间点方差的平均值
        avg_variance = np.mean(simulation.energy_Standards) if simulation.energy_Standards else 0
        
        # 2. 总发送能量 = 所有节点传输的能量总和
        total_sent_energy = sum(sum(node.transferred_history) for node in simulation.network.nodes)
        
        # 3. 总接收能量 = 所有节点接收的能量总和
        total_received_energy = sum(sum(node.received_history) for node in simulation.network.nodes)
        
        # 4. 总损失能量 = 总发送能量 - 总接收能量
        total_loss_energy = total_sent_energy - total_received_energy
        
        # 5. 能量传输效率
        energy_transfer_efficiency = (total_received_energy / total_sent_energy * 100 if total_sent_energy > 0 else 0)
        
        # 收集仿真统计信息
        simulation_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "time_steps": simulation.time_steps,
            "num_nodes": len(simulation.network.nodes),
            "reward_weights": reward_weights,  # 记录权重参数
            "nodes": nodes_data,
            "energy_stats": {
                "final_avg_energy": np.mean([n["final_energy"] for n in nodes_data]),
                "final_std_energy": np.std([n["final_energy"] for n in nodes_data])
            },
            # 与main.py相同的统计数据
            "main_py_stats": {
                "avg_variance": avg_variance,
                "total_sent_energy": total_sent_energy,
                "total_received_energy": total_received_energy,
                "total_loss_energy": total_loss_energy,
                "energy_transfer_efficiency": energy_transfer_efficiency
            },
            "k_history": getattr(simulation, "K_history", []),
            "k_stats": getattr(simulation, "K_stats", [])
        }
        
        # 保存到JSON文件
        json_path = os.path.join(output_dir, f"simulation_data_run_{run_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simulation_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"保存仿真数据时出错: {e}")


def run_parallel_simulations(num_runs=10, max_workers=4, use_same_seed=True, network_config=None, scheduler_config=None, weight_config=None):
    """
    并行运行多次仿真 - 支持动态权重参数
    
    Args:
        num_runs: 运行次数
        max_workers: 最大并行进程数
        use_same_seed: 是否使用相同种子（用于对比实验）
        network_config: 网络配置
        scheduler_config: 调度器配置
        weight_config: 权重配置
            {
                "w_b_start": 0.5,    # w_b起始值
                "w_b_step": 0.1,     # w_b步长
                "w_d_fixed": 0.8,    # w_d固定值
                "w_l_fixed": 1.5     # w_l固定值
            }
    
    Returns:
        list: 运行结果列表
    """
    print(f"开始并行运行 {num_runs} 次仿真...")
    print(f"使用 {max_workers} 个并行进程")
    print(f"种子模式: {'固定种子（对比实验）' if use_same_seed else '不同种子（独立实验）'}")
    
    # 权重配置
    if weight_config is None:
        weight_config = {
            "w_b_start": 0.5,
            "w_b_step": 0.1,
            "w_d_fixed": 0.8,
            "w_l_fixed": 1.5
        }
    
    print(f"权重配置: {weight_config}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 默认配置
    if network_config is None:
        network_config = {
            'low_threshold': 0.1,
            'high_threshold': 0.9,
            'node_initial_energy': 40000,
            'max_hops': 3,
            'random_seed': 42,
            'distribution_mode': 'uniform'
        }
    
    if scheduler_config is None:
        scheduler_config = {
            'V': 0.5,
            'K': 1,
            'max_hops': 3
        }
    
    start_time = time.time()
    results = []
    
    # 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for run_id in range(num_runs):
            # 🔥 关键步骤：计算当前运行的权重参数
            current_weights = {
                "w_b": weight_config["w_b_start"] + run_id * weight_config["w_b_step"],
                "w_d": weight_config["w_d_fixed"],
                "w_l": weight_config["w_l_fixed"]
            }
            
            print(f"准备运行 {run_id}: 权重 w_b={current_weights['w_b']:.1f}")
            
            future = executor.submit(
                run_single_simulation, 
                run_id, 
                network_config, 
                scheduler_config, 
                "data", 
                use_same_seed,
                current_weights  # 传递权重参数
            )
            futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":
                weights_info = result["reward_weights"]
                print(f"✅ 运行 {result['run_id']} 完成 (耗时: {result['duration']:.2f}s, "
                      f"权重: w_b={weights_info['w_b']:.1f}, w_d={weights_info['w_d']:.1f}, w_l={weights_info['w_l']:.1f})")
            else:
                print(f"❌ 运行 {result['run_id']} 失败: {result['error']}")
    
    end_time = time.time()
    
    # 统计结果
    successful_runs = [r for r in results if r["status"] == "success"]
    failed_runs = [r for r in results if r["status"] == "failed"]
    
    print("-" * 60)
    print(f"=== 运行完成 ===")
    print(f"总耗时: {end_time - start_time:.2f}s")
    print(f"成功: {len(successful_runs)}/{num_runs}")
    print(f"失败: {len(failed_runs)}/{num_runs}")
    
    if failed_runs:
        print(f"失败的运行: {[r['run_id'] for r in failed_runs]}")
    
    # 生成汇总报告
    if successful_runs:
        generate_summary_report(successful_runs, use_same_seed)
    
    return results


def generate_summary_report(successful_runs, use_same_seed):
    """
    生成汇总报告
    
    Args:
        successful_runs: 成功运行的列表
        use_same_seed: 是否使用相同种子
    """
    try:
        # 创建汇总目录
        summary_dir = "data/summary"
        os.makedirs(summary_dir, exist_ok=True)
        
        # 计算统计信息
        avg_energies = [r["avg_energy"] for r in successful_runs]
        durations = [r["duration"] for r in successful_runs]
        
        # 计算与main.py相同的统计数据
        avg_variances = [r["avg_variance"] for r in successful_runs]
        total_sent_energies = [r["total_sent_energy"] for r in successful_runs]
        total_received_energies = [r["total_received_energy"] for r in successful_runs]
        total_loss_energies = [r["total_loss_energy"] for r in successful_runs]
        energy_transfer_efficiencies = [r["energy_transfer_efficiency"] for r in successful_runs]
        
        # 按权重参数分组分析
        weight_analysis = analyze_weight_effects(successful_runs)
        
        summary_data = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "num_runs": len(successful_runs),
                "seed_mode": "fixed" if use_same_seed else "different",
                "total_duration": sum(durations)
            },
            "energy_statistics": {
                "avg_energy_mean": np.mean(avg_energies),
                "avg_energy_std": np.std(avg_energies),
                "avg_energy_min": np.min(avg_energies),
                "avg_energy_max": np.max(avg_energies)
            },
            "performance_statistics": {
                "duration_mean": np.mean(durations),
                "duration_std": np.std(durations),
                "duration_min": np.min(durations),
                "duration_max": np.max(durations)
            },
            # 与main.py相同的统计数据
            "main_py_statistics": {
                "avg_variance_mean": np.mean(avg_variances),
                "avg_variance_std": np.std(avg_variances),
                "total_sent_energy_mean": np.mean(total_sent_energies),
                "total_sent_energy_std": np.std(total_sent_energies),
                "total_received_energy_mean": np.mean(total_received_energies),
                "total_received_energy_std": np.std(total_received_energies),
                "total_loss_energy_mean": np.mean(total_loss_energies),
                "total_loss_energy_std": np.std(total_loss_energies),
                "energy_transfer_efficiency_mean": np.mean(energy_transfer_efficiencies),
                "energy_transfer_efficiency_std": np.std(energy_transfer_efficiencies)
            },
            "weight_analysis": weight_analysis,  # 权重效果分析
            "individual_runs": successful_runs
        }
        
        # 保存汇总数据
        summary_path = os.path.join(summary_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # 打印汇总信息
        print(f"\n=== 汇总统计 ===")
        print(f"平均能量: {summary_data['energy_statistics']['avg_energy_mean']:.2f} ± {summary_data['energy_statistics']['avg_energy_std']:.2f} J")
        print(f"运行时间: {summary_data['performance_statistics']['duration_mean']:.2f} ± {summary_data['performance_statistics']['duration_std']:.2f} s")
        
        # 打印与main.py相同的统计数据
        print(f"\n=== 与main.py相同的统计数据 ===")
        print(f"所有时间点方差的平均值: {summary_data['main_py_statistics']['avg_variance_mean']:.4f} ± {summary_data['main_py_statistics']['avg_variance_std']:.4f}")
        print(f"总发送能量: {summary_data['main_py_statistics']['total_sent_energy_mean']:.2f} ± {summary_data['main_py_statistics']['total_sent_energy_std']:.2f} J")
        print(f"总接收能量: {summary_data['main_py_statistics']['total_received_energy_mean']:.2f} ± {summary_data['main_py_statistics']['total_received_energy_std']:.2f} J")
        print(f"总损失能量: {summary_data['main_py_statistics']['total_loss_energy_mean']:.2f} ± {summary_data['main_py_statistics']['total_loss_energy_std']:.2f} J")
        print(f"能量传输效率: {summary_data['main_py_statistics']['energy_transfer_efficiency_mean']:.2f} ± {summary_data['main_py_statistics']['energy_transfer_efficiency_std']:.2f}%")
        
        # 打印权重效果分析
        if summary_data.get("weight_analysis"):
            print(f"\n=== 权重效果分析 ===")
            for weight_key, analysis in summary_data["weight_analysis"].items():
                w_b = analysis["w_b_value"]
                print(f"w_b = {w_b:.1f}:")
                print(f"  平均能量: {analysis['avg_energy']['mean']:.2f} ± {analysis['avg_energy']['std']:.2f} J")
                print(f"  运行次数: {analysis['num_runs']}")
                
                # 打印与main.py相同的统计数据
                main_py_stats = analysis.get("main_py_stats", {})
                if main_py_stats:
                    print(f"  [与main.py相同的统计数据]")
                    print(f"    所有时间点方差的平均值: {main_py_stats['avg_variance']['mean']:.4f} ± {main_py_stats['avg_variance']['std']:.4f}")
                    print(f"    总发送能量: {main_py_stats['total_sent_energy']['mean']:.2f} ± {main_py_stats['total_sent_energy']['std']:.2f} J")
                    print(f"    总接收能量: {main_py_stats['total_received_energy']['mean']:.2f} ± {main_py_stats['total_received_energy']['std']:.2f} J")
                    print(f"    总损失能量: {main_py_stats['total_loss_energy']['mean']:.2f} ± {main_py_stats['total_loss_energy']['std']:.2f} J")
                    print(f"    能量传输效率: {main_py_stats['energy_transfer_efficiency']['mean']:.2f} ± {main_py_stats['energy_transfer_efficiency']['std']:.2f}%")
            
            # 打印直观的权重参数对比表格
            print_weight_comparison_table(summary_data["weight_analysis"])
        
        print(f"\n汇总报告已保存到: {summary_path}")
        
    except Exception as e:
        print(f"生成汇总报告时出错: {e}")


def print_weight_comparison_table(weight_analysis):
    """
    打印直观的权重参数对比表格
    
    Args:
        weight_analysis: 权重效果分析结果
    """
    try:
        print(f"\n=== 权重参数对比表格 ===")
        
        # 按权重值排序
        sorted_weights = sorted(weight_analysis.items(), key=lambda x: x[1]["w_b_value"])
        
        # 1. 方差对比
        print(f"\n📊 所有时间点方差的平均值:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            main_py_stats = analysis.get("main_py_stats", {})
            if main_py_stats and "avg_variance" in main_py_stats:
                variance_mean = main_py_stats["avg_variance"]["mean"]
                variance_std = main_py_stats["avg_variance"]["std"]
                print(f"  {w_b:.1f}: {variance_mean:.4f} ± {variance_std:.4f}")
        
        # 2. 能量传输效率对比
        print(f"\n⚡ 能量传输效率:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            main_py_stats = analysis.get("main_py_stats", {})
            if main_py_stats and "energy_transfer_efficiency" in main_py_stats:
                efficiency_mean = main_py_stats["energy_transfer_efficiency"]["mean"]
                efficiency_std = main_py_stats["energy_transfer_efficiency"]["std"]
                print(f"  {w_b:.1f}: {efficiency_mean:.2f}% ± {efficiency_std:.2f}%")
        
        # 3. 总发送能量对比
        print(f"\n📤 总发送能量:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            main_py_stats = analysis.get("main_py_stats", {})
            if main_py_stats and "total_sent_energy" in main_py_stats:
                sent_mean = main_py_stats["total_sent_energy"]["mean"]
                sent_std = main_py_stats["total_sent_energy"]["std"]
                print(f"  {w_b:.1f}: {sent_mean:.2f}J ± {sent_std:.2f}J")
        
        # 4. 总接收能量对比
        print(f"\n📥 总接收能量:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            main_py_stats = analysis.get("main_py_stats", {})
            if main_py_stats and "total_received_energy" in main_py_stats:
                received_mean = main_py_stats["total_received_energy"]["mean"]
                received_std = main_py_stats["total_received_energy"]["std"]
                print(f"  {w_b:.1f}: {received_mean:.2f}J ± {received_std:.2f}J")
        
        # 5. 总损失能量对比
        print(f"\n💸 总损失能量:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            main_py_stats = analysis.get("main_py_stats", {})
            if main_py_stats and "total_loss_energy" in main_py_stats:
                loss_mean = main_py_stats["total_loss_energy"]["mean"]
                loss_std = main_py_stats["total_loss_energy"]["std"]
                print(f"  {w_b:.1f}: {loss_mean:.2f}J ± {loss_std:.2f}J")
        
        # 6. 平均能量对比
        print(f"\n🔋 最终平均能量:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            avg_energy_mean = analysis["avg_energy"]["mean"]
            avg_energy_std = analysis["avg_energy"]["std"]
            print(f"  {w_b:.1f}: {avg_energy_mean:.2f}J ± {avg_energy_std:.2f}J")
        
        # 7. 运行时间对比
        print(f"\n⏱️ 运行时间:")
        for weight_key, analysis in sorted_weights:
            w_b = analysis["w_b_value"]
            duration_mean = analysis["duration"]["mean"]
            duration_std = analysis["duration"]["std"]
            print(f"  {w_b:.1f}: {duration_mean:.2f}s ± {duration_std:.2f}s")
        
    except Exception as e:
        print(f"打印权重对比表格时出错: {e}")


def analyze_weight_effects(successful_runs):
    """
    分析不同权重参数的效果
    
    Args:
        successful_runs: 成功运行的列表
    
    Returns:
        dict: 权重效果分析结果
    """
    try:
        # 按权重参数分组
        weight_groups = {}
        for result in successful_runs:
            if result["status"] == "success" and result.get("reward_weights"):
                weights = result["reward_weights"]
                w_b = weights["w_b"]
                if w_b not in weight_groups:
                    weight_groups[w_b] = []
                weight_groups[w_b].append(result)
        
        # 分析每个权重参数的效果
        weight_analysis = {}
        for w_b in sorted(weight_groups.keys()):
            group_results = weight_groups[w_b]
            avg_energies = [r["avg_energy"] for r in group_results]
            durations = [r["duration"] for r in group_results]
            
            # 与main.py相同的统计数据
            avg_variances = [r["avg_variance"] for r in group_results]
            total_sent_energies = [r["total_sent_energy"] for r in group_results]
            total_received_energies = [r["total_received_energy"] for r in group_results]
            total_loss_energies = [r["total_loss_energy"] for r in group_results]
            energy_transfer_efficiencies = [r["energy_transfer_efficiency"] for r in group_results]
            
            weight_analysis[f"w_b_{w_b:.1f}"] = {
                "w_b_value": w_b,
                "num_runs": len(group_results),
                "avg_energy": {
                    "mean": float(np.mean(avg_energies)),
                    "std": float(np.std(avg_energies)),
                    "min": float(np.min(avg_energies)),
                    "max": float(np.max(avg_energies))
                },
                "duration": {
                    "mean": float(np.mean(durations)),
                    "std": float(np.std(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations))
                },
                # 与main.py相同的统计数据
                "main_py_stats": {
                    "avg_variance": {
                        "mean": float(np.mean(avg_variances)),
                        "std": float(np.std(avg_variances)),
                        "min": float(np.min(avg_variances)),
                        "max": float(np.max(avg_variances))
                    },
                    "total_sent_energy": {
                        "mean": float(np.mean(total_sent_energies)),
                        "std": float(np.std(total_sent_energies)),
                        "min": float(np.min(total_sent_energies)),
                        "max": float(np.max(total_sent_energies))
                    },
                    "total_received_energy": {
                        "mean": float(np.mean(total_received_energies)),
                        "std": float(np.std(total_received_energies)),
                        "min": float(np.min(total_received_energies)),
                        "max": float(np.max(total_received_energies))
                    },
                    "total_loss_energy": {
                        "mean": float(np.mean(total_loss_energies)),
                        "std": float(np.std(total_loss_energies)),
                        "min": float(np.min(total_loss_energies)),
                        "max": float(np.max(total_loss_energies))
                    },
                    "energy_transfer_efficiency": {
                        "mean": float(np.mean(energy_transfer_efficiencies)),
                        "std": float(np.std(energy_transfer_efficiencies)),
                        "min": float(np.min(energy_transfer_efficiencies)),
                        "max": float(np.max(energy_transfer_efficiencies))
                    }
                }
            }
        
        return weight_analysis
        
    except Exception as e:
        print(f"分析权重效果时出错: {e}")
        return {}


def main():
    """主函数"""
    # 配置参数
    NUM_RUNS = 20
    MAX_WORKERS = 6  # 根据CPU核心数调整
    USE_SAME_SEED = True  # 对比实验使用固定种子
    
    # 🔥 权重配置
    weight_config = {
        "w_b_start": 0.1,    # w_b从0.5开始
        "w_b_step": 0.1,     # 每次+0.1
        "w_d_fixed": 0.8,    # w_d固定为0.8
        "w_l_fixed": 1.5     # w_l固定为1.5
    }
    
    # 网络配置
    network_config = {
        'low_threshold': 0.1,
        'high_threshold': 0.9,
        'node_initial_energy': 40000,
        'max_hops': 3,
        'random_seed': 42,
        'distribution_mode': 'uniform'
    }
    
    # 调度器配置
    scheduler_config = {
        'V': 0.5,
        'K': 1,
        'max_hops': 3
    }
    
    print("=" * 60)
    print("并行仿真启动器（动态权重版本）")
    print("=" * 60)
    print(f"运行次数: {NUM_RUNS}")
    print(f"并行进程数: {MAX_WORKERS}")
    print(f"种子模式: {'固定种子（对比实验）' if USE_SAME_SEED else '不同种子（独立实验）'}")
    print(f"权重配置: {weight_config}")
    print(f"网络配置: {network_config}")
    print(f"调度器配置: {scheduler_config}")
    print("=" * 60)
    
    # 运行并行仿真
    results = run_parallel_simulations(
        num_runs=NUM_RUNS,
        max_workers=MAX_WORKERS,
        use_same_seed=USE_SAME_SEED,
        network_config=network_config,
        scheduler_config=scheduler_config,
        weight_config=weight_config  # 传递权重配置
    )
    
    print("\n所有运行完成！")


if __name__ == "__main__":
    # 设置多进程启动方法（Windows兼容）
    mp.set_start_method('spawn', force=True)
    main()
