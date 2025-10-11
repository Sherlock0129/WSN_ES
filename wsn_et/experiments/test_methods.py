#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同能量调度方法的性能比较脚本
比较指标：各节点能量的方差、最低能量、平均能量
（从 T1/test_methods.py 复制，仅调整导入到 wsn_et 包）
"""

import os
import sys
import numpy as np
import pandas as pd

# 调整为新包路径
from wsn_et.core.network import Network
from wsn_et.sim.energy_simulation import EnergySimulation
from wsn_et.scheduling import schedulers


class MethodTester:
    def __init__(self, network_config, time_steps=10080):
        """
        初始化测试器
        
        :param network_config: 网络配置参数
        :param time_steps: 仿真时间步数
        """
        self.network_config = network_config
        self.time_steps = time_steps
        self.results = []
        
    def create_fresh_network(self):
        """创建一个新的网络实例（确保每次测试都是相同的初始条件）"""
        return Network(num_nodes=self.network_config["num_nodes"], 
                      network_config=self.network_config)
    
    def calculate_metrics(self, network):
        """
        计算网络的性能指标
        
        :param network: 网络实例
        :return: 包含各项指标的字典
        """
        energies = [node.current_energy for node in network.nodes]
        
        metrics = {
            'variance': np.var(energies),  # 方差
            'std_deviation': np.std(energies),  # 标准差
            'min_energy': np.min(energies),  # 最低能量
            'max_energy': np.max(energies),  # 最高能量
            'mean_energy': np.mean(energies),  # 平均能量
            'median_energy': np.median(energies),  # 中位数能量
            'energy_range': np.max(energies) - np.min(energies),  # 能量范围
            'cv': np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 0,  # 变异系数
            'total_energy': np.sum(energies),  # 总能量
            'dead_nodes': sum(1 for e in energies if e <= 0),  # 死亡节点数
            'low_energy_nodes': sum(1 for e in energies if e < 0.1 * self.network_config["node_initial_energy"])  # 低能量节点数
        }
        
        return metrics
    
    def test_method(self, method_name, scheduler=None, runs=1):
        """
        测试单个方法
        
        :param method_name: 方法名称
        :param scheduler: 调度器实例，None表示使用提出的方法
        :param runs: 运行次数（用于取平均值）
        :return: 测试结果
        """
        print(f"\n=== 测试方法: {method_name} ===")
        
        all_metrics = []
        
        for run in range(runs):
            print(f"运行 {run + 1}/{runs}...")
            
            # 创建新的网络实例
            network = self.create_fresh_network()
            
            # 创建仿真实例
            simulation = EnergySimulation(network, self.time_steps, scheduler=scheduler)
            
            # 运行仿真
            simulation.simulate()
            
            # 计算指标
            metrics = self.calculate_metrics(network)
            metrics['method'] = method_name
            metrics['run'] = run + 1
            
            all_metrics.append(metrics)
            
            print(f"  运行 {run + 1} 完成 - 平均能量: {metrics['mean_energy']:.2f}, "
                  f"方差: {metrics['variance']:.2f}, 最低能量: {metrics['min_energy']:.2f}")
        
        # 计算多次运行的平均值
        if runs > 1:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if key not in ['method', 'run']:
                    avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            avg_metrics['method'] = method_name
            avg_metrics['run'] = 'average'
            all_metrics.append(avg_metrics)
        
        return all_metrics
    
    def run_all_tests(self, runs=1):
        """
        运行所有方法的测试
        
        :param runs: 每个方法的运行次数
        """
        print("开始测试所有能量调度方法...")
        print(f"网络配置: {self.network_config['num_nodes']} 个节点, {self.time_steps} 时间步")
        
        # 测试提出的方法 (scheduler=None)
        proposed_results = self.test_method("Proposed Method", scheduler=None, runs=runs)
        self.results.extend(proposed_results)
        
        # 测试 Lyapunov 方法
        lyapunov_sched = schedulers.LyapunovScheduler(V=0.5, K=3)
        lyapunov_results = self.test_method("Lyapunov", scheduler=lyapunov_sched, runs=runs)
        self.results.extend(lyapunov_results)
        
        # Cluster method removed as requested
        
        # 测试 Prediction 方法
        prediction_sched = schedulers.PredictionScheduler(alpha=0.6, horizon_min=60, K=3)
        prediction_results = self.test_method("Prediction", scheduler=prediction_sched, runs=runs)
        self.results.extend(prediction_results)
        
        # 测试 PowerControl 方法
        power_control_sched = schedulers.PowerControlScheduler(target_eta=0.25, K=3)
        power_control_results = self.test_method("PowerControl", scheduler=power_control_sched, runs=runs)
        self.results.extend(power_control_results)
        
        # 测试 Baseline 方法
        baseline_sched = schedulers.BaselineHeuristic(K=3)
        baseline_results = self.test_method("Baseline", scheduler=baseline_sched, runs=runs)
        self.results.extend(baseline_results)
        
        print("\n所有测试完成！")
    
    def save_results(self, filename="test_results.csv"):
        """
        保存测试结果到CSV文件
        
        :param filename: 输出文件名
        """
        if not self.results:
            print("没有测试结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        df.to_csv(filepath, index=False)
        print(f"测试结果已保存到: {filepath}")
    
    def display_summary(self):
        """
        显示测试结果摘要
        """
        if not self.results:
            print("没有测试结果可显示")
            return
        
        df = pd.DataFrame(self.results)
        
        # 只显示平均值结果（如果有多次运行）或单次运行结果
        summary_df = df[df['run'].isin(['average', 1])].copy()
        
        print("\n=== 测试结果摘要 ===")
        print("\n主要性能指标:")
        
        # 选择关键指标进行显示
        key_metrics = ['method', 'mean_energy', 'variance', 'std_deviation', 
                      'min_energy', 'max_energy', 'cv', 'dead_nodes', 'low_energy_nodes']
        
        display_df = summary_df[key_metrics].copy()
        
        # 格式化数值显示
        for col in ['mean_energy', 'variance', 'std_deviation', 'min_energy', 'max_energy']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        if 'cv' in display_df.columns:
            display_df['cv'] = display_df['cv'].round(4)
        
        print(display_df.to_string(index=False))
        
        # 找出最佳方法
        print("\n=== 性能排名 ===")
        
        # 按不同指标排名
        print("\n1. 最低方差(能量分布最均匀):")
        best_variance = summary_df.loc[summary_df['variance'].idxmin()]
        print(f"   {best_variance['method']}: 方差 = {best_variance['variance']:.2f}")
        
        print("\n2. 最高最低能量(避免节点死亡):")
        best_min_energy = summary_df.loc[summary_df['min_energy'].idxmax()]
        print(f"   {best_min_energy['method']}: 最低能量 = {best_min_energy['min_energy']:.2f}")
        
        print("\n3. 最高平均能量(整体能量保持):")
        best_mean_energy = summary_df.loc[summary_df['mean_energy'].idxmax()]
        print(f"   {best_mean_energy['method']}: 平均能量 = {best_mean_energy['mean_energy']:.2f}")
        
        print("\n4. 最低变异系数(相对均匀性):")
        best_cv = summary_df.loc[summary_df['cv'].idxmin()]
        print(f"   {best_cv['method']}: 变异系数 = {best_cv['cv']:.4f}")
        
        print("\n5. 最少死亡节点")
        best_dead = summary_df.loc[summary_df['dead_nodes'].idxmin()]
        print(f"   {best_dead['method']}: 死亡节点数 = {int(best_dead['dead_nodes'])}")


def main():
    """
    主函数
    """
    # 网络配置（与 main.py 保持一致）
    network_config = {
        "num_nodes": 25,
        "low_threshold": 0.1,
        "high_threshold": 0.9,
        "node_initial_energy": 40000,
        "max_hops": 3
    }
    
    # 创建测试器
    tester = MethodTester(network_config, time_steps=10080)
    
    # 运行所有测试（每个方法运行3次取平均值）
    tester.run_all_tests(runs=3)
    
    # 显示结果摘要
    tester.display_summary()
    
    # 保存结果
    tester.save_results("method_comparison_results.csv")
    
    print("\n测试完成！详细结果已保存到 data/method_comparison_results.csv")


if __name__ == "__main__":
    main()

