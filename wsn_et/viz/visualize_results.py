#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试结果可视化脚本
读取 method_comparison_results.csv 文件，生成各种性能指标的比较图
（从 T1/visualize_results.py 复制，逻辑未改动）
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 设置通用字体
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
rcParams['axes.unicode_minus'] = False

class ResultsVisualizer:
    def __init__(self, csv_file_path, output_dir="testRe"):
        """
        初始化可视化器
        
        :param csv_file_path: CSV结果文件路径
        :param output_dir: 输出图片的目录
        """
        self.csv_file_path = csv_file_path
        self.output_dir = output_dir
        self.df = None
        self.methods = None
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            # 只保留单次运行的结果（run=1）或平均结果（run='average'）
            self.df = self.df[self.df['run'].isin([1, 'average'])]
            self.methods = self.df['method'].unique()
            print(f"成功加载数据，包含 {len(self.methods)} 种方法的结果")
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def plot_energy_metrics_comparison(self):
        """Plot energy-related metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy Metrics Comparison', fontsize=16, fontweight='bold')
        
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#10AC84']
        colors = base_colors[:len(self.methods)]
        
        # 1. Mean energy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(self.methods, self.df['mean_energy'], color=colors)
        ax1.set_title('Mean Energy Comparison', fontweight='bold')
        ax1.set_ylabel('Mean Energy (J)')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # 2. Minimum energy comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(self.methods, self.df['min_energy'], color=colors)
        ax2.set_title('Minimum Energy Comparison', fontweight='bold')
        ax2.set_ylabel('Minimum Energy (J)')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # 3. Maximum energy comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(self.methods, self.df['max_energy'], color=colors)
        ax3.set_title('Maximum Energy Comparison', fontweight='bold')
        ax3.set_ylabel('Maximum Energy (J)')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        # 4. Total energy comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(self.methods, self.df['total_energy'], color=colors)
        ax4.set_title('Total Energy Comparison', fontweight='bold')
        ax4.set_ylabel('Total Energy (J)')
        ax4.tick_params(axis='x', rotation=45)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'energy_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 energy_metrics_comparison.png")
    
    def plot_variance_and_uniformity(self):
        """Plot variance and uniformity metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy Distribution Uniformity Metrics Comparison', fontsize=16, fontweight='bold')
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#10AC84']
        colors = base_colors[:len(self.methods)]
        # 1. Variance comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(self.methods, self.df['variance'], color=colors)
        ax1.set_title('Variance Comparison (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Variance')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        # 2. Standard deviation comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(self.methods, self.df['std_deviation'], color=colors)
        ax2.set_title('Standard Deviation Comparison (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        # 3. Coefficient of variation comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(self.methods, self.df['cv'], color=colors)
        ax3.set_title('Coefficient of Variation Comparison (Lower is Better)', fontweight='bold')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        # 4. Energy range comparison
        ax4 = axes[1, 1]
        bars4 = ax4.bar(self.methods, self.df['energy_range'], color=colors)
        ax4.set_title('Energy Range Comparison (Lower is Better)', fontweight='bold')
        ax4.set_ylabel('Energy Range (J)')
        ax4.tick_params(axis='x', rotation=45)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'variance_uniformity_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 variance_uniformity_comparison.png")
    
    def plot_node_survival_metrics(self):
        """Plot node survival related metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Node Survival Metrics Comparison', fontsize=16, fontweight='bold')
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#10AC84']
        colors = base_colors[:len(self.methods)]
        # 1. Dead nodes comparison
        ax1 = axes[0]
        bars1 = ax1.bar(self.methods, self.df['dead_nodes'], color=colors)
        ax1.set_title('Dead Nodes Comparison (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Number of Dead Nodes')
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        # 2. Low energy nodes comparison
        ax2 = axes[1]
        bars2 = ax2.bar(self.methods, self.df['low_energy_nodes'], color=colors)
        ax2.set_title('Low Energy Nodes Comparison (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Number of Low Energy Nodes')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'node_survival_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 node_survival_comparison.png")
    
    def plot_comprehensive_radar_chart(self):
        """Plot comprehensive performance radar chart"""
        metrics = ['mean_energy', 'min_energy', 'variance', 'cv', 'dead_nodes', 'low_energy_nodes']
        metric_labels = ['Mean Energy', 'Min Energy', 'Variance', 'CV', 'Dead Nodes', 'Low Energy Nodes']
        normalized_data = self.df[metrics].copy()
        reverse_metrics = ['variance', 'cv', 'dead_nodes', 'low_energy_nodes']
        for metric in metrics:
            if metric in reverse_metrics:
                min_val = normalized_data[metric].min(); max_val = normalized_data[metric].max()
                normalized_data[metric] = 1 - ((normalized_data[metric] - min_val) / (max_val - min_val) if max_val > min_val else 1)
            else:
                min_val = normalized_data[metric].min(); max_val = normalized_data[metric].max()
                normalized_data[metric] = ((normalized_data[metric] - min_val) / (max_val - min_val) if max_val > min_val else 1)
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#10AC84']
        colors = base_colors[:len(self.methods)]
        for i, method in enumerate(self.methods):
            values = normalized_data.iloc[i][metrics].tolist(); values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metric_labels); ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Performance Radar Chart\n(Values closer to outer ring indicate better performance)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0)); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 comprehensive_radar_chart.png")
    
    def plot_performance_ranking(self):
        """Plot performance ranking chart"""
        ranking_metrics = {
            'Variance Ranking': 'variance',
            'Min Energy Ranking': 'min_energy', 
            'Mean Energy Ranking': 'mean_energy',
            'CV Ranking': 'cv',
            'Dead Nodes Ranking': 'dead_nodes'
        }
        ranking_data = pd.DataFrame(index=self.methods)
        for rank_name, metric in ranking_metrics.items():
            if metric in ['variance', 'cv', 'dead_nodes', 'low_energy_nodes']:
                ranking_data[rank_name] = self.df.set_index('method')[metric].rank(method='min')
            else:
                ranking_data[rank_name] = self.df.set_index('method')[metric].rank(method='min', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.heatmap(ranking_data.T, annot=True, cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Ranking (1=Best)'}, fmt='.0f')
        plt.title('Performance Ranking Heatmap\n(Greener color indicates better ranking)', fontsize=14, fontweight='bold')
        plt.xlabel('Scheduling Methods'); plt.ylabel('Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_ranking_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 performance_ranking_heatmap.png")
    
    def plot_method_comparison_summary(self):
        """Plot method comparison summary chart"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sensor Network Energy Scheduling Methods Performance Comparison Summary', fontsize=16, fontweight='bold')
        metrics_info = [
            ('mean_energy', 'Mean Energy (J)', 'Higher is Better'),
            ('min_energy', 'Minimum Energy (J)', 'Higher is Better'),
            ('variance', 'Variance', 'Lower is Better'),
            ('cv', 'Coefficient of Variation', 'Lower is Better'),
            ('dead_nodes', 'Dead Nodes Count', 'Lower is Better'),
            ('total_energy', 'Total Energy (J)', 'Higher is Better')
        ]
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF9F43', '#10AC84']
        colors = base_colors[:len(self.methods)]
        for idx, (metric, title, note) in enumerate(metrics_info):
            row = idx // 3; col = idx % 3; ax = axes[row, col]
            bars = ax.bar(self.methods, self.df[metric], color=colors)
            ax.set_title(f'{title}\n({note})', fontweight='bold'); ax.tick_params(axis='x', rotation=45)
            for bar in bars:
                height = bar.get_height()
                if metric in ['variance', 'total_energy']:
                    label = f'{height:.0f}'
                elif metric == 'cv':
                    label = f'{height:.4f}'
                else:
                    label = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=8)
            if metric in ['variance', 'cv', 'dead_nodes', 'low_energy_nodes']:
                best_idx = self.df[metric].idxmin()
            else:
                best_idx = self.df[metric].idxmax()
            if 0 <= best_idx < len(bars):
                bars[best_idx].set_color('#FFD700')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'method_comparison_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("已生成 method_comparison_summary.png")
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        print("Starting to generate visualization charts...")
        try:
            self.plot_energy_metrics_comparison()
            self.plot_variance_and_uniformity()
            self.plot_node_survival_metrics()
            self.plot_comprehensive_radar_chart()
            self.plot_performance_ranking()
            self.plot_method_comparison_summary()
            print(f"\nAll visualization charts have been generated successfully!")
            print(f"Charts saved to: {os.path.abspath(self.output_dir)}")
            print("\nGenerated charts include:")
            print("1. energy_metrics_comparison.png - Energy Metrics Comparison")
            print("2. variance_uniformity_comparison.png - Variance and Uniformity Comparison")
            print("3. node_survival_comparison.png - Node Survival Metrics Comparison")
            print("4. comprehensive_radar_chart.png - Comprehensive Performance Radar Chart")
            print("5. performance_ranking_heatmap.png - Performance Ranking Heatmap")
            print("6. method_comparison_summary.png - Method Comparison Summary Chart")
        except Exception as e:
            print(f"Error generating visualization charts: {e}")
            raise

