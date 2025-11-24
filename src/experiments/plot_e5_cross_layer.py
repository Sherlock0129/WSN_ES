"""
实验5图表生成：四层机制协同 vs 基线系统
生成四张图：
1. 能量时间变化图（四层机制启用）
2. 能量时间变化图（四层机制关闭）
3. 雷达图（四层机制启用）
4. 雷达图（四层机制关闭）
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches

# Style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 24

def load_energy_over_time(csv_path: str) -> pd.DataFrame:
    """从virtual_center_node_info.csv加载能量时间序列数据"""
    if not os.path.exists(csv_path):
        print(f"Warning: Data file not found at {os.path.abspath(csv_path)}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        # 排除物理中心节点（ID=0）
        df = df[df['node_id'] != 0]
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return pd.DataFrame()

def plot_energy_over_time(
    csv_path: str,
    output_path: str,
    title: str = "Energy Evolution Over Time"
):
    """
    绘制节点能量随时间变化的图
    """
    df = load_energy_over_time(csv_path)
    if df.empty:
        print(f"Error: No data loaded from {csv_path}")
        return
    
    # 获取所有节点ID和时间步
    node_ids = sorted(df['node_id'].unique())
    time_steps = sorted(df['time_step'].unique()) if 'time_step' in df.columns else range(len(df))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    
    # 颜色方案
    colors = plt.cm.tab20(np.linspace(0, 1, len(node_ids)))
    
    # 绘制每个节点的能量曲线
    for idx, node_id in enumerate(node_ids):
        node_data = df[df['node_id'] == node_id]
        if 'time_step' in df.columns:
            node_data = node_data.sort_values('time_step')
            x = node_data['time_step'].values
        else:
            node_data = node_data.sort_index()
            x = range(len(node_data))
        y = node_data['energy'].values if 'energy' in node_data.columns else node_data['current_energy'].values
        
        ax.plot(x, y, label=f'Node {node_id}', color=colors[idx], linewidth=1.5, alpha=0.7)
    
    # 添加阈值线
    if 'energy' in df.columns or 'current_energy' in df.columns:
        energy_col = 'energy' if 'energy' in df.columns else 'current_energy'
        max_energy = df[energy_col].max()
        min_energy = df[energy_col].min()
        
        # 低阈值线（30%）
        low_threshold = min_energy + 0.3 * (max_energy - min_energy)
        ax.axhline(y=low_threshold, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Low Threshold (30%)')
        
        # 高阈值线（80%）
        high_threshold = min_energy + 0.8 * (max_energy - min_energy)
        ax.axhline(y=high_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High Threshold (80%)')
    
    ax.set_xlabel('Time Step', fontsize=22, fontweight='bold')
    ax.set_ylabel('Energy (J)', fontsize=22, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', ncol=2, fontsize=14, framealpha=0.9)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.close()

def load_statistics(json_path: str) -> dict:
    """从JSON文件加载统计数据"""
    if not os.path.exists(json_path):
        print(f"Warning: Statistics file not found at {os.path.abspath(json_path)}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}

def calculate_metrics_from_statistics(stats: dict) -> dict:
    """从统计数据计算所需指标"""
    metrics = {}
    
    # 从statistics中提取
    stats_data = stats.get('statistics', {})
    
    # 平均能量
    final_energies = stats_data.get('final_energies', [])
    if final_energies:
        metrics['mean_energy'] = np.mean(final_energies)
        metrics['min_energy'] = np.min(final_energies)
        metrics['std_energy'] = np.std(final_energies)
        metrics['cv'] = metrics['std_energy'] / metrics['mean_energy'] if metrics['mean_energy'] > 0 else 0
    else:
        metrics['mean_energy'] = 0
        metrics['min_energy'] = 0
        metrics['std_energy'] = 0
        metrics['cv'] = 0
    
    # 传输效率
    total_sent = stats_data.get('total_sent_energy', 0)
    total_received = stats_data.get('total_received_energy', 0)
    metrics['efficiency'] = (total_received / total_sent * 100) if total_sent > 0 else 0
    
    # 网络寿命（首个节点死亡时间）
    metrics['lifetime'] = stats_data.get('first_death_time', 0)
    
    return metrics

def extract_metrics_from_json(stats: dict) -> dict:
    """从JSON统计数据中提取关键指标"""
    metrics = {}
    
    stats_data = stats.get('statistics', {})
    network_data = stats.get('network_info', {})
    feedback_data = stats_data.get('feedback', {})
    
    # 传输效率
    metrics['efficiency'] = stats_data.get('efficiency', 0)
    
    # 平均方差（越低越好，需要反转）
    metrics['avg_variance'] = stats_data.get('avg_variance', 0)
    
    # 最低能量节点
    metrics['min_energy'] = network_data.get('min_energy', 0)
    
    # 存活节点数
    metrics['alive_nodes'] = network_data.get('alive_nodes', 0)
    metrics['total_nodes'] = network_data.get('total_nodes', 0)
    metrics['survival_rate'] = (metrics['alive_nodes'] / metrics['total_nodes'] * 100) if metrics['total_nodes'] > 0 else 0
    
    # 平均能量
    metrics['avg_energy'] = network_data.get('avg_energy', 0)
    
    # 总传输次数
    metrics['total_transfers'] = network_data.get('total_transfers', 0)
    
    return metrics

def plot_comparison_radar_chart(
    stats_path1: str,
    stats_path2: str,
    output_path: str,
    label1: str = "Configuration 1",
    label2: str = "Configuration 2",
    title: str = "Performance Comparison Radar Chart"
):
    """
    绘制两个配置的对比雷达图
    
    Args:
        stats_path1: 第一个配置的统计文件路径
        stats_path2: 第二个配置的统计文件路径
        output_path: 输出图片路径
        label1: 第一个配置的标签
        label2: 第二个配置的标签
        title: 图表标题
    """
    # 加载统计数据
    stats1 = load_statistics(stats_path1)
    stats2 = load_statistics(stats_path2)
    
    if not stats1 or not stats2:
        print(f"Error: Failed to load statistics from {stats_path1} or {stats_path2}")
        return
    
    # 提取指标
    metrics1 = extract_metrics_from_json(stats1)
    metrics2 = extract_metrics_from_json(stats2)
    
    # 定义雷达图的指标（选择关键指标）
    categories = [
        'Efficiency (%)',
        'Survival Rate (%)',
        'Avg Energy',
        'Min Energy',
        'Total Transfers',
        'Variance (reversed)'
    ]
    
    # 提取原始值
    values1 = [
        metrics1['efficiency'],
        metrics1['survival_rate'],
        metrics1['avg_energy'],
        metrics1['min_energy'],
        metrics1['total_transfers'],
        metrics1['avg_variance']
    ]
    
    values2 = [
        metrics2['efficiency'],
        metrics2['survival_rate'],
        metrics2['avg_energy'],
        metrics2['min_energy'],
        metrics2['total_transfers'],
        metrics2['avg_variance']
    ]
    
    # 计算归一化范围（使用两个配置的最大值）
    max_values = {}
    min_values = {}
    
    for i, key in enumerate(['efficiency', 'survival_rate', 'avg_energy', 'min_energy', 'total_transfers', 'avg_variance']):
        val1 = values1[i]
        val2 = values2[i]
        max_values[key] = max(val1, val2) * 1.1  # 留10%余量
        min_values[key] = min(val1, val2) * 0.9  # 留10%余量
        if max_values[key] == min_values[key]:
            max_values[key] = max(val1, val2) + 1
            min_values[key] = min(val1, val2) - 1
    
    # 归一化到0-1范围
    normalized_values1 = []
    normalized_values2 = []
    
    for i, key in enumerate(['efficiency', 'survival_rate', 'avg_energy', 'min_energy', 'total_transfers', 'avg_variance']):
        val1 = values1[i]
        val2 = values2[i]
        max_val = max_values[key]
        min_val = min_values[key]
        range_val = max_val - min_val
        
        if i < 5:  # 前5个是"越高越好"
            norm1 = (val1 - min_val) / range_val if range_val > 0 else 0.5
            norm2 = (val2 - min_val) / range_val if range_val > 0 else 0.5
        else:  # 方差是"越低越好"，需要反转
            norm1 = 1.0 - (val1 - min_val) / range_val if range_val > 0 else 0.5
            norm2 = 1.0 - (val2 - min_val) / range_val if range_val > 0 else 0.5
        
        # 限制在0-1范围内
        norm1 = max(0, min(1, norm1))
        norm2 = max(0, min(1, norm2))
        
        normalized_values1.append(norm1)
        normalized_values2.append(norm2)
    
    # 闭合图形
    normalized_values1 += [normalized_values1[0]]
    normalized_values2 += [normalized_values2[0]]
    categories_angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    categories_angles += [categories_angles[0]]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # 绘制两个配置
    ax.plot(categories_angles, normalized_values1, 'o-', linewidth=2.5, 
            color='#2E86AB', label=label1, markersize=8)
    ax.fill(categories_angles, normalized_values1, alpha=0.15, color='#2E86AB')
    
    ax.plot(categories_angles, normalized_values2, 's-', linewidth=2.5, 
            color='#A23B72', label=label2, markersize=8)
    ax.fill(categories_angles, normalized_values2, alpha=0.15, color='#A23B72')
    
    # 设置标签
    ax.set_xticks(categories_angles[:-1])
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=16, framealpha=0.9)
    
    # 添加标题
    plt.title(title, fontsize=24, fontweight='bold', pad=30)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison radar chart saved to: {output_path}")
    plt.close()

def plot_radar_chart(
    stats_path: str,
    output_path: str,
    title: str = "Performance Radar Chart"
):
    """
    绘制多维度性能雷达图
    指标：平均能量、最低节点能量、传输效率、方差、CV、网络寿命
    """
    stats = load_statistics(stats_path)
    if not stats:
        print(f"Error: No statistics loaded from {stats_path}")
        return
    
    metrics = calculate_metrics_from_statistics(stats)
    
    # 定义雷达图的指标
    categories = ['Mean Energy', 'Min Energy', 'Efficiency', 'Lifetime', 'CV (reversed)', 'Std (reversed)']
    
    # 原始值
    values = [
        metrics['mean_energy'],
        metrics['min_energy'],
        metrics['efficiency'],
        metrics['lifetime'],
        metrics['cv'],
        metrics['std_energy']
    ]
    
    # 归一化到0-1范围（对于"越高越好"的指标直接归一化，对于"越低越好"的指标反转）
    # 这里我们需要一个参考范围，可以使用合理的最大值
    max_values = {
        'mean_energy': 50000,
        'min_energy': 40000,
        'efficiency': 100,
        'lifetime': 10080,
        'cv': 1.0,
        'std_energy': 10000
    }
    
    normalized_values = []
    for i, (key, val) in enumerate(zip(['mean_energy', 'min_energy', 'efficiency', 'lifetime', 'cv', 'std_energy'], values)):
        if i < 4:  # 前4个是"越高越好"
            normalized = min(1.0, val / max_values[key]) if max_values[key] > 0 else 0
        else:  # 后2个是"越低越好"，需要反转
            normalized = 1.0 - min(1.0, val / max_values[key]) if max_values[key] > 0 else 1.0
        normalized_values.append(normalized)
    
    # 闭合图形
    normalized_values += [normalized_values[0]]
    categories_angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    categories_angles += [categories_angles[0]]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制
    ax.plot(categories_angles, normalized_values, 'o-', linewidth=2.5, color='#2E86AB', label='Performance')
    ax.fill(categories_angles, normalized_values, alpha=0.25, color='#2E86AB')
    
    # 设置标签
    ax.set_xticks(categories_angles[:-1])
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加标题
    plt.title(title, fontsize=24, fontweight='bold', pad=30)
    
    # 添加数值标注
    for angle, value, label in zip(categories_angles[:-1], normalized_values[:-1], categories):
        ax.text(angle, value + 0.1, f'{value:.2f}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.close()

def plot_e5_all_figures(
    proposed_dir: str,
    baseline_dir: str,
    output_base_dir: str = "paper_p/sections/figures/experiments"
):
    """
    生成实验5的所有图表
    
    Args:
        proposed_dir: 四层机制全部启用的实验数据目录
        baseline_dir: 四层机制全部关闭的实验数据目录
        output_base_dir: 输出目录
    """
    # 1. 能量时间变化图（四层机制启用）
    proposed_csv = os.path.join(proposed_dir, "virtual_center_node_info.csv")
    plot_energy_over_time(
        proposed_csv,
        os.path.join(output_base_dir, "e5_energy_over_time_proposed.png"),
        "E5: Energy Evolution (Four-Layer Mechanisms Enabled)"
    )
    
    # 2. 能量时间变化图（四层机制关闭）
    baseline_csv = os.path.join(baseline_dir, "virtual_center_node_info.csv")
    plot_energy_over_time(
        baseline_csv,
        os.path.join(output_base_dir, "e5_energy_over_time_baseline.png"),
        "E5: Energy Evolution (Four-Layer Mechanisms Disabled)"
    )
    
    # 3. 雷达图（四层机制启用）
    proposed_stats = os.path.join(proposed_dir, "simulation_statistics.json")
    plot_radar_chart(
        proposed_stats,
        os.path.join(output_base_dir, "e5_radar_proposed.png"),
        "E5: Performance Radar (Four-Layer Mechanisms Enabled)"
    )
    
    # 4. 雷达图（四层机制关闭）
    baseline_stats = os.path.join(baseline_dir, "simulation_statistics.json")
    plot_radar_chart(
        baseline_stats,
        os.path.join(output_base_dir, "e5_radar_baseline.png"),
        "E5: Performance Radar (Four-Layer Mechanisms Disabled)"
    )
    
    # 5. 对比雷达图
    plot_comparison_radar_chart(
        proposed_stats,
        baseline_stats,
        os.path.join(output_base_dir, "e5_radar_comparison.png"),
        label1="Four-Layer Mechanisms Enabled",
        label2="Four-Layer Mechanisms Disabled",
        title="E5: Performance Comparison Radar Chart"
    )
    
    print("\n所有实验5图表已生成完成！")

if __name__ == "__main__":
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # 实验数据目录（需要根据实际路径修改）
    data_base_dir = os.path.join(script_dir, 'data')
    
    # 实验5数据目录（需要根据实际实验名称修改）
    proposed_dir = os.path.join(data_base_dir, "exp5_four_layer_enabled")  # 修改为实际目录名
    baseline_dir = os.path.join(data_base_dir, "exp5_four_layer_disabled")  # 修改为实际目录名
    
    # 输出目录
    output_dir = os.path.join(project_root, "paper_p", "sections", "figures", "experiments")
    
    # 生成所有图表
    plot_e5_all_figures(proposed_dir, baseline_dir, output_dir)

