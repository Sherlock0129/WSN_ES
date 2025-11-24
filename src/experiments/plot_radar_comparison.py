"""
根据两个simulation_statistics.json文件生成对比雷达图
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 24

def load_statistics(json_path: str) -> dict:
    """从JSON文件加载统计数据"""
    if not os.path.exists(json_path):
        print(f"Error: Statistics file not found at {os.path.abspath(json_path)}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}

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
    
    # 打印原始值用于调试
    print("\n配置1指标:")
    for key, value in metrics1.items():
        print(f"  {key}: {value}")
    print("\n配置2指标:")
    for key, value in metrics2.items():
        print(f"  {key}: {value}")
    
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
    
    # 计算归一化范围（使用两个配置的最大值和最小值）
    max_values = {}
    min_values = {}
    
    for i, key in enumerate(['efficiency', 'survival_rate', 'avg_energy', 'min_energy', 'total_transfers', 'avg_variance']):
        val1 = values1[i]
        val2 = values2[i]
        max_val = max(val1, val2)
        min_val = min(val1, val2)
        
        # 对于"越高越好"的指标，使用0作为最小值（如果min_val > 0）
        if i < 5:  # 前5个是"越高越好"
            if min_val < 0:
                min_values[key] = min_val * 1.1
            else:
                min_values[key] = 0
            max_values[key] = max_val * 1.1 if max_val > 0 else 1
        else:  # 方差是"越低越好"
            min_values[key] = 0
            max_values[key] = max_val * 1.1 if max_val > 0 else 1
    
    # 归一化到0-1范围
    normalized_values1 = []
    normalized_values2 = []
    
    print("\n归一化计算详情:")
    print("=" * 80)
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
        
        # 打印归一化详情
        print(f"\n指标: {categories[i]}")
        print(f"  原始值 - 配置1: {val1:.2f}, 配置2: {val2:.2f}")
        print(f"  归一化范围: [{min_val:.2f}, {max_val:.2f}], 范围大小: {range_val:.2f}")
        print(f"  归一化后 - 配置1: {norm1:.3f}, 配置2: {norm2:.3f}")
        if i < 5:
            print(f"  计算公式: norm = (val - {min_val:.2f}) / {range_val:.2f}")
        else:
            print(f"  计算公式: norm = 1.0 - (val - {min_val:.2f}) / {range_val:.2f} (反转)")
    
    print("=" * 80)
    
    # 闭合图形
    normalized_values1 += [normalized_values1[0]]
    normalized_values2 += [normalized_values2[0]]
    categories_angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    categories_angles += [categories_angles[0]]
    
    # 创建雷达图
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'), 
                          facecolor='white', edgecolor='none')
    
    # 设置背景色
    ax.set_facecolor('#FAFAFA')
    
    # 绘制网格线（同心圆）
    ax.set_ylim(0, 1.1)  # 留出少量空间给标签
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                       fontsize=16, color='#666666', fontweight='normal')
    ax.grid(True, linestyle='-', linewidth=1.2, alpha=0.3, color='#CCCCCC')
    
    # 绘制径向线
    for angle in categories_angles[:-1]:
        ax.plot([angle, angle], [0, 1.1], '--', linewidth=1, alpha=0.2, color='#999999')
    
    # 定义更现代的颜色方案
    color1 = '#1E88E5'  # 蓝色 - 更现代
    color2 = '#D81B60'  # 粉红色 - 更现代
    
    # 绘制两个配置（先绘制填充，再绘制线条，确保层次感）
    ax.fill(categories_angles, normalized_values1, alpha=0.25, color=color1, 
            edgecolor='none', zorder=1)
    ax.fill(categories_angles, normalized_values2, alpha=0.25, color=color2, 
            edgecolor='none', zorder=1)
    
    # 绘制线条和标记点
    line1 = ax.plot(categories_angles, normalized_values1, 'o-', 
                    linewidth=3.5, color=color1, label=label1, 
                    markersize=10, markerfacecolor=color1, 
                    markeredgecolor='white', markeredgewidth=2.5,
                    zorder=3, alpha=0.95)
    
    line2 = ax.plot(categories_angles, normalized_values2, 's-', 
                    linewidth=3.5, color=color2, label=label2, 
                    markersize=10, markerfacecolor=color2, 
                    markeredgecolor='white', markeredgewidth=2.5,
                    zorder=3, alpha=0.95)
    
    # 设置标签 - 改进位置和样式
    ax.set_xticks(categories_angles[:-1])
    labels = ax.set_xticklabels(categories, fontsize=20, fontweight='bold', 
                                color='#333333')
    
    # 添加图例 - 改进样式
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), 
                      fontsize=18, framealpha=0.95, 
                      edgecolor='#CCCCCC', facecolor='white',
                      frameon=True, fancybox=True, shadow=True,
                      borderpad=1.2, labelspacing=1.0)
    legend.get_frame().set_linewidth(1.5)
    
    # 添加标题 - 改进样式
    plt.title(title, fontsize=26, fontweight='bold', pad=40, 
             color='#1A1A1A', y=1.08)
    
    # 添加中心点标记
    ax.plot(0, 0, 'o', markersize=8, color='#666666', zorder=5)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n对比雷达图已保存到: {output_path}")
    plt.close()

if __name__ == "__main__":
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # 默认路径
    stats_path1 = os.path.join(script_dir, 'data', 'exp5_four_layer_enabled', 'simulation_statistics.json')
    stats_path2 = os.path.join(script_dir, 'data', 'exp5_four_layer_disabled', 'simulation_statistics.json')
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) >= 3:
        stats_path1 = sys.argv[1]
        stats_path2 = sys.argv[2]
    
    # 输出路径
    output_path = os.path.join(project_root, "paper_p", "sections", "figures", "experiments", "e5_radar_comparison.png")
    
    # 生成对比雷达图
    plot_comparison_radar_chart(
        stats_path1,
        stats_path2,
        output_path,
        label1="Four-Layer Mechanisms Enabled",
        label2="Four-Layer Mechanisms Disabled",
        title="E5: Performance Comparison Radar Chart"
    )

