"""
绘制E1实验：触发次数与累计发送能量对比图
根据两次实验数据（智能被动传能 vs 固定60分钟传能）绘制对比图
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体和字体大小（根据之前的要求，文字和数字需放大2倍）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20  # 基础字体大小（2倍）
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18  # 图例字体大小（2倍）
rcParams['figure.titlesize'] = 22

def load_statistics(json_path):
    """从JSON文件加载统计数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_e1_triggers_energy_costs(
    passive_dir: str,
    period60_dir: str,
    output_path: str = "paper/figures/e1_triggers_energy_costs.png"
):
    """
    绘制E1：触发次数与累计发送能量对比图
    
    Args:
        passive_dir: 智能被动传能实验目录
        period60_dir: 固定60分钟传能实验目录
        output_path: 输出图片路径
    """
    # 加载统计数据
    passive_stats_path = os.path.join(passive_dir, "simulation_statistics.json")
    period60_stats_path = os.path.join(period60_dir, "simulation_statistics.json")
    
    passive_data = load_statistics(passive_stats_path)
    period60_data = load_statistics(period60_stats_path)
    
    # 提取数据
    # 智能被动传能
    passive_transfers = passive_data["network_info"]["total_transfers"]
    passive_energy_j = passive_data["statistics"]["total_sent_energy"]
    passive_energy_kj = passive_energy_j / 1000.0
    
    # 固定60分钟传能
    period60_transfers = period60_data["network_info"]["total_transfers"]
    period60_energy_j = period60_data["statistics"]["total_sent_energy"]
    period60_energy_kj = period60_energy_j / 1000.0
    
    # 计算相对减少百分比
    transfer_reduction = (1 - passive_transfers / period60_transfers) * 100
    energy_reduction = (1 - passive_energy_kj / period60_energy_kj) * 100
    
    print(f"Intelligent Passive Transfer: {passive_transfers} times, {passive_energy_kj:.2f} kJ")
    print(f"Fixed 60-min Periodic Transfer: {period60_transfers} times, {period60_energy_kj:.2f} kJ")
    print(f"Trigger count reduction: {transfer_reduction:.1f}%")
    print(f"Energy reduction: {energy_reduction:.1f}%")
    
    # Define colors: Intelligent Passive Transfer and Fixed 60-min Periodic Transfer
    color_passive = '#2E86AB'  # Blue - Intelligent Passive Transfer
    color_period60 = '#A23B72'  # Magenta - Fixed 60-min Periodic Transfer
    
    # 创建左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 设置x轴位置
    x = np.arange(2)
    width = 0.6  # 柱状图宽度
    
    # Left plot: Trigger count comparison
    bars1_passive = ax1.bar(x[0], passive_transfers, width, 
                            label='Intelligent Passive', color=color_passive, 
                            alpha=0.8, edgecolor='black', linewidth=1.5)
    bars1_period60 = ax1.bar(x[1], period60_transfers, width,
                            label='Fixed 60-min', color=color_period60,
                            alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Trigger Count', fontsize=20, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Intelligent Passive', 'Fixed 60-min'], fontsize=18)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    ax1.text(x[0], passive_transfers, f'{passive_transfers}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax1.text(x[1], period60_transfers, f'{period60_transfers}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Right plot: Cumulative sent energy comparison
    bars2_passive = ax2.bar(x[0], passive_energy_kj, width,
                            label='Intelligent Passive', color=color_passive,
                            alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2_period60 = ax2.bar(x[1], period60_energy_kj, width,
                            label='Fixed 60-min', color=color_period60,
                            alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Cumulative Sent Energy (kJ)', fontsize=20, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Intelligent Passive', 'Fixed 60-min'], fontsize=18)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    ax2.text(x[0], passive_energy_kj, f'{passive_energy_kj:.1f}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax2.text(x[1], period60_energy_kj, f'{period60_energy_kj:.1f}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Set main title
    fig.suptitle('E1: Trigger Count and Cumulative Energy Cost Comparison', fontsize=22, fontweight='bold', y=0.98)
    
    # Add unified legend below title
    handles = [bars1_passive, bars1_period60]
    labels = ['Intelligent Passive Transfer', 'Fixed 60-min Periodic Transfer']
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=18, 
              framealpha=0.9, bbox_to_anchor=(0.5, 0.95))
    
    # Add unified reduction percentage annotation below the plots (split into two lines)
    text_line1 = f'Compared to fixed 60-min periodic transfer, intelligent passive transfer'
    text_line2 = f'reduces trigger count by {transfer_reduction:.1f}% and cumulative sent energy by {energy_reduction:.1f}%'
    full_text = f'{text_line1}\n{text_line2}'
    
    # Position text lower and adjust layout to create more space between plots and text
    fig.text(0.5, 0.02, full_text,
            ha='center', va='bottom', fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5),
            multialignment='center')
    
    # Adjust layout to leave more space between plots and bottom annotation
    plt.tight_layout(rect=[0, 0.13, 1, 0.96])
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    # 实验数据目录
    passive_dir = "data/20251115_144649智能被动"
    period60_dir = "data/20251115_144859固定时长传能"
    
    # 输出路径
    output_path = "paper/figures/e1_triggers_energy_costs.png"
    
    # 绘制图表
    plot_e1_triggers_energy_costs(passive_dir, period60_dir, output_path)









