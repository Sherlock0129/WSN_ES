"""
绘制E3实验图1：通信能耗与上报频次对比图
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置字体和字体大小
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18
rcParams['figure.titlesize'] = 22

def load_statistics(json_path):
    """从JSON文件加载统计数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_e3_comm_costs(
    opportunistic_dir: str,
    direct_dir: str,
    adcr_dir: str,
    output_path: str = "paper/figures/e3_comm_costs_reports.png"
):
    """
    绘制E3：通信能耗与上报频次对比图
    """
    # 加载统计数据
    opp_data = load_statistics(os.path.join(opportunistic_dir, "simulation_statistics.json"))
    direct_data = load_statistics(os.path.join(direct_dir, "simulation_statistics.json"))
    adcr_data = load_statistics(os.path.join(adcr_dir, "simulation_statistics.json"))
    
    # 提取数据
    labels = ['Opportunistic', 'Direct Report', 'ADCR']
    
    comm_energy = [
        opp_data["additional_info"]["info_transmission"]["total_energy"] / 1000.0,  # kJ
        direct_data["additional_info"]["info_transmission"]["total_energy"] / 1000.0, # kJ
        adcr_data["additional_info"]["info_transmission"]["total_energy"] / 1000.0   # kJ
    ]
    
    # 对于ADCR，信息上报独立于能量传输，这里用总传输次数作为代理指标
    report_freq = [
        opp_data["network_info"]["total_transfers"],
        direct_data["network_info"]["total_transfers"],
        adcr_data["network_info"]["total_transfers"] # ADCR的能量传输次数
    ]

    # 创建图形和双Y轴
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    
    x = np.arange(len(labels))
    width = 0.35
    
    # 左Y轴：信息通信能耗
    color1 = '#2E86AB'  # 蓝色
    bars1 = ax1.bar(x - width/2, comm_energy, width, label='Info Comm. Energy (kJ)', color=color1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Reporting Strategy', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Info Comm. Energy (kJ)', fontsize=20, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=18)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 右Y轴：上报频次
    ax2 = ax1.twinx()
    color2 = '#A23B72'  # 紫红色
    bars2 = ax2.bar(x + width/2, report_freq, width, label='Reporting/Transfer Freq.', color=color2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Reporting/Transfer Freq.', fontsize=20, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 在柱状图上添加数值标签
    def autolabel(bars, ax, fmt='{:.1f}'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=16, fontweight='bold')

    autolabel(bars1, ax1, '{:.2f}')
    autolabel(bars2, ax2, '{:.0f}')

    # 设置总标题
    fig.suptitle('E3: Communication Cost and Reporting Frequency Comparison', fontsize=22, fontweight='bold', y=1.02)

    # 在标题下面添加统一图例
    handles = [bars1[0], bars2[0]]
    labels = ['Info Comm. Energy (kJ)', 'Reporting/Transfer Freq.']
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=18, 
              framealpha=0.9, bbox_to_anchor=(0.5, 0.95))
    
    # 调整布局，为标题和图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot E3: Communication Cost and Reporting Frequency Comparison.')
    parser.add_argument('--opportunistic_dir', type=str, required=True, help='Directory for the opportunistic reporting experiment.')
    parser.add_argument('--direct_dir', type=str, required=True, help='Directory for the direct report experiment.')
    parser.add_argument('--adcr_dir', type=str, required=True, help='Directory for the ADCR experiment.')
    parser.add_argument('--output_path', type=str, default="paper/figures/e3_comm_costs_reports.png", help='Output path for the figure.')

    args = parser.parse_args()

    # Get project root to resolve output path correctly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_path = os.path.join(project_root, args.output_path)

    # Plot the chart
    plot_e3_comm_costs(args.opportunistic_dir, args.direct_dir, args.adcr_dir, output_path)
