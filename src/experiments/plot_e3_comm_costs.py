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
rcParams['font.size'] = 26
rcParams['axes.labelsize'] = 26
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24
rcParams['figure.titlesize'] = 30

def load_statistics(json_path):
    """从JSON文件加载统计数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_e3_comm_costs(
    opportunistic_dir: str,
    adcr_dir: str,
    output_path: str = "paper/figures/e3_comm_costs_reports.png"
):
    """
    绘制E3：通信能耗与上报频次对比图（仅比较 Opportunistic 与 ADCR）
    """
    # 加载统计数据
    opp_data = load_statistics(os.path.join(opportunistic_dir, "simulation_statistics.json"))
    adcr_data = load_statistics(os.path.join(adcr_dir, "simulation_statistics.json"))
    
    # 提取数据
    methods = ['Opportunistic', 'ADCR']
    
    comm_energy = [
        opp_data["additional_info"]["info_transmission"]["total_energy"] / 1000.0,  # kJ
        adcr_data["additional_info"]["info_transmission"]["total_energy"] / 1000.0   # kJ
    ]
    
    # 对于ADCR，信息上报独立于能量传输，这里用总传输次数作为代理指标
    report_freq = [
        opp_data["network_info"]["total_transfers"],
        adcr_data["network_info"]["total_transfers"] # ADCR的能量传输次数
    ]

    # 创建图形和双Y轴
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    
    x = np.arange(len(methods))
    width = 0.35
    
    # 左Y轴：信息通信能耗
    color1 = '#2E86AB'  # 蓝色
    bars1 = ax1.bar(x - width/2, comm_energy, width, label='Info Comm. Energy (kJ)', color=color1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Reporting Strategy', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Info Comm. Energy (kJ)', fontsize=26, fontweight='bold', color=color1)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22, labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=22)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 右Y轴：上报频次
    ax2 = ax1.twinx()
    color2 = '#A23B72'  # 紫红色
    bars2 = ax2.bar(x + width/2, report_freq, width, label='Reporting/Transfer Freq.', color=color2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Reporting/Transfer Freq.', fontsize=26, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelsize=22, labelcolor=color2)

    # 调整两侧Y轴上限，使两组柱子的视觉高度有明显差异
    # 左轴按能耗的最大值留白15%，右轴按频次最大值留白35%（柱体会更“矮”一点，避免与左轴视觉等高）
    max_energy = max(comm_energy) if len(comm_energy) > 0 else 1
    max_freq = max(report_freq) if len(report_freq) > 0 else 1
    ax1.set_ylim(0, max_energy * 1.15)
    ax2.set_ylim(0, max_freq * 1.35)

    # 在柱状图上添加数值标签
    def autolabel(bars, ax, fmt='{:.1f}'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=20, fontweight='bold')

    autolabel(bars1, ax1, '{:.2f}')
    autolabel(bars2, ax2, '{:.0f}')

    # 设置总标题
    fig.suptitle('E3: Communication Cost and Reporting Frequency Comparison', fontsize=30, fontweight='bold', y=1.02)

    # 在标题下面添加统一图例
    handles = [bars1[0], bars2[0]]
    labels = ['Info Comm. Energy (kJ)', 'Reporting/Transfer Freq.']
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=24, 
              framealpha=0.9, bbox_to_anchor=(0.5, 0.96))
    
    # 调整布局，为标题和图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
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

    # Resolve project/data dirs first to build useful defaults
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.join(script_dir, 'data')
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    default_opp = os.path.join(data_base_dir, "20251113_195332_exp3_baseline_opportunistic")
    default_adcr = os.path.join(data_base_dir, "20251116_230231_exp3_adcr")

    parser = argparse.ArgumentParser(description='Plot E3: Communication Cost and Reporting Frequency Comparison.')
    parser.add_argument('--opportunistic_dir', type=str, default=default_opp, help='Directory for the opportunistic reporting experiment.')
    parser.add_argument('--adcr_dir', type=str, default=default_adcr, help='Directory for the ADCR experiment.')
    parser.add_argument('--output_path', type=str, default="paper/figures/e3_comm_costs_reports.png", help='Output path for the figure.')

    args = parser.parse_args()

    # Resolve output path relative to project root
    output_path = os.path.join(project_root, args.output_path)

    # Plot the chart
    plot_e3_comm_costs(args.opportunistic_dir, args.adcr_dir, output_path)
