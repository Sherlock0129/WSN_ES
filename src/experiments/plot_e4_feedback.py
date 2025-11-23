"""
E4 Figure 1: Feedback score evolution for AdaptiveDurationAwareScheduler
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 18  # Slightly smaller for more labels
rcParams['figure.titlesize'] = 24

def plot_e4_feedback_scores(csv_path: str, output_path: str = "paper/figures/e4_feedback_scores_adaptive.png"):
    """
    流程化主图（简洁版）：按 决策 → 执行 → 反馈 顺序绘制三张图
    - 子图1（决策）：selected_plan_count、K_used（左轴），avg_duration（右轴）
    - 子图2（执行）：sent_total、delivered_total、total_loss（左轴），efficiency（右轴）
    - 子图3（反馈）：total_score + 10-step MA
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 18), dpi=300)

    # -------- 子图1：决策（Duration-aware 结果） --------
    ax1 = axes[0]
    has_left = False
    # 决策图仅保留 selected_plan_count
    if 'selected_plan_count' in cols:
        has_left = True
        ax1.plot(df['time_step'], df['selected_plan_count'], label='selected_plan_count', color='#2E86AB', linewidth=2.0)
    ax1.set_ylabel('Selected Plans (count)')

    if has_left:
        ax1.legend(loc='upper right', ncol=1, framealpha=0.9)
    else:
        ax1.text(0.5, 0.5, 'No decision metrics in CSV', transform=ax1.transAxes, ha='center', va='center')

    # 为图例留出纵向空间
    try:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * 1.18)
    except Exception:
        pass
    ax1.set_xlabel('Time Step (minutes)')
    ax1.set_title('Decision (Duration-aware)')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # -------- 子图2：执行（能量与效率） --------
    ax2 = axes[1]
    has_exec = False
    if 'sent_total' in cols:
        has_exec = True
        ax2.plot(df['time_step'], df['sent_total'], label='sent_total', color='#8e44ad', linewidth=2.0)
    if 'delivered_total' in cols:
        has_exec = True
        ax2.plot(df['time_step'], df['delivered_total'], label='delivered_total', color='#16a085', linewidth=2.0)
    if 'total_loss' in cols:
        has_exec = True
        ax2.plot(df['time_step'], df['total_loss'], label='total_loss', color='#c0392b', linewidth=1.8)
    ax2.set_ylabel('Energy (J)')

    # 右轴：efficiency（0~1）
    if 'efficiency' in cols:
        ax2b = ax2.twinx()
        ax2b.plot(df['time_step'], df['efficiency'], label='efficiency', color='#27AE60', linewidth=2.0, linestyle='--')
        ax2b.set_ylabel('Efficiency')
        ax2b.set_ylim(0, 1)
        lines_l, labels_l = ax2.get_legend_handles_labels()
        lines_r, labels_r = ax2b.get_legend_handles_labels()
        ax2.legend(lines_l + lines_r, labels_l + labels_r, loc='upper right', ncol=2, framealpha=0.9)
    else:
        if has_exec:
            ax2.legend(loc='upper right', ncol=3, framealpha=0.9)
        else:
            ax2.text(0.5, 0.5, 'No execution metrics in CSV', transform=ax2.transAxes, ha='center', va='center')

    # 为图例留出纵向空间
    try:
        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(ymin2, ymax2 * 1.18)
    except Exception:
        pass
    ax2.set_xlabel('Time Step (minutes)')
    ax2.set_title('Execution (Energy and Efficiency)')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # -------- 子图3：反馈（四维分数 + 总分） --------
    ax3 = axes[2]
    has_any_score = False
    # 四个维度分数
    if 'balance_score' in cols:
        has_any_score = True
        ax3.plot(df['time_step'], df['balance_score'], label='Balance (40%)', color='#fca311', linewidth=1.8, alpha=0.9)
    if 'survival_score' in cols:
        has_any_score = True
        ax3.plot(df['time_step'], df['survival_score'], label='Survival (30%)', color='#e5383b', linewidth=1.8, alpha=0.9)
    if 'efficiency_score' in cols:
        has_any_score = True
        ax3.plot(df['time_step'], df['efficiency_score'], label='Efficiency (20%)', color='#43aa8b', linewidth=1.8, alpha=0.9)
    if 'energy_score' in cols:
        has_any_score = True
        ax3.plot(df['time_step'], df['energy_score'], label='Energy Level (10%)', color='#6a4c93', linewidth=1.8, alpha=0.9)
    # 总分及其MA
    if 'total_score' in cols:
        has_any_score = True
        ax3.plot(df['time_step'], df['total_score'], label='Total Score', color='#0077b6', linewidth=2.4, zorder=10)
    window_size = 10
    if len(df) > window_size:
            df['total_score_ma'] = df['total_score'].rolling(window=window_size, min_periods=1).mean()
            ax3.plot(df['time_step'], df['total_score_ma'], label=f'Total Score ({window_size}-step MA)', color='#023e8a', linestyle='--', linewidth=2.0, zorder=11)
    if has_any_score:
        ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax3.legend(loc='upper right', ncol=2, framealpha=0.9)
    else:
        ax3.text(0.5, 0.5, 'No feedback scores in CSV', transform=ax3.transAxes, ha='center', va='center')

    # 为图例留出纵向空间
    try:
        ymin3, ymax3 = ax3.get_ylim()
        ax3.set_ylim(ymin3, ymax3 * 1.18)
    except Exception:
        pass
    ax3.set_xlabel('Time Step (minutes)')
    ax3.set_ylabel('Feedback Score')
    ax3.set_title('Feedback (Scores Breakdown)')
    ax3.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout(h_pad=2.0)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Get script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct data directory path relative to the script's location
    data_base_dir = os.path.join(script_dir, 'data')
    # Project root directory
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Experiment data directory
    exp_dir = os.path.join(data_base_dir, "20251117_214833_exp4_baseline_adaptive_duration")
    csv_file_path = os.path.join(exp_dir, "feedback_scores.csv")
    
    # Output path (relative to project root)
    output_path = os.path.join(project_root, "paper", "figures", "e4_feedback_scores_adaptive.png")
    
    # Plot the chart
    plot_e4_feedback_scores(csv_file_path, output_path)
