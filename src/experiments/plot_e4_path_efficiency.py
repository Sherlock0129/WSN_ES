"""
E4 Figure 2: Path efficiency comparison (with and without EETOR)
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16
rcParams['figure.titlesize'] = 22

def parse_efficiencies_from_plans(session_dir: str) -> list[float]:
    """Parses plans.txt to extract a list of path efficiencies."""
    efficiencies = []
    p = os.path.join(session_dir, 'plans.txt')
    if not os.path.exists(p):
        print(f"Warning: plans.txt not found in {session_dir}")
        return efficiencies

    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Try both English and Chinese formats
                m_eng = re.search(r'delivered=([\d.]+), loss=([\d.]+)', line)
                m_chn = re.search(r'传输:\s*([\d.]+?)J\s*\|\s*损失:\s*([\d.]+?)J', line)
                
                m = m_eng or m_chn

                if m:
                    delivered = float(m.group(1))
                    loss = float(m.group(2))
                    total_sent = delivered + loss
                    if total_sent > 0:
                        efficiency = delivered / total_sent
                        efficiencies.append(efficiency)
    except Exception as e:
        print(f"Error parsing {p}: {e}")
    
    return efficiencies

def plot_e4_path_efficiency(
    proposed_dir: str,
    baseline_dir: str,
    output_path: str = "paper/figures/e4_path_eff.png",
    efficiency_threshold: float = 0.15
):
    """
    Plots the path efficiency distribution and low-efficiency ratio comparison.
    """
    # Load efficiency data
    proposed_eff = parse_efficiencies_from_plans(proposed_dir)
    baseline_eff = parse_efficiencies_from_plans(baseline_dir)

    if not proposed_eff or not baseline_eff:
        print("Error: Could not load efficiency data from one or both directories. Aborting plot.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # --- (a) Left plot: Path Efficiency Distribution ---
    bins = np.linspace(0, 1, 50)
    ax1.hist(proposed_eff, bins=bins, color='#2E86AB', alpha=0.7, label='Proposed (with EETOR)', density=True)
    ax1.hist(baseline_eff, bins=bins, color='#A23B72', alpha=0.7, label='Baseline (no EETOR)', density=True)
    ax1.axvline(efficiency_threshold, color='red', linestyle='--', linewidth=2, label=f'Efficiency Threshold ({efficiency_threshold})')
    ax1.set_xlabel('Path Efficiency')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Path Efficiency Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')

    # --- (b) Right plot: Low-Efficiency Path Ratio ---
    low_eff_proposed = (np.array(proposed_eff) < efficiency_threshold).mean() * 100
    low_eff_baseline = (np.array(baseline_eff) < efficiency_threshold).mean() * 100

    labels = ['Proposed (with EETOR)', 'Baseline (no EETOR)']
    values = [low_eff_proposed, low_eff_baseline]
    colors = ['#2E86AB', '#A23B72']

    bars = ax2.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Low-Efficiency Path Ratio (%)')
    ax2.set_title('(b) Low-Efficiency Path Ratio')
    ax2.set_ylim(0, max(values) * 1.2 if values else 10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Overall title and layout
    fig.suptitle('E4: EETOR Path Governance Efficiency Comparison', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    # Get script's directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.join(script_dir, 'data')
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Experiment data directories
    proposed_dir = os.path.join(data_base_dir, "20251113_195728_exp4_baseline_adaptive_duration")
    baseline_dir = os.path.join(data_base_dir, "20251113_195853_exp4_traditional_lyapunov")
    
    # Output path
    output_path = os.path.join(project_root, "paper", "figures", "e4_path_eff.png")
    
    # Plot the chart
    plot_e4_path_efficiency(proposed_dir, baseline_dir, output_path)

