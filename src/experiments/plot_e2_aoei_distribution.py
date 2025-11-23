"""
E2 Figure 1: AOEI Distribution Comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Style configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 22
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
rcParams['figure.titlesize'] = 24

def load_aoei_data(csv_path: str) -> pd.Series:
    """Loads the AOEI column from the virtual_center_node_info.csv file."""
    if not os.path.exists(csv_path):
        print(f"Warning: Data file not found at {os.path.abspath(csv_path)}")
        return pd.Series(dtype='float64')
    
    try:
        df = pd.read_csv(csv_path)
        # Exclude the physical center node (ID=0)
        df = df[df['node_id'] != 0]
        return df['aoi'].dropna()
    except Exception as e:
        print(f"Error reading or processing {csv_path}: {e}")
        return pd.Series(dtype='float64')

def plot_e2_aoei_distribution(
    baseline_dir: str,
    no_reward_dir: str,
    output_path: str = "paper/figures/e2_aoei_hist.png"
):
    """
    Plots the AOEI distribution comparison for E2 experiment.
    """
    # Load data
    baseline_aoei = load_aoei_data(os.path.join(baseline_dir, "virtual_center_node_info.csv"))
    no_reward_aoei = load_aoei_data(os.path.join(no_reward_dir, "virtual_center_node_info.csv"))

    if baseline_aoei.empty or no_reward_aoei.empty:
        print("Error: Could not load AOEI data from one or both directories. Aborting plot.")
        return

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Plot KDE distributions
    sns.kdeplot(baseline_aoei, ax=ax, color='#2E86AB', label='Baseline (with Info Reward)', fill=True, alpha=0.6, linewidth=2.5)
    sns.kdeplot(no_reward_aoei, ax=ax, color='#A23B72', label='No Info Reward', fill=True, alpha=0.6, linewidth=2.5)

    # Add vertical lines for the mean
    mean_baseline = baseline_aoei.mean()
    mean_no_reward = no_reward_aoei.mean()
    ax.axvline(mean_baseline, color='#2E86AB', linestyle='--', linewidth=2, label=f'Mean (Baseline): {mean_baseline:.1f}')
    ax.axvline(mean_no_reward, color='#A23B72', linestyle='--', linewidth=2, label=f'Mean (No Reward): {mean_no_reward:.1f}')

    # Set labels, title, and legend
    ax.set_xlabel('Age of Energy Information (AOEI, minutes)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Density', fontsize=22, fontweight='bold')
    ax.set_title('E2: AOEI Distribution with and without Information Reward', fontsize=24, fontweight='bold', pad=24)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    # Adjust layout
    plt.tight_layout()

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
    # Assuming the data is in 'src/experiments/data/'
    data_base_dir = os.path.join(script_dir, 'data')
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Experiment data directories
    baseline_dir = os.path.join(data_base_dir, "20251115_222442_exp2_baseline_with_info")
    no_reward_dir = os.path.join(data_base_dir, "20251115_222712_exp2_no_info_reward")
    
    # Output path
    output_path = os.path.join(project_root, "paper", "figures", "e2_aoei_hist.png")
    
    # Plot the chart
    plot_e2_aoei_distribution(baseline_dir, no_reward_dir, output_path)

