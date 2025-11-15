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
rcParams['font.size'] = 20
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16  # Slightly smaller for more labels
rcParams['figure.titlesize'] = 22

def plot_e4_feedback_scores(csv_path: str, output_path: str = "paper/figures/e4_feedback_scores_adaptive.png"):
    """
    Plots the evolution of feedback scores from a CSV file.
    
    Args:
        csv_path: Path to the feedback_scores.csv file.
        output_path: Path to save the output figure.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {os.path.abspath(csv_path)}")
        return

    # Load data
    df = pd.read_csv(csv_path)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    # Plot main score (total_score)
    ax.plot(df['time_step'], df['total_score'], label='Total Score', color='#0077b6', linewidth=3, zorder=10)

    # Plot sub-scores
    ax.plot(df['time_step'], df['balance_score'], label='Balance (40%)', color='#fca311', linewidth=1.5, alpha=0.8)
    ax.plot(df['time_step'], df['survival_score'], label='Survival (30%)', color='#e5383b', linewidth=1.5, alpha=0.8)
    ax.plot(df['time_step'], df['efficiency_score'], label='Efficiency (20%)', color='#43aa8b', linewidth=1.5, alpha=0.8)
    ax.plot(df['time_step'], df['energy_score'], label='Energy Level (10%)', color='#6a4c93', linewidth=1.5, alpha=0.8)

    # Add a moving average for the total score to show the trend
    window_size = 10
    if len(df) > window_size:
        df['total_score_ma'] = df['total_score'].rolling(window=window_size).mean()
        ax.plot(df['time_step'], df['total_score_ma'], label=f'Total Score ({window_size}-step MA)', color='#023e8a', linestyle='--', linewidth=2.5, zorder=11)

    # Add reference line at y=0
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    # Set labels, title, and legend
    ax.set_xlabel('Time Step (minutes)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Feedback Score', fontsize=20, fontweight='bold')
    ax.set_title('E4: Feedback Score Evolution for Adaptive Scheduler', fontsize=22, fontweight='bold', pad=20)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

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
    # Get script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct data directory path relative to the script's location
    data_base_dir = os.path.join(script_dir, 'data')
    # Project root directory
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Experiment data directory
    exp_dir = os.path.join(data_base_dir, "20251113_195728_exp4_baseline_adaptive_duration")
    csv_file_path = os.path.join(exp_dir, "feedback_scores.csv")
    
    # Output path (relative to project root)
    output_path = os.path.join(project_root, "paper", "figures", "e4_feedback_scores_adaptive.png")
    
    # Plot the chart
    plot_e4_feedback_scores(csv_file_path, output_path)
