"""
Script to generate spatial heatmaps and Sankey diagrams for energy flow.
- Heatmap: from virtual_center_node_info.csv
- Sankey: now parsed from plans.txt (Selected Plans), aggregating delivered energy along path edges
"""

import os
import re
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define paths
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')

# Ensure output directory exists
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# Experiment directories and their plot-friendly names
EXPERIMENTS = {
    '20251112_203918_exp1_baseline_passive': 'Passive-20251112',
    '20251112_204201_exp1_active_60min': 'Active-60m-20251112',
    '20251112_204544_exp2_baseline_with_info': 'With-Info-20251112',
    '20251112_204721_exp2_no_info_reward': 'No-Reward-20251112',
    '20251112_205209_exp2_no_info_routing': 'No-Routing-20251112',
    '20251112_205335_exp2_no_info_both': 'No-Info-Both-20251112',
    '20251112_205844_exp3_baseline_opportunistic': 'Opportunistic-20251112',
    # New batch (2025-11-13)
    '20251113_193911_exp1_baseline_passive': 'Passive',
    '20251113_194024_exp1_active_60min': 'Active-60m',
    '20251113_194355_exp2_baseline_with_info': 'With-Info',
    '20251113_194518_exp2_no_info_reward': 'No-Reward',
    '20251113_194851_exp2_no_info_routing': 'No-Routing',
    '20251113_195000_exp2_no_info_both': 'No-Info-Both',
    '20251113_195332_exp3_baseline_opportunistic': 'Opportunistic',
    '20251113_195454_exp3_adcr': 'ADCR',
    '20251113_195613_exp3_direct_report': 'Direct-Report',
    '20251113_195728_exp4_baseline_adaptive_duration': 'AdaptiveDuration',
    '20251113_195853_exp4_traditional_lyapunov': 'Lyapunov',
    '20251113_200227_exp4_duration_aware_only': 'DurationAware',
}

def load_node_csv(exp_dir):
    """Load node info CSV if exists."""
    node_file = os.path.join(exp_dir, 'virtual_center_node_info.csv')
    if os.path.exists(node_file):
        return pd.read_csv(node_file)
    return None

SELECTED_PLAN_PATTERN = re.compile(
    r"^\s*\[\s*\d+\]\s*路径:\s*([0-9\-\>]+)\s*\|\s*距离:\s*([0-9\.]+)m\s*\|\s*传输:\s*([0-9\.]+)J\s*\|\s*损失:\s*([0-9\.]+)J\s*$"
)

def parse_plans_file(exp_dir):
    """Parse plans.txt and aggregate delivered energy along path edges.
    Returns:
        flows_df: DataFrame with columns [u, v, delivered]
    """
    plans_path = os.path.join(exp_dir, 'plans.txt')
    if not os.path.exists(plans_path):
        print(f"No plans.txt in {exp_dir}")
        return None

    edge_to_energy = {}
    in_selected_block = False

    def extract_from_line(line: str):
        """Robustly extract path and delivered from a Selected Plan line."""
        if '路径' not in line or '传输' not in line:
            return None
        try:
            # path between '路径:' and the next '|'
            path_part = line.split('路径', 1)[1]
            path_part = path_part.split(':', 1)[1]
            path_str = path_part.split('|', 1)[0].strip()
            # delivered between '传输:' and 'J'
            delivered_part = line.split('传输', 1)[1]
            delivered_part = delivered_part.split(':', 1)[1]
            delivered_str = delivered_part.split('J', 1)[0].strip()
            delivered_val = float(delivered_str)
            return path_str, delivered_val
        except Exception:
            return None

    with open(plans_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if '选中计划' in line:
                in_selected_block = True
                continue
            if in_selected_block and (line == '' or line.startswith('=') or line.startswith('时间步')):
                in_selected_block = False
                continue
            if not in_selected_block:
                continue
            result = extract_from_line(line)
            if not result:
                continue
            path_str, delivered = result
            try:
                node_ids = [int(tok) for tok in path_str.split('->') if tok.strip()]
            except ValueError:
                continue
            for u, v in zip(node_ids[:-1], node_ids[1:]):
                edge = (u, v)
                edge_to_energy[edge] = edge_to_energy.get(edge, 0.0) + delivered

    if not edge_to_energy:
        print(f"Parsed zero flows from {plans_path}")
        return None

    flows_df = pd.DataFrame([{'u': u, 'v': v, 'delivered': en} for (u, v), en in edge_to_energy.items()])
    flows_df.sort_values('delivered', ascending=False, inplace=True)
    return flows_df


def plot_energy_heatmap(df_nodes, exp_name, time_step):
    """Generate and save a spatial energy heatmap for a specific time step."""
    t_data = df_nodes[df_nodes['t'] == time_step]
    if t_data.empty:
        print(f"No data for time step {time_step} in {exp_name}")
        return

    points = t_data[['position_x', 'position_y']].values
    values = t_data['energy'].values

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(points) >= 3:
        try:
            grid_x, grid_y = np.mgrid[points[:,0].min():points[:,0].max():100j,
                                      points[:,1].min():points[:,1].max():100j]
            grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
            im = ax.imshow(grid_z.T, extent=(points[:,0].min(), points[:,0].max(),
                                             points[:,1].min(), points[:,1].max()),
                           origin='lower', cmap='viridis', aspect='auto')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Node Energy (J)')
            ax.scatter(points[:,0], points[:,1], c=values, cmap='viridis',
                       edgecolors='white', linewidth=0.5, s=50)
        except Exception as e:
            print(f"Heatmap fallback to scatter for {exp_name} at t={time_step}: {e}")
            sc = ax.scatter(points[:,0], points[:,1], c=values, cmap='viridis',
                            edgecolors='white', linewidth=0.5, s=80)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Node Energy (J)')
    else:
        print(f"Not enough points ({len(points)}) for interpolation at t={time_step} in {exp_name}. Scatter only.")
        sc = ax.scatter(points[:,0], points[:,1], c=values, cmap='viridis',
                        edgecolors='white', linewidth=0.5, s=100)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Node Energy (J)')

    ax.set_title(f'Energy Distribution: {exp_name} at t={time_step}')
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')

    output_path = os.path.join(OUTPUT_FIG_DIR, f'heatmap_{exp_name}_t{time_step}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def plot_sankey_from_plans(exp_dir, exp_name):
    """Generate Sankey diagram using aggregated flows from plans.txt."""
    flows_df = parse_plans_file(exp_dir)
    if flows_df is None or flows_df.empty:
        print(f"No flows to plot for {exp_name}")
        return

    # Take top-N edges for clarity
    topN = 30
    flows = flows_df.head(topN)

    # Build node list
    node_ids = sorted(set(flows['u']).union(set(flows['v'])))
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    sources = flows['u'].map(node_index).tolist()
    targets = flows['v'].map(node_index).tolist()
    values = flows['delivered'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=12, thickness=14, line=dict(color="black", width=0.4),
            label=[f"Node {nid}" for nid in node_ids],
        ),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=f"Top {topN} Energy Flows (from plans): {exp_name}", font_size=10)

    out_png = os.path.join(OUTPUT_FIG_DIR, f'sankey_{exp_name}.png')
    out_html = os.path.join(OUTPUT_FIG_DIR, f'sankey_{exp_name}.html')
    try:
        fig.write_image(out_png, scale=2)
        print(f"Saved Sankey PNG to {out_png}")
    except Exception as e:
        print(f"PNG export failed (install kaleido to enable). Saving HTML instead. Error: {e}")
        fig.write_html(out_html, include_plotlyjs='cdn')
        print(f"Saved Sankey HTML to {out_html}")


def main():
    # Iterate through defined experiments if directory exists
    for exp_id, exp_name in EXPERIMENTS.items():
        exp_dir = os.path.join(BASE_DATA_DIR, exp_id)
        if not os.path.isdir(exp_dir):
            # Skip missing
            continue
        print(f"\nProcessing {exp_name}: {exp_dir}")

        # Heatmap (last time step)
        df_nodes = load_node_csv(exp_dir)
        if df_nodes is not None and 't' in df_nodes.columns:
            plot_energy_heatmap(df_nodes, exp_name, df_nodes['t'].max())
        else:
            print(f"No node CSV for heatmap in {exp_dir}")

        # Sankey from plans.txt
        plot_sankey_from_plans(exp_dir, exp_name)

if __name__ == '__main__':
    main()
