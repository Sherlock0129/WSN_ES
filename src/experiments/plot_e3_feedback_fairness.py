"""
E3 Figure 2: Feedback score and weak-node service coverage comparison

- Input: two experiment directories (opportunistic, ADCR)
- Data needed per directory:
  * simulation_statistics.json -> statistics.feedback.avg_score
  * virtual_center_node_info.csv (or simulation_results.csv) -> node energy to identify weak nodes
  * plans.txt (or simulation_results.csv) -> receivers served (to measure coverage)

Coverage definition (robust heuristic):
- Compute each node's average energy across the run (excluding physical center node id 0)
  * Prefer reading virtual_center_node_info.csv (many rows with 'node_id' and 'energy')
  * Fallback: simulation_results.csv with columns ['time','node_id','energy'] or wide format
- Identify weak set = bottom 20% nodes by average energy (ceil)
- Extract served receivers set by parsing plans.txt:
  * Regex for 'receiver' and integer id; collect all ids seen
  * Fallback: simulation_results.csv column 'receiver_id' if present
- Coverage = |weak ∩ served| / |weak|

The script is defensive and will log if some metric cannot be computed; in that case it will write 'NA' on the bar label.

NOTE: Do NOT run this script automatically in tooling if the user asked not to. It is intended to be executed manually.
"""

from __future__ import annotations
import os
import json
import math
import re
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Style
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 26
rcParams['axes.labelsize'] = 26
rcParams['xtick.labelsize'] = 22
rcParams['ytick.labelsize'] = 22
rcParams['legend.fontsize'] = 24
rcParams['figure.titlesize'] = 30

# ------------------------ IO helpers ------------------------

def load_feedback_avg(session_dir: str, debug: bool = False) -> Optional[float]:
    p = os.path.join(session_dir, 'simulation_statistics.json')
    if debug: print(f"  [fb] Trying to load feedback from: {os.path.abspath(p)}")
    if not os.path.exists(p):
        if debug: print("  [fb] FAILED: simulation_statistics.json not found.")
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        score = data.get('statistics', {}).get('feedback', {}).get('avg_score', None)
        if debug: print(f"  [fb] SUCCESS: Loaded avg_score = {score}")
        return score
    except Exception as e:
        if debug: print(f"  [fb] FAILED: Error reading or parsing JSON: {e}")
        return None

def read_virtual_center_info(session_dir: str) -> Optional[pd.DataFrame]:
    for name in ['virtual_center_node_info.csv', 'virtual_center_node_info.parquet']:
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            try:
                if name.endswith('.parquet'):
                    return pd.read_parquet(p)
                return pd.read_csv(p)
            except Exception:
                return None
    return None

def read_sim_results(session_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(session_dir, 'simulation_results.csv')
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def average_energy_per_node(session_dir: str, debug: bool = False) -> Optional[Dict[int, float]]:
    # Prefer VC info
    vc_path = os.path.join(session_dir, 'virtual_center_node_info.csv')
    if debug: print(f"  [cov] Trying to load energy data from: {os.path.abspath(vc_path)}")
    df = read_virtual_center_info(session_dir)
    if df is not None and 'node_id' in df.columns and 'energy' in df.columns:
        if debug: print("  [cov] Found virtual_center_node_info.csv, calculating average energy.")
        try:
            grp = df.groupby('node_id')['energy'].mean()
            d = grp.to_dict()
            return {int(k): float(v) for k, v in d.items()}
        except Exception as e:
            if debug: print(f"  [cov] FAILED: Error processing virtual_center_node_info.csv: {e}")
            pass
    elif debug:
        print("  [cov] virtual_center_node_info.csv not found or invalid.")

    # Fallback: simulation_results
    sim_path = os.path.join(session_dir, 'simulation_results.csv')
    if debug: print(f"  [cov] Trying fallback: {os.path.abspath(sim_path)}")
    df = read_sim_results(session_dir)
    if df is not None:
        if debug: print("  [cov] Found simulation_results.csv, attempting to calculate average energy.")
        # Try long format
        cols = [c.lower() for c in df.columns]
        colmap = {c.lower(): c for c in df.columns}
        if 'node_id' in cols and 'energy' in cols:
            try:
                grp = df.groupby(colmap['node_id'])[colmap['energy']].mean()
                return {int(k): float(v) for k, v in grp.to_dict().items()}
            except Exception:
                pass
        # Try wide format: columns like energy_1, energy_2, ...
        energy_cols = [c for c in df.columns if re.match(r'(?i)energy_\\d+', c)]
        if energy_cols:
            try:
                means = df[energy_cols].mean(axis=0)
                d: Dict[int, float] = {}
                for c, v in means.items():
                    m = re.search(r'(\\d+)', c)
                    if m:
                        d[int(m.group(1))] = float(v)
                return d
            except Exception:
                pass
    elif debug:
        print("  [cov] simulation_results.csv not found or invalid.")

    # Fallback 2: Parse energy from plans.txt
    plans_path = os.path.join(session_dir, 'plans.txt')
    if debug: print(f"  [cov] Trying fallback 2: {os.path.abspath(plans_path)}")
    if os.path.exists(plans_path):
        try:
            energy_readings = []
            with open(plans_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    m = re.search(r'Node\s+(\d+):\s*([\d.]+?)J', line)
                    if m:
                        energy_readings.append({'node_id': int(m.group(1)), 'energy': float(m.group(2))})
            if energy_readings:
                df_plans = pd.DataFrame(energy_readings)
                grp = df_plans.groupby('node_id')['energy'].mean()
                if debug: print(f"  [cov] SUCCESS: Parsed {len(df_plans)} energy readings from plans.txt.")
                return {int(k): float(v) for k, v in grp.to_dict().items()}
        except Exception as e:
            if debug: print(f"  [cov] FAILED: Error parsing plans.txt: {e}")

    if debug: print("  [cov] FAILED: All methods to get average energy failed.")
    return None

def parse_receivers_from_plans(session_dir: str, debug: bool = False) -> Set[int]:
    receivers: Set[int] = set()
    p = os.path.join(session_dir, 'plans.txt')
    if debug: print(f"  [cov] Trying to parse receivers from: {os.path.abspath(p)}")
    if not os.path.exists(p):
        if debug: print("  [cov] plans.txt not found.")
        return receivers
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Updated regex to match the format '路径: 30->1'
                m = re.search(r'路径:\s*\d+->(\d+)', line)
                if m:
                    receivers.add(int(m.group(1)))
        if debug: print(f"  [cov] Found {len(receivers)} receivers in plans.txt.")
    except Exception as e:
        if debug: print(f"  [cov] FAILED: Error reading plans.txt: {e}")
        return receivers
    return receivers

def parse_receivers_from_results(session_dir: str, debug: bool = False) -> Set[int]:
    receivers: Set[int] = set()
    df = read_sim_results(session_dir)
    if df is None:
        return receivers
    cols_lower = [c.lower() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    if 'receiver_id' in cols_lower:
        try:
            vals = df[colmap['receiver_id']].dropna().astype(int).unique().tolist()
            receivers = set(int(v) for v in vals)
            if debug: print(f"  [cov] Found {len(receivers)} receivers from 'receiver_id' column.")
            return receivers
        except Exception:
            return receivers
    return receivers

def compute_weak_coverage(session_dir: str, bottom_ratio: float = 0.2, debug: bool = False) -> Optional[float]:
    if debug: print(f"  [cov] Computing weak-node coverage for: {session_dir}")

    energies = average_energy_per_node(session_dir, debug=debug)
    if not energies:
        if debug: print("  [cov] FAILED: Could not calculate average energy per node.")
        return None
    if debug: print(f"  [cov] Found average energies for {len(energies)} nodes.")

    energies = {nid: e for nid, e in energies.items() if nid != 0}
    if not energies:
        if debug: print("  [cov] FAILED: No nodes left after excluding center node 0.")
        return None

    n = len(energies)
    k = max(1, math.ceil(n * bottom_ratio))
    sorted_nodes = sorted(energies.items(), key=lambda kv: kv[1])
    weak_nodes = {nid for nid, _ in sorted_nodes[:k]}
    if debug: print(f"  [cov] Identified {len(weak_nodes)} weak nodes (bottom {k}/{n}): {weak_nodes}")

    receivers = parse_receivers_from_plans(session_dir, debug=debug)
    if not receivers:
        if debug: print("  [cov] plans.txt yielded no receivers, trying simulation_results.csv")
        receivers = parse_receivers_from_results(session_dir, debug=debug)
    
    # If no energy receivers are found, coverage is 0. Do not fall back to info coverage.
    if not receivers:
        if debug: print("  [cov] No energy receivers found in logs. Weak-node energy service coverage is 0%.")
        return 0.0

    covered = weak_nodes.intersection(receivers)
    if debug: print(f"  [cov] Weak nodes served/reported: {len(covered)} -> {covered}")
    
    coverage_rate = len(covered) / len(weak_nodes) if weak_nodes else 0.0
    if debug: print(f"  [cov] Final coverage rate: {coverage_rate:.2%}")
    return coverage_rate

# ------------------------ Plot ------------------------

def plot_e3_feedback_fairness(opportunistic_dir: str,
                               adcr_dir: str,
                               output_path: str = 'paper/figures/e3_feedback_fairness.png') -> None:
    methods = ['Opportunistic', 'ADCR']
    dirs = [opportunistic_dir, adcr_dir]

    feedback_avgs: List[Optional[float]] = []
    coverage_rates: List[Optional[float]] = []

    for i, d in enumerate(dirs):
        print(f"\n--- Processing directory: {methods[i]} ({os.path.abspath(d)}) ---")
        feedback_avgs.append(load_feedback_avg(d, debug=True))
        coverage_rates.append(compute_weak_coverage(d, debug=True))
        print("--- End of Processing ---")

    y1 = [np.nan if v is None else float(v) for v in feedback_avgs]
    y2 = [np.nan if v is None else float(v) * 100.0 for v in coverage_rates]

    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    x = np.arange(len(methods))
    width = 0.35

    color1 = '#2E86AB'
    bars1 = ax1.bar(x - width/2, y1, width, label='Avg Feedback Score',
                    color=color1, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Feedback Score', fontsize=26, fontweight='bold', color=color1)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22, labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=22)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

    ax2 = ax1.twinx()
    color2 = '#A23B72'
    bars2 = ax2.bar(x + width/2, y2, width, label='Weak-node Coverage (%)',
                    color=color2, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Weak-node Coverage (%)', fontsize=26, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelsize=22, labelcolor=color2)

    def autolabel(bars, ax, fmt: str):
        for b in bars:
            h = b.get_height()
            label = 'NA' if np.isnan(h) else (fmt.format(h))
            ax.annotate(label,
                        xy=(b.get_x() + b.get_width()/2, 0 if np.isnan(h) else h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=20, fontweight='bold')

    autolabel(bars1, ax1, '{:.2f}')
    autolabel(bars2, ax2, '{:.0f}')

    fig.suptitle('E3: Feedback Score and Weak-node Service Coverage', fontsize=30, fontweight='bold', y=1.02)
    handles = [bars1[0], bars2[0]]
    labels = ['Avg Feedback Score', 'Weak-node Coverage (%)']
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=24,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.97))

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\nFigure saved to: {output_path}')


if __name__ == '__main__':
    # Get script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct data directory path relative to the script's location
    data_base_dir = os.path.join(script_dir, 'data')
    # Project root directory
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Experiment data directories
    opportunistic = os.path.join(data_base_dir, "20251113_195332_exp3_baseline_opportunistic")
    adcr = os.path.join(data_base_dir, "20251116_230231_exp3_adcr")

    # Output path (relative to project root)
    output_path = os.path.join(project_root, "paper", "figures", "e3_feedback_fairness.png")

    plot_e3_feedback_fairness(opportunistic, adcr, output_path=output_path)
