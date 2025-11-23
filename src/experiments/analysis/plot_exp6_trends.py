import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')

EXP6_DATASETS = [
    {
        "name": "Exp6 ALDP",
        "dir": "20251119_170559_exp6_aldp",
        "color": "#1f77b4",
    },
    {
        "name": "Exp6 Traditional Lyapunov",
        "dir": "20251119_170808_exp6_traditional_lyapunov",
        "color": "#ff7f0e",
    },
]

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
})


def load_feedback_scores(dataset_dir: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, dataset_dir, 'feedback_scores.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Feedback file not found: {path}')
    df = pd.read_csv(path)
    required = {'time_step', 'total_score', 'impact'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns {missing} in {path}')
    return df.sort_values('time_step').reset_index(drop=True)


def plot_total_score_trend(dataframes):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(12, 7))
    for info, df in dataframes:
        plt.plot(df['time_step'], df['total_score'], label=info['name'], color=info['color'], linewidth=2.5)

    plt.title('Exp6 Total Score Trend Comparison', fontsize=28, fontweight='bold', pad=20)
    plt.xlabel('Time Step', fontsize=24, fontweight='bold')
    plt.ylabel('Total Score', fontsize=24, fontweight='bold')
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.legend(fontsize=22, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'exp6_total_score_trend.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved: {out_path}')


def plot_positive_ratio(dataframes):
    ratios = []
    for info, df in dataframes:
        impact_col = df['impact'].fillna('').astype(str)
        positive_mask = impact_col.str.contains('positive', case=False) | impact_col.str.contains('\u6b63\u76f8\u5173')
        total = len(df)
        ratio = positive_mask.sum() / total if total else 0.0
        ratios.append({'method': info['name'], 'ratio': ratio})

    ratio_df = pd.DataFrame(ratios)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=ratio_df, x='method', y='ratio', palette=[info['color'] for info, _ in dataframes])
    ax.set_title('Exp6 Positive Scheduling Ratio Comparison', fontsize=28, fontweight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Positive Scheduling Ratio', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    for patch, (_, df) in zip(ax.patches, dataframes):
        height = patch.get_height()
        ax.annotate(f"{height:.2%}", (patch.get_x() + patch.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=18, weight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'exp6_positive_ratio.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved: {out_path}')


def load_final_min_energy(dataset_dir: str) -> float:
    path = os.path.join(DATA_DIR, dataset_dir, 'plans.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Plan file not found: {path}')

    last_energies = None
    node_line_pattern = re.compile(r'Node\s+(\d+):\s*([0-9.]+)J')

    with open(path, 'r', encoding='utf-8') as f:
        capture = False
        current = {}
        for line in f:
            stripped = line.strip()
            if stripped.startswith('\u8282\u70b9\u80fd\u91cf\u72b6\u6001') or stripped.startswith('Node energy state'):
                capture = True
                current = {}
                continue
            if capture:
                if not stripped:
                    if current:
                        last_energies = current
                    capture = False
                    continue
                match = node_line_pattern.search(stripped)
                if match:
                    node_id = int(match.group(1))
                    energy = float(match.group(2))
                    current[node_id] = energy

    if not last_energies:
        raise ValueError(f'No node energy state data found in {path}')

    min_energy = min(last_energies.values())
    return min_energy


def plot_final_min_energy_bar(dataset_infos):
    records = []
    colors = []
    for info in dataset_infos:
        min_energy = load_final_min_energy(info['dir'])
        records.append({'method': info['name'], 'min_energy': min_energy})
        colors.append(info['color'])

    df = pd.DataFrame(records)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df, x='method', y='min_energy', palette=colors)
    ax.set_title('Exp6 Final-Step Minimum Node Energy', fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Min Node Energy at Final Step (J)', fontsize=22, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f"{height:,.0f} J", (patch.get_x() + patch.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=18, weight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'exp6_final_min_energy.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    dataframes = []
    for info in EXP6_DATASETS:
        try:
            df = load_feedback_scores(info['dir'])
        except (FileNotFoundError, ValueError) as exc:
            print(f"[Skip] {info['name']}: {exc}")
            continue
        dataframes.append((info, df))

    if not dataframes:
        print('No Exp6 dataset could be loaded; abort plotting.')
        return

    plot_total_score_trend(dataframes)
    plot_positive_ratio(dataframes)
    plot_final_min_energy_bar([info for info, _ in dataframes])


if __name__ == '__main__':
    main()

