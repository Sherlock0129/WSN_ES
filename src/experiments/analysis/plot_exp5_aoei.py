import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')

EXP5_DATASETS = [
    {
        "name": "Exp5 Dynamic AoEI Cap",
        "dir": "20251120_001125_exp5_dynamic_aoei_cap",
        "color": "#1f77b4",
    },
    {
        "name": "Exp5 Static AoEI Cap",
        "dir": "20251120_001341_exp5_static_aoei_cap",
        "color": "#ff7f0e",
    },
]

AOI_MAX = 1200
NUM_BINS = 3
BIN_EDGES = [i * AOI_MAX / NUM_BINS for i in range(NUM_BINS + 1)]
BIN_LABELS = [f"{int(BIN_EDGES[i])}-{int(BIN_EDGES[i+1])}" for i in range(NUM_BINS)]

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 36,
    'legend.fontsize': 22,
})


def load_aoei_series(dataset_dir: str) -> pd.Series:
    file_path = os.path.join(DATA_DIR, dataset_dir, 'virtual_center_node_info.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'未找到文件: {file_path}')
    df = pd.read_csv(file_path, usecols=['aoi'])
    if df.empty:
        raise ValueError(f'{file_path} 无有效数据')
    series = df['aoi'].dropna()
    if series.empty:
        raise ValueError(f'{file_path} 中 aoI 列为空')
    return series


def compute_bin_ratios(series: pd.Series) -> pd.Series:
    clipped = series[series <= AOI_MAX]
    if clipped.empty:
        return pd.Series([0] * NUM_BINS, index=BIN_LABELS)

    counts = pd.cut(clipped, bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True).value_counts().sort_index()
    ratios = counts / counts.sum()
    return ratios.reindex(BIN_LABELS, fill_value=0)


def plot_binned_comparison(ratios_df: pd.DataFrame):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))

    bar_width = 0.35
    x = range(len(BIN_LABELS))

    for idx, dataset in enumerate(EXP5_DATASETS):
        offsets = [pos + (idx - 0.5) * bar_width for pos in x]
        plt.bar(
            offsets,
            ratios_df.loc[dataset['name']],
            width=bar_width,
            label=dataset['name'],
            color=dataset['color'],
        )

    plt.xticks(x, BIN_LABELS)
    plt.xlabel('AoI Range (minutes)', fontsize=24, fontweight='bold')
    plt.ylabel('Proportion', fontsize=24, fontweight='bold')
    plt.title('Exp5 AoEI Segment Distribution Comparison', fontsize=28, fontweight='bold', pad=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylim(0, 1)
    plt.legend(fontsize=22, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'exp5_aoei_segment_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    ratio_records = []

    for dataset in EXP5_DATASETS:
        try:
            series = load_aoei_series(dataset['dir'])
            ratios = compute_bin_ratios(series)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[Skip] {dataset['name']}: {exc}")
            continue
        ratio_records.append((dataset['name'], ratios))

    if not ratio_records:
        print("No Exp5 datasets were processed.")
        return

    ratios_df = pd.DataFrame(
        {name: ratios for name, ratios in ratio_records}
    ).T  # index: dataset name, columns: bins

    plot_binned_comparison(ratios_df)


if __name__ == '__main__':
    main()

