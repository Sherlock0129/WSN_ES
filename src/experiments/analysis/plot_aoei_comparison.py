import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'legend.title_fontsize': 22,
})

# --- 配置加载函数 ---
def load_config():
    """从 analysis_config.json 加载配置。"""
    config_path = os.path.join(os.path.dirname(__file__), 'analysis_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['DATA_DIR'], config['OUTPUT_DIR'], config['EXPERIMENT_DIRS']

# --- 全局配置 ---
try:
    DATA_DIR, OUTPUT_DIR, EXPERIMENT_DIRS = load_config()
except FileNotFoundError as e:
    print(e)
    DATA_DIR, OUTPUT_DIR, EXPERIMENT_DIRS = '', '', {}

# --- 数据解析函数 ---
def load_aoei_data():
    """加载所有实验的 virtual_center_node_info.csv 文件。"""
    all_aoei_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        file_path = os.path.join(DATA_DIR, dir_name, 'virtual_center_node_info.csv')
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"警告: 在 {dir_name} 中未找到或文件为空: virtual_center_node_info.csv")
            continue
        
        print(f"正在处理: {method_name}")
        df = pd.read_csv(file_path)
        if 'aoi' not in df.columns:
            print(f"错误: {file_path} 中缺少 'aoi' 列。")
            continue

        df['method'] = method_name
        all_aoei_data.append(df)
            
    if not all_aoei_data:
        return pd.DataFrame()
        
    return pd.concat(all_aoei_data, ignore_index=True)

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return

    df = load_aoei_data()

    if df.empty:
        print("错误: 未能处理任何 AOEI 数据，无法生成图像。")
        return

    # --- 绘图 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.boxplot(
        data=df,
        x='method',
        y='aoi',
        palette='pastel',
        showfliers=False
    )

    ax.set_title('Information Freshness (AoEI) Distribution Comparison', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Age of Information (AoEI)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', rotation=10, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # 计算并显示中位数
    medians = df.groupby(['method'])['aoi'].median().round(1)
    vertical_offset = df['aoi'].median() * 0.05

    for xtick in ax.get_xticks():
        ax.text(xtick, medians[ax.get_xticklabels()[xtick].get_text()] + vertical_offset,
                f"{medians[ax.get_xticklabels()[xtick].get_text()]}", 
                horizontalalignment='center', fontsize=18, color='black', weight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 保存图像 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_5.7_aoei_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
