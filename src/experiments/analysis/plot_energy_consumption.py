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
def extract_energy_consumption_data():
    """解析 simulation_statistics.json 文件，提取总发送能量和总通信能耗。"""
    consumption_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        file_path = os.path.join(DATA_DIR, dir_name, 'simulation_statistics.json')
        if not os.path.exists(file_path):
            print(f"警告: 在 {dir_name} 中未找到 simulation_statistics.json")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        # 提取能量共享成本 (总发送能量)
        sharing_cost = stats.get('statistics', {}).get('total_sent_energy', 0)

        # 提取信息通信成本
        comm_cost = stats.get('additional_info', {}).get('info_transmission', {}).get('total_energy', 0)

        consumption_data.append({
            'method': method_name,
            'Cost Type': 'Energy Sharing',
            'Energy (J)': sharing_cost
        })
        consumption_data.append({
            'method': method_name,
            'Cost Type': 'Communication',
            'Energy (J)': comm_cost
        })
            
    return pd.DataFrame(consumption_data)

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return
        
    df = extract_energy_consumption_data()

    if df.empty:
        print("错误: 未能处理任何实验数据。")
        return

    # --- 绘图 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        data=df,
        x='method',
        y='Energy (J)',
        hue='Cost Type',
        palette='viridis'
    )

    ax.set_title('Overall Energy Consumption Comparison', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Total Energy (Joules)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', rotation=10, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(title='Cost Type', fontsize=22, title_fontsize=22, framealpha=0.9)
    
    # 在柱状图上添加数值标签
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 保存图像 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_5.6_energy_consumption.png')
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
