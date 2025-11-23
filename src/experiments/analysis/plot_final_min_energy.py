import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
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
def get_final_min_energy(file_path):
    """从 plans.txt 解析最后一个时间步的最低节点能量。如果文件或数据缺失，则返回0。"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"警告: 文件不存在或为空: {os.path.basename(file_path)}. 默认最低能量为 0.")
            return 0

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        time_steps_data = re.findall(r'时间步 t=(\d+)\n-+\n节点能量状态:(.*?)(?=\n\n候选计划|\n\n=)', content, re.DOTALL)
        
        if not time_steps_data:
            print(f"警告: 在 {os.path.basename(file_path)} 中未找到能量状态块. 默认最低能量为 0.")
            return 0

        last_energy_block = time_steps_data[-1][1]
        energies = [float(e) for e in re.findall(r'Node \d+: ([\d\.]+)J', last_energy_block)]
        
        if energies:
            return min(energies)
        else:
            print(f"警告: 在 {os.path.basename(file_path)} 的最后一个时间步未找到节点能量. 默认最低能量为 0.")
            return 0
            
    except Exception as e:
        print(f"解析 {os.path.basename(file_path)} 时出错: {e}. 默认最低能量为 0.")
        return 0

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return

    summary_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        plans_path = os.path.join(DATA_DIR, dir_name, 'plans.txt')
        final_min_energy = get_final_min_energy(plans_path)
        summary_data.append({
            'method': method_name,
            'Final Minimum Energy (J)': final_min_energy
        })

    if not summary_data:
        print("错误: 未能处理任何数据，无法生成图像。")
        return

    df = pd.DataFrame(summary_data)

    # --- 绘图 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    ax = sns.barplot(
        data=df,
        x='method',
        y='Final Minimum Energy (J)',
        palette='plasma'
    )

    ax.set_title('Final Minimum Node Energy Comparison', fontsize=20, weight='bold')
    ax.set_xlabel('Method', fontsize=16)
    ax.set_ylabel('Energy of the Weakest Node (J)', fontsize=16)
    ax.tick_params(axis='x', rotation=10, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=13)

    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_5.10_final_min_energy.png')
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
