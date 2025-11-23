import os
import re
import json
import pandas as pd
import numpy as np
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
def parse_plans_for_cv(file_path):
    """解析 plans.txt 文件，提取每个时间步的能量状态并计算CV。"""
    cv_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配每个时间步的数据块
    time_steps_data = re.findall(r'时间步 t=(\d+)\n-+\n节点能量状态:(.*?)(?=\n\n候选计划|\n\n=)', content, re.DOTALL)

    for t_str, energy_block in time_steps_data:
        time_step = int(t_str)
        
        # 提取所有节点的能量值
        energies = [float(e) for e in re.findall(r'Node \d+: ([\d\.]+)J', energy_block)]
        
        if len(energies) > 1:
            energies_np = np.array(energies)
            mean_energy = np.mean(energies_np)
            std_energy = np.std(energies_np)
            
            # 计算变异系数 (CV)
            cv = std_energy / mean_energy if mean_energy > 0 else 0
            cv_data.append({'time_step': time_step, 'cv': cv})
            
    return pd.DataFrame(cv_data)

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return

    all_cv_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        file_path = os.path.join(DATA_DIR, dir_name, 'plans.txt')
        if not os.path.exists(file_path):
            print(f"警告: 在 {dir_name} 中未找到 plans.txt")
            continue
        
        print(f"正在处理: {method_name}")
        df = parse_plans_for_cv(file_path)
        if not df.empty:
            df['method'] = method_name
            all_cv_data.append(df)

    if not all_cv_data:
        print("错误: 未能处理任何能量均衡度数据。")
        return

    full_df = pd.concat(all_cv_data, ignore_index=True)

    # --- 绘图 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.lineplot(
        data=full_df,
        x='time_step',
        y='cv',
        hue='method',
        style='method',
        markers=False,
        linewidth=2.5
    )

    ax.set_title('Energy Equity Evolution (Coefficient of Variation)', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel('Time Step', fontsize=24, fontweight='bold')
    ax.set_ylabel('Energy Coefficient of Variation (CV)', fontsize=24, fontweight='bold')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(title='Method', fontsize=22, title_fontsize=22, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 保存图像 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_5.9_energy_equity.png')
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
