import os
import re
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
def parse_plans_for_efficiency(file_path):
    """解析 plans.txt 文件，为每个选中的计划计算路径效率。"""
    efficiency_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配所有“选中计划”部分
    selected_plans_blocks = re.findall(r'选中计划 \(Selected Plans\):\n(.*?)(?=\n\n==|\Z)', content, re.DOTALL)

    for block in selected_plans_blocks:
        # 匹配每一个计划行
        plan_lines = re.findall(r'传输: *([\d\.]+)J \| 损失: *([\d\.]+)J', block)
        for transmitted_str, loss_str in plan_lines:
            transmitted = float(transmitted_str)
            loss = float(loss_str)
            sent = transmitted + loss
            if sent > 0:
                efficiency = transmitted / sent
                efficiency_data.append({'efficiency': efficiency})
            
    return pd.DataFrame(efficiency_data)

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return

    all_efficiency_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        file_path = os.path.join(DATA_DIR, dir_name, 'plans.txt')
        if not os.path.exists(file_path):
            print(f"警告: 在 {dir_name} 中未找到 plans.txt")
            continue
        
        print(f"正在处理: {method_name}")
        df = parse_plans_for_efficiency(file_path)
        if not df.empty:
            df['method'] = method_name
            all_efficiency_data.append(df)

    if not all_efficiency_data:
        print("错误: 未能处理任何路径效率数据。")
        return

    full_df = pd.concat(all_efficiency_data, ignore_index=True)

    # --- 绘图 ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    ax = sns.violinplot(
        data=full_df,
        x='method',
        y='efficiency',
        palette='muted',
        inner='quartile' # 在小提琴内部显示四分位数
    )

    ax.set_title('Path Transfer Efficiency Distribution', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Path Efficiency (η)', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1.05) # 效率范围是0到1
    ax.tick_params(axis='x', rotation=10, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 保存图像 ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, 'fig_5.8_path_efficiency.png')
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
