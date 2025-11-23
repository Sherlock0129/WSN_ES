import os
import re
import json
import pandas as pd
import numpy as np

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

# --- 数据解析与计算函数 ---
def get_first_death_time(file_path, total_duration=10080):
    """从 plans.txt 解析首个节点死亡时间。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        time_steps_data = re.findall(r'时间步 t=(\d+)\n-+\n节点能量状态:(.*?)(?=\n\n候选计划|\n\n=)', content, re.DOTALL)
        if not time_steps_data:
            return total_duration # 文件为空或格式不符
        initial_nodes = len(re.findall(r'Node \d+:', time_steps_data[0][1]))
        for t_str, energy_block in time_steps_data:
            alive_nodes = len(re.findall(r'Node \d+: (?!0\.00J|0J)[\d\.]+J', energy_block))
            if alive_nodes < initial_nodes:
                return int(t_str)
        return total_duration
    except Exception:
        return total_duration

def get_mean_path_efficiency(file_path):
    """从 plans.txt 计算平均路径效率。"""
    efficiencies = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        selected_plans_blocks = re.findall(r'选中计划 \(Selected Plans\):\n(.*?)(?=\n\n==|\Z)', content, re.DOTALL)
        for block in selected_plans_blocks:
            plan_lines = re.findall(r'传输: *([\d\.]+)J \| 损失: *([\d\.]+)J', block)
            for transmitted_str, loss_str in plan_lines:
                transmitted = float(transmitted_str)
                loss = float(loss_str)
                sent = transmitted + loss
                if sent > 0:
                    efficiencies.append(transmitted / sent)
        return np.mean(efficiencies) if efficiencies else 0
    except Exception:
        return 0

def get_final_cv(file_path):
    """从 plans.txt 计算最后一个时间步的能量CV。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        time_steps_data = re.findall(r'时间步 t=(\d+)\n-+\n节点能量状态:(.*?)(?=\n\n候选计划|\n\n=)', content, re.DOTALL)
        if not time_steps_data:
            return 0
        last_energy_block = time_steps_data[-1][1]
        energies = [float(e) for e in re.findall(r'Node \d+: ([\d\.]+)J', last_energy_block)]
        if len(energies) > 1:
            mean = np.mean(energies)
            std = np.std(energies)
            return std / mean if mean > 0 else 0
        return 0
    except Exception:
        return 0

def get_mean_aoei(file_path):
    """从 virtual_center_node_info.csv 计算平均AOEI。"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return np.nan
        df = pd.read_csv(file_path)
        return df['aoi'].mean()
    except Exception:
        return np.nan

# --- 主逻辑 ---
def main():
    if not EXPERIMENT_DIRS:
        print("错误: 实验目录配置为空或未加载，请检查 analysis_config.json 文件。")
        return

    summary_data = []

    for method_name, dir_name in EXPERIMENT_DIRS.items():
        print(f"正在处理: {method_name}")
        stats_path = os.path.join(DATA_DIR, dir_name, 'simulation_statistics.json')
        plans_path = os.path.join(DATA_DIR, dir_name, 'plans.txt')
        aoi_path = os.path.join(DATA_DIR, dir_name, 'virtual_center_node_info.csv')

        metrics = {'Method': method_name}

        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            metrics['Total Sent Energy (kJ)'] = stats.get('statistics', {}).get('total_sent_energy', 0) / 1000
            metrics['Total Comm. Cost (kJ)'] = stats.get('additional_info', {}).get('info_transmission', {}).get('total_energy', 0) / 1000
        else:
            metrics['Total Sent Energy (kJ)'] = np.nan
            metrics['Total Comm. Cost (kJ)'] = np.nan

        metrics['Network Lifetime (min)'] = get_first_death_time(plans_path)
        metrics['Mean Path Efficiency (%)'] = get_mean_path_efficiency(plans_path) * 100
        metrics['Final Energy CV'] = get_final_cv(plans_path)
        metrics['Mean AoEI (min)'] = get_mean_aoei(aoi_path)

        summary_data.append(metrics)

    summary_df = pd.DataFrame(summary_data).set_index('Method')
    
    ordered_cols = [
        'Network Lifetime (min)',
        'Total Sent Energy (kJ)',
        'Total Comm. Cost (kJ)',
        'Mean AoEI (min)',
        'Mean Path Efficiency (%)',
        'Final Energy CV'
    ]
    summary_df = summary_df[ordered_cols]

    print("\n--- 汇总性能指标 --- ")
    print(summary_df.round(2))

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, 'table_5.2_summary_metrics.csv')
    summary_df.round(2).to_csv(csv_path)
    print(f"\nCSV文件已保存至: {csv_path}")

    latex_str = summary_df.round(2).to_latex(
        caption='四种方法核心性能指标汇总表',
        label='tab:summary_metrics',
        position='!htbp',
        column_format='l' + 'c'*len(summary_df.columns)
    )
    print("\n--- LaTeX 表格代码 --- ")
    print(latex_str)
    latex_path = os.path.join(OUTPUT_DIR, 'table_5.2_summary_metrics.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"\nLaTeX代码已保存至: {latex_path}")

if __name__ == '__main__':
    main()
