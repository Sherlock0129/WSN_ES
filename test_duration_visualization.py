"""
测试Duration传输的逐分钟可视化

验证：
1. 插值函数正确工作
2. 能量大幅跳变被检测
3. 生成平滑的逐分钟曲线
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from viz.plotter import _interpolate_energy_changes

print("="*70)
print("Duration传输逐分钟可视化测试")
print("="*70)

# 模拟数据：3个节点，10个时间步
time_steps = list(range(1, 11))
energy_data = {
    1: [1000, 1000, 1000, 400, 400, 400, 400, 400, 400, 400],  # Node 1: 第3步大幅下降（donor）
    2: [500, 500, 500, 1050, 1050, 1050, 1050, 1050, 1050, 1050],  # Node 2: 第3步大幅上升（receiver）
    3: [800, 790, 780, 770, 760, 750, 740, 730, 720, 710]  # Node 3: 平滑下降（正常消耗）
}

print("\n原始数据:")
print(f"Time steps: {len(time_steps)} 个")
print(f"Node 1 (donor): 能量变化 {energy_data[1][2]} → {energy_data[1][3]} (Δ={energy_data[1][3]-energy_data[1][2]}J)")
print(f"Node 2 (receiver): 能量变化 {energy_data[2][2]} → {energy_data[2][3]} (Δ={energy_data[2][3]-energy_data[2][2]}J)")
print(f"Node 3 (normal): 平滑下降")

# 执行插值
interpolated_time, interpolated_data = _interpolate_energy_changes(energy_data, time_steps, step_duration=60, threshold=500.0)

print("\n插值后数据:")
print(f"Time steps: {len(interpolated_time)} 个")
print(f"Node 1: {len(interpolated_data[1])} 个数据点")
print(f"Node 2: {len(interpolated_data[2])} 个数据点")
print(f"Node 3: {len(interpolated_data[3])} 个数据点")

if len(interpolated_data[1]) > len(energy_data[1]):
    print(f"\n✓ 插值成功：数据点从 {len(energy_data[1])} 增加到 {len(interpolated_data[1])}")
    
    # 显示Node 1的插值细节
    print(f"\nNode 1 (donor) 在第3-4步之间的插值:")
    # 找到原始第3步和第4步之间的插值点
    idx_start = 2  # 第3步
    idx_end = idx_start + 1
    for i in range(3):  # 显示原始第3步前后的几个点
        if idx_start + i < len(interpolated_data[1]):
            print(f"  索引{idx_start + i}: 时间={interpolated_time[idx_start + i]:.2f}, 能量={interpolated_data[1][idx_start + i]:.2f}J")
    
    print(f"\nNode 2 (receiver) 在第3-4步之间的插值:")
    for i in range(3):
        if idx_start + i < len(interpolated_data[2]):
            print(f"  索引{idx_start + i}: 时间={interpolated_time[idx_start + i]:.2f}, 能量={interpolated_data[2][idx_start + i]:.2f}J")
else:
    print(f"\n○ 未触发插值（能量变化 < 阈值）")

print("\n"+"="*70)
print("测试完成")
print("="*70)
print("\n说明:")
print("- 当能量变化 > 500J 时，自动插值")
print("- 插值点数根据能量变化幅度自动调整")
print("- 生成平滑的逐分钟能量变化曲线")
print("- 用于可视化duration>1的传输过程")

