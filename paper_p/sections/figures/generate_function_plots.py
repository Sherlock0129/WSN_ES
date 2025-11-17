#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成机制设计章节的函数图
包括：
1. M1: 动态AOI上限调整示意图
2. M3: ALDP评分函数随时长变化示意图
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置输出目录
import os
# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir + os.sep

# ============================================================================
# 1. M1: 动态AOI上限调整示意图
# ============================================================================
print("生成 M1: 动态AOI上限调整示意图...")

# 参数设置
A_max_0 = 60  # 基准AOI上限（分钟）
gamma = 200   # 刻度因子
I = np.linspace(0, 1000, 1000)  # 信息量范围

# 计算动态AOI上限
A_max = A_max_0 / (1 + I / gamma)

# 创建图形
plt.figure(figsize=(8, 6))
plt.plot(I, A_max, 'b-', linewidth=2.5, label='动态AOI上限 $A_{\\max,i}(t)$')

# 标注几个关键点
key_points = [0, 100, 300, 500, 1000]
for I_val in key_points:
    if I_val <= I.max():
        A_val = A_max_0 / (1 + I_val / gamma)
        plt.plot(I_val, A_val, 'ro', markersize=8)
        plt.annotate(f'({I_val}, {A_val:.1f})', 
                    xy=(I_val, A_val), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.xlabel('信息量 $I_i(t)$', fontsize=13, fontweight='bold')
plt.ylabel('AOI上限 $A_{\\max,i}(t)$ (分钟)', fontsize=13, fontweight='bold')
plt.title('动态AOI上限调整示意图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='upper right')
plt.xlim(0, 1000)
plt.ylim(0, 65)
plt.tight_layout()
plt.savefig(f'{output_dir}aoei_adaptive_threshold.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: {output_dir}aoei_adaptive_threshold.png")
plt.close()

# ============================================================================
# 2. M3: ALDP评分函数随时长变化示意图
# ============================================================================
print("生成 M3: ALDP评分函数随时长变化示意图...")

# 参数设置
tau = np.linspace(1, 5, 200)  # 传输时长（分钟），更密集的点以获得平滑曲线
E_char = 300  # 特征能量 (J/分钟)
eta_P = 0.6   # 路径效率
Q_norm = 0.5  # 归一化能量缺口
V = 1.0       # Lyapunov参数
w_aoi = 0.3   # 时效惩罚权重
w_info = 0.5  # 信息奖励权重
r_info = 10   # 信息收集速率 (单位/分钟)

# 计算各项指标
E_sent = tau * E_char
E_recv = eta_P * E_sent
E_loss = E_sent - E_recv

B_energy = E_recv * Q_norm
P_loss = V * E_loss
P_aoi = w_aoi * tau * Q_norm
R_info = w_info * r_info * tau * 1.0  # 假设有新信息

Score = B_energy - P_loss - P_aoi + R_info

# 创建图形
plt.figure(figsize=(10, 7))

# 绘制各项指标
plt.plot(tau, B_energy, 'g-', linewidth=2.5, label='能量收益 $B_{\\text{energy}}(\\tau)$')
plt.plot(tau, -P_loss, 'r--', linewidth=2, label='损耗惩罚 $-P_{\\text{loss}}(\\tau)$')
plt.plot(tau, -P_aoi, 'm--', linewidth=2, label='时效惩罚 $-P_{\\text{aoi}}(\\tau)$')
plt.plot(tau, R_info, 'b--', linewidth=2, label='信息奖励 $R_{\\text{info}}(\\tau)$')
plt.plot(tau, Score, 'k-', linewidth=3.5, label='综合评分 $\\text{Score}(\\tau)$')

# 标注最优时长
tau_star_idx = np.argmax(Score)
tau_star = tau[tau_star_idx]
score_star = Score[tau_star_idx]
plt.plot(tau_star, score_star, 'ro', markersize=12, label=f'最优时长 $\\tau^*={tau_star:.2f}$ 分钟')
plt.axvline(tau_star, color='r', linestyle=':', linewidth=2, alpha=0.7)
plt.axhline(score_star, color='r', linestyle=':', linewidth=2, alpha=0.7)

# 添加文本标注
plt.text(tau_star + 0.2, score_star, f'$\\tau^*={tau_star:.2f}$\\n$\\text{{Score}}={score_star:.1f}$', 
         fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.xlabel('传输时长 $\\tau$ (分钟)', fontsize=13, fontweight='bold')
plt.ylabel('评分值', fontsize=13, fontweight='bold')
plt.title('ALDP评分函数随时长变化示意图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='best', framealpha=0.9)
plt.xlim(1, 5)
plt.tight_layout()
plt.savefig(f'{output_dir}aldp_score_function.png', dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: {output_dir}aldp_score_function.png")
plt.close()

print("\n所有函数图生成完成！")
print(f"输出目录: {output_dir}")

