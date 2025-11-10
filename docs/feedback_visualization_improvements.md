# 反馈评分可视化改进说明

## 问题描述

用户报告 "Dimensional Scores Over Time" 图表显示有问题，无法清楚看到四个维度的分数。

---

## 改进内容

### 1. 添加数据调试信息

在绘图前打印数据统计，便于诊断问题：

```python
print(f"\n[Feedback Scores Debug Info]")
print(f"Total records: {len(self.feedback_scores)}")
print(f"Balance scores - min: {min(balance_scores):.2f}, max: {max(balance_scores):.2f}, avg: {np.mean(balance_scores):.2f}")
print(f"Survival scores - min: {min(survival_scores):.2f}, max: {max(survival_scores):.2f}, avg: {np.mean(survival_scores):.2f}")
print(f"Efficiency scores - min: {min(efficiency_scores):.2f}, max: {max(efficiency_scores):.2f}, avg: {np.mean(efficiency_scores):.2f}")
print(f"Energy scores - min: {min(energy_scores):.2f}, max: {max(energy_scores):.2f}, avg: {np.mean(energy_scores):.2f}\n")
```

### 2. 增强线条可视化

#### 修改前
```python
ax2.plot(time_steps, balance_scores, marker='s', linestyle='-', 
         linewidth=1.5, markersize=3, label='Balance Score', alpha=0.8)
```

#### 修改后
```python
ax2.plot(time_steps, balance_scores, marker='s', linestyle='-', 
         linewidth=2.5, markersize=5, label='Balance Score (40%)', 
         color='#2E86AB', alpha=0.9)
```

**改进点：**
- ✅ 线条宽度：1.5 → 2.5（更粗更明显）
- ✅ 标记大小：3 → 5（更大更清晰）
- ✅ 透明度：0.8 → 0.9（更不透明）
- ✅ 颜色：使用明确的十六进制颜色代码
- ✅ 图例：添加权重百分比

### 3. 使用区分度高的颜色

为四个维度选择了对比度高的颜色：

| 维度 | 颜色代码 | 颜色名称 | 权重 |
|------|---------|---------|------|
| Balance Score | `#2E86AB` | 蓝色 (Deep Blue) | 40% |
| Survival Score | `#A23B72` | 紫色 (Purple) | 30% |
| Efficiency Score | `#F18F01` | 橙色 (Orange) | 20% |
| Energy Level Score | `#C73E1D` | 红色 (Red) | 10% |

**选择理由：**
- 高对比度，易于区分
- 色盲友好的配色方案
- 符合数据重要性（权重从高到低）

### 4. 改进零线显示

```python
# 修改前
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# 修改后
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
```

更明显的零参考线。

### 5. 优化图例和网格

```python
# 图例改进
ax2.legend(loc='best', ncol=2, fontsize=9, framealpha=0.9)

# 网格改进
ax2.grid(True, linestyle='--', alpha=0.3)  # 降低透明度，不遮挡线条
```

### 6. 添加权重说明

在图表左上角添加权重说明文本框：

```python
info_text = 'Weights: Balance=40%, Survival=30%, Efficiency=20%, Energy=10%'
ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
        fontsize=8, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
```

---

## 改进效果对比

### 修改前的问题
- ❌ 线条太细，不明显
- ❌ 标记太小，难以看清
- ❌ 颜色对比度不够
- ❌ 没有显示权重信息
- ❌ 图例信息不完整

### 修改后的改进
- ✅ 线条加粗（2.5px）
- ✅ 标记加大（5px）
- ✅ 使用高对比度颜色
- ✅ 图例显示权重百分比
- ✅ 添加权重说明文本框
- ✅ 调试信息便于诊断

---

## 可视化参数总结

### 线条样式参数

```python
{
    'linewidth': 2.5,        # 线条宽度（原1.5）
    'markersize': 5,         # 标记大小（原3）
    'alpha': 0.9,            # 透明度（原0.8）
    'linestyle': '-',        # 实线
}
```

### 颜色方案

```python
colors = {
    'balance': '#2E86AB',      # 蓝色 - 最重要（40%）
    'survival': '#A23B72',     # 紫色 - 次重要（30%）
    'efficiency': '#F18F01',   # 橙色 - 中等（20%）
    'energy': '#C73E1D'        # 红色 - 较小（10%）
}
```

### 标记样式

```python
markers = {
    'balance': 's',      # 方形
    'survival': '^',     # 三角形
    'efficiency': 'o',   # 圆形
    'energy': 'd'        # 菱形
}
```

---

## 测试验证

运行仿真后检查：

```bash
python src/sim/refactored_main.py
```

**检查点：**
1. ✅ 控制台显示调试信息
2. ✅ 四条曲线清晰可见
3. ✅ 颜色区分明显
4. ✅ 图例显示完整（含权重）
5. ✅ 权重说明文本框显示

**预期输出示例：**
```
[Feedback Scores Debug Info]
Total records: 8
Balance scores - min: -2.15, max: 3.42, avg: 0.85
Survival scores - min: 0.00, max: 0.00, avg: 0.00
Efficiency scores - min: -8.50, max: 2.30, avg: -2.10
Energy scores - min: -0.35, max: 0.12, avg: -0.08
```

---

## 常见问题

### Q1: 为什么某些线重叠？

**A**: 如果数据值相近，线条可能重叠。可以通过调试信息查看数值范围。

### Q2: 某个维度的分数始终为0？

**A**: 
- Survival Score 在节点存活数不变时为0（正常）
- Energy Score 通常值较小（权重只有10%）

### Q3: 图例遮挡了线条？

**A**: 可以修改图例位置：
```python
ax2.legend(loc='upper left')  # 或 'upper right', 'lower left', 'lower right'
```

### Q4: 如何调整线条颜色？

**A**: 修改颜色代码：
```python
color='#YOUR_HEX_COLOR'
```

推荐使用在线工具选择色盲友好的配色：
- https://colorbrewer2.org/
- https://coolors.co/

---

## 进一步优化建议

### 1. 动态Y轴范围

如果某个维度的分数范围很小，可以使用双Y轴：

```python
ax2_right = ax2.twinx()
ax2_right.plot(time_steps, energy_scores, ...)
```

### 2. 添加平滑曲线

对于噪声较大的数据，可以添加移动平均线：

```python
from scipy.ndimage import uniform_filter1d
smooth_balance = uniform_filter1d(balance_scores, size=5)
ax2.plot(time_steps, smooth_balance, linestyle='--', alpha=0.5)
```

### 3. 交互式图表

使用 plotly 替代 matplotlib：

```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_steps, y=balance_scores, name='Balance'))
fig.show()
```

---

## 修改的文件

- **`src/core/simulation_stats.py`**
  - `plot_feedback_scores()` 方法

---

## 更新日期

2025-01-09

---

## 总结

通过增强线条、优化颜色、添加权重信息和调试输出，显著改善了"Dimensional Scores Over Time"图表的可读性和信息密度。现在四个维度的分数应该能清晰显示了。

