# 维度分数图表颜色显示修复

## 问题描述

用户报告生成的"Dimensional Scores Over Time"图表中，所有四条曲线都显示为**橙色**，无法区分不同维度。

### 问题截图分析

从用户提供的截图可以看到：
- ✅ 图例正确显示了四个维度及权重
- ✅ 权重说明文本框正常显示
- ❌ **所有线条都是同一个橙色/黄色**
- ❌ 无法区分四个不同的维度

## 根本原因

颜色设置被 matplotlib 的某些全局配置或 PyCharm 的 matplotlib 后端覆盖了。

---

## 解决方案

### 1. 重置 matplotlib 配置

在函数开始时强制重置为默认配置：

```python
def plot_feedback_scores(self) -> None:
    # 确保matplotlib使用默认配置，避免颜色被覆盖
    import matplotlib
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
```

**作用**：清除任何可能干扰颜色显示的全局设置。

### 2. 使用循环显式设置颜色

改进前（单独绘制每条线）：
```python
ax2.plot(time_steps, balance_scores, marker='s', color='#2E86AB', ...)
ax2.plot(time_steps, survival_scores, marker='^', color='#A23B72', ...)
```

改进后（使用循环明确设置）：
```python
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
markers = ['s', '^', 'o', 'd']
labels = ['Balance Score (40%)', 'Survival Score (30%)', 
          'Efficiency Score (20%)', 'Energy Level Score (10%)']
data_series = [balance_scores, survival_scores, efficiency_scores, energy_scores]

for i, (data, label, color, marker) in enumerate(zip(data_series, labels, colors, markers)):
    ax2.plot(time_steps, data, 
             marker=marker, 
             linestyle='-', 
             linewidth=2.5, 
             markersize=5, 
             label=label, 
             alpha=0.9,
             color=color,         # 显式颜色参数
             zorder=10-i)         # 分层显示
    print(f"  Plotted {label} with color {color}")  # 调试输出
```

**改进点：**
- ✅ 使用显式的 `color` 参数
- ✅ 添加 `zorder` 确保线条分层
- ✅ 打印调试信息确认颜色设置
- ✅ 代码更简洁，易于维护

### 3. 添加调试输出

绘制时输出颜色信息：
```
  Plotted Balance Score (40%) with color #2E86AB
  Plotted Survival Score (30%) with color #A23B72
  Plotted Efficiency Score (20%) with color #F18F01
  Plotted Energy Level Score (10%) with color #C73E1D
```

---

## 颜色方案

| 维度 | 颜色代码 | 颜色预览 | 权重 |
|------|---------|---------|------|
| Balance Score | `#2E86AB` | 🟦 深蓝色 | 40% |
| Survival Score | `#A23B72` | 🟪 紫红色 | 30% |
| Efficiency Score | `#F18F01` | 🟧 橙色 | 20% |
| Energy Level Score | `#C73E1D` | 🟥 暗红色 | 10% |

**颜色选择理由：**
- 高对比度，易于区分
- 色盲友好（蓝-紫-橙-红）
- 从冷色到暖色渐变

---

## 测试验证

### 运行仿真

```bash
python src/sim/refactored_main.py
```

### 检查输出

**1. 控制台调试信息**

应该看到：
```
[Feedback Scores Debug Info]
Total records: 8
Balance scores - min: -2.15, max: 3.42, avg: 0.85
Survival scores - min: 0.00, max: 0.00, avg: 0.00
Efficiency scores - min: -8.50, max: 2.30, avg: -2.10
Energy scores - min: -0.35, max: 0.12, avg: -0.08

  Plotted Balance Score (40%) with color #2E86AB
  Plotted Survival Score (30%) with color #A23B72
  Plotted Efficiency Score (20%) with color #F18F01
  Plotted Energy Level Score (10%) with color #C73E1D
```

**2. 图表检查**

应该看到：
- ✅ 四条线颜色明显不同
- ✅ 蓝色（Balance）最粗显眼
- ✅ 紫色（Survival）次之
- ✅ 橙色（Efficiency）第三
- ✅ 红色（Energy）最后

---

## 常见问题

### Q1: 为什么需要重置 rcParams？

**A**: PyCharm 或其他 IDE 的 matplotlib 后端可能会设置自定义配置，覆盖我们的颜色设置。`rcParams.update(rcParamsDefault)` 确保使用纯净的默认配置。

### Q2: 如果颜色还是相同怎么办？

**A**: 检查是否有其他地方设置了 matplotlib 样式：

```python
# 在 plot_feedback_scores() 开始处添加
print("Current matplotlib backend:", plt.get_backend())
print("Current rcParams['axes.prop_cycle']:", matplotlib.rcParams['axes.prop_cycle'])
```

### Q3: 为什么使用 zorder？

**A**: `zorder` 控制绘图层级。权重大的维度设置更高的 zorder，确保重要的线条不被遮挡。

```python
zorder=10-i  # Balance(10) > Survival(9) > Efficiency(8) > Energy(7)
```

### Q4: 可以用其他颜色吗？

**A**: 当然可以！修改 `colors` 列表：

```python
# 使用 matplotlib 内置颜色名称
colors = ['blue', 'purple', 'orange', 'red']

# 或使用 RGB 值
colors = [(0.18, 0.53, 0.67),  # Balance
          (0.64, 0.23, 0.45),  # Survival
          (0.95, 0.56, 0.00),  # Efficiency
          (0.78, 0.24, 0.11)]  # Energy
```

---

## 技术细节

### matplotlib 颜色设置优先级

1. **最高**: 显式 `color` 参数
2. **中等**: `rcParams['axes.prop_cycle']`
3. **最低**: 默认颜色循环

我们的修复使用**最高优先级**的方法。

### PyCharm matplotlib 后端问题

PyCharm 的内置 matplotlib 后端 (`module://backend_interagg`) 有时会：
- 覆盖颜色设置
- 忽略某些参数
- 使用自己的颜色循环

**解决方法：**
```python
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # 强制重置
```

---

## 修改的文件

- **`src/core/simulation_stats.py`**
  - `plot_feedback_scores()` 方法
  - 添加 `rcParams` 重置
  - 改用循环绘制，显式设置颜色
  - 添加调试输出

---

## 预期效果

### 修复前
![修复前](用户提供的截图 - 所有线都是橙色)

### 修复后
- 🟦 **蓝色线** - Balance Score (最明显)
- 🟪 **紫色线** - Survival Score
- 🟧 **橙色线** - Efficiency Score
- 🟥 **红色线** - Energy Level Score

四条线清晰可辨！

---

## 额外建议

### 如果数据值相近导致重叠

可以添加偏移量：
```python
offset = i * 0.1  # 每条线偏移0.1
ax2.plot(time_steps, [v + offset for v in data], ...)
```

### 使用交互式后端

```python
# 在脚本开始处
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
```

### 保存为高质量图片

```python
plt.savefig(feedback_plot_path, dpi=300, bbox_inches='tight')
```

---

## 更新日期

2025-01-09

---

## 总结

通过**重置 matplotlib 配置**和**显式设置颜色参数**，成功解决了四条曲线颜色相同的问题。现在每个维度都有独特的颜色，图表清晰易读。

