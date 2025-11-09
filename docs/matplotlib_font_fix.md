# Matplotlib 中文字体问题修复说明

## 问题描述

在运行仿真时，matplotlib 生成图表时出现大量中文字符缺失警告：

```
UserWarning: Glyph 32593 (\N{CJK UNIFIED IDEOGRAPH-7F51}) missing from font(s) DejaVu Sans.
UserWarning: Glyph 32476 (\N{CJK UNIFIED IDEOGRAPH-7EDC}) missing from font(s) DejaVu Sans.
...
```

**原因**：matplotlib 默认字体 DejaVu Sans 不支持中文字符。

---

## 解决方案

将所有图表标签改为英文，避免使用中文。

---

## 修改内容

### 文件：`src/core/simulation_stats.py`

修改了 `plot_feedback_scores()` 方法中的所有标签：

#### 1. 图表标题和轴标签

| 原中文 | 新英文 |
|--------|--------|
| 调度反馈分数随时间变化 | Overall Feedback Score Over Time |
| 时间步 | Time Step |
| 反馈分数 | Feedback Score |
| 各维度分数变化 | Dimensional Scores Over Time |
| 分数 | Score |
| 调度影响分布统计 | Impact Distribution Statistics |
| 次数 | Count |

#### 2. 图例标签

| 原中文 | 新英文 |
|--------|--------|
| 显著改善阈值 | Significant Improvement |
| 显著恶化阈值 | Significant Degradation |
| 能量均衡性 | Balance Score |
| 网络存活率 | Survival Score |
| 传输效率 | Efficiency Score |
| 能量水平 | Energy Level Score |

#### 3. 柱状图分类

| 原中文 | 新英文 |
|--------|--------|
| 正相关 | Positive |
| 中性 | Neutral |
| 负相关 | Negative |

#### 4. 统计摘要

| 原中文 | 新英文 |
|--------|--------|
| 统计摘要 | Statistics Summary |
| 平均分 | Average |
| 最高分 | Maximum |
| 最低分 | Minimum |
| 总次数 | Total |

#### 5. 控制台输出

| 原中文 | 新英文 |
|--------|--------|
| 反馈分数图表已保存到 | Feedback scores chart saved to |
| 没有反馈分数数据可供绘制 | No feedback score data available for plotting |

---

## 修改前后对比

### 修改前（中文标签）

```python
ax1.set_title('调度反馈分数随时间变化 (Overall Feedback Score)', fontsize=12)
ax1.set_xlabel('时间步 (Time Step)', fontsize=10)
ax1.set_ylabel('反馈分数 (Score)', fontsize=10)
```

### 修改后（英文标签）

```python
ax1.set_title('Overall Feedback Score Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time Step', fontsize=10)
ax1.set_ylabel('Feedback Score', fontsize=10)
```

---

## 测试验证

运行仿真：

```bash
python src/sim/refactored_main.py
```

**预期结果**：
- ✅ 不再出现中文字符缺失警告
- ✅ 图表正常显示，所有标签使用英文
- ✅ 图表保存成功

---

## 其他可选方案（未采用）

如果未来仍需要使用中文，有以下替代方案：

### 方案1：配置中文字体

在绘图代码前添加：

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
matplotlib.rcParams['axes.unicode_minus'] = False
```

**优点**：可以显示中文  
**缺点**：需要系统安装相应字体，跨平台兼容性差

### 方案2：使用 fontproperties

```python
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='path/to/chinese_font.ttf')
ax.set_title('中文标题', fontproperties=font)
```

**优点**：更灵活  
**缺点**：需要配置字体路径，代码复杂

### 方案3：混合显示（当前方案）

使用英文作为主标签，必要时在括号中添加中文：

```python
ax.set_title('Overall Feedback Score (调度反馈分数)', fontsize=12)
```

**优点**：兼顾可读性和兼容性  
**缺点**：可能仍有部分中文无法显示

---

## 推荐方案

**当前采用的全英文方案是最佳选择**，因为：

1. ✅ **跨平台兼容**：不依赖特定字体
2. ✅ **国际化**：符合学术论文标准
3. ✅ **无警告**：完全解决字体缺失问题
4. ✅ **简洁清晰**：英文标签更简洁专业

---

## 影响范围

### 已修改

- ✅ `src/core/simulation_stats.py` - `plot_feedback_scores()` 方法

### 未修改（可能仍有中文）

如果其他绘图函数也出现中文警告，可以按相同方式修改：

- `viz/plotter.py` - 各种可视化函数
- `viz/duration_aware_plotter.py` - 时长感知可视化
- 其他自定义绘图代码

---

## 快速检查方法

运行以下命令检查是否还有中文标签：

```bash
# 检查 simulation_stats.py
grep -n "[\u4e00-\u9fa5]" src/core/simulation_stats.py

# 检查所有 Python 文件中的中文
find src -name "*.py" -exec grep -l "[\u4e00-\u9fa5]" {} \;
```

---

## 更新日期

2025-01-09

---

## 总结

通过将所有图表标签改为英文，彻底解决了 matplotlib 中文字体缺失警告问题，提升了代码的国际化水平和跨平台兼容性。

