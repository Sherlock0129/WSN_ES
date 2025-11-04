# Viz 模块

## 概述

`viz` 模块提供数据可视化功能，用于生成各种图表和可视化结果。

## 主要组件

### 1. Plotter (plotter.py)

绘图工具类，提供多种图表生成功能。

**核心功能：**
- 节点分布图
- 能量随时间变化图
- 能量传输路径图
- K值变化图
- 中心节点能量图

**关键特性：**
- 使用Plotly进行交互式可视化
- 支持IEEE风格图表
- 支持多种图表类型
- 自动保存图表文件

**主要函数：**
- `plot_node_distribution()`: 绘制节点分布图
- `plot_energy_over_time()`: 绘制能量随时间变化图
- `plot_energy_paths()`: 绘制能量传输路径图
- `plot_k_values()`: 绘制K值变化图
- `plot_center_node_energy()`: 绘制中心节点能量图

**使用方法：**
```python
from viz.plotter import (
    plot_node_distribution,
    plot_energy_over_time,
    plot_energy_paths,
    plot_k_values
)

# 绘制节点分布
plot_node_distribution(
    nodes=network.nodes,
    output_dir="data",
    session_dir=session_dir
)

# 绘制能量随时间变化
plot_energy_over_time(
    stats=simulation_stats,
    output_dir="data",
    session_dir=session_dir
)

# 绘制能量传输路径
plot_energy_paths(
    network=network,
    plans=energy_plans,
    time_step=current_time,
    output_path="data/paths.png"
)

# 绘制K值变化
plot_k_values(
    stats=simulation_stats,
    output_dir="data",
    session_dir=session_dir
)
```

## 文件结构

```
viz/
├── __pycache__/
└── plotter.py    # 绘图工具类
```

## 图表类型

### 1. 节点分布图

展示网络中所有节点的位置分布，区分：
- 太阳能节点（黄色圆点）
- 非太阳能节点（灰色三角）
- 移动节点路径（可选）

### 2. 能量随时间变化图

展示网络能量随时间的变化趋势：
- 平均能量
- 能量标准差
- 总传输能量
- 总损耗能量

### 3. 能量传输路径图

展示特定时间步的能量传输路径：
- 节点位置
- 传输路径（连线）
- 传输方向（箭头）

### 4. K值变化图

展示K值随时间的自适应变化：
- K值历史
- 调整原因（可选）

### 5. 中心节点能量图

展示物理中心节点（ID=0）的能量变化：
- 能量历史
- 能量趋势

## 图表样式

- **IEEE风格**：符合学术论文要求
- **交互式**：使用Plotly，支持缩放、平移、悬停查看
- **高质量**：支持高分辨率导出

## 使用示例

### 示例1：基本可视化

```python
from viz.plotter import plot_node_distribution, plot_energy_over_time

# 绘制节点分布
plot_node_distribution(
    nodes=network.nodes,
    output_dir="data",
    session_dir=session_dir
)

# 绘制能量变化
plot_energy_over_time(
    stats=simulation.stats.stats_list,
    output_dir="data",
    session_dir=session_dir
)
```

### 示例2：完整可视化流程

```python
from viz.plotter import (
    plot_node_distribution,
    plot_energy_over_time,
    plot_k_values,
    plot_center_node_energy
)

# 1. 节点分布
plot_node_distribution(nodes, session_dir=session_dir)

# 2. 能量变化
plot_energy_over_time(stats, session_dir=session_dir)

# 3. K值变化
plot_k_values(stats, session_dir=session_dir)

# 4. 中心节点能量
plot_center_node_energy(network, session_dir=session_dir)
```

## 相关文档

- [快速启动指南](../../docs/快速启动指南.md)

