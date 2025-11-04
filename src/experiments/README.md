# Experiments 模块

## 概述

`experiments` 模块提供实验相关的可视化工具，用于分析和展示仿真结果。

## 主要组件

### 1. visualize_results.py

结果可视化工具，提供多种图表生成功能。

**主要功能：**
- 结果数据读取和解析
- 多种图表生成（能量变化、K值变化、传输统计等）
- 结果对比分析

## 文件结构

```
experiments/
└── visualize_results.py    # 结果可视化工具
```

## 使用方法

```python
from experiments.visualize_results import visualize_simulation_results

# 可视化仿真结果
visualize_simulation_results(
    results_file="data/results.csv",
    output_dir="data/plots"
)
```

## 相关文档

- [快速启动指南](../../docs/快速启动指南.md)

