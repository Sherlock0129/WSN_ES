# AdaptiveDurationAwareLyapunovScheduler

## 概述

`AdaptiveDurationAwareLyapunovScheduler` 是一个高级调度器，结合了自适应参数调整和传输时长优化的双重优势。它继承自 `DurationAwareLyapunovScheduler`，并融合了 `AdaptiveLyapunovScheduler` 的自适应机制。

## 核心创新

### 1. 双重优化维度

#### 传输时长优化（继承自 DurationAwareLyapunovScheduler）
- **能量传输量优化**: 对每个donor-receiver对，尝试不同传输时长（1-5分钟），选择最优时长
- **AoI感知**: 考虑传输期间AoI增长的影响
- **信息量累积**: 利用传输时间累积更多信息，实现信息搭便车
- **多目标综合得分**: 能量收益 - 能量损耗 - AoI惩罚 + 信息奖励

#### 自适应参数调整（继承自 AdaptiveLyapunovScheduler）
- **V参数动态调整**: 根据网络反馈自动调整Lyapunov控制参数V
- **多维度反馈**: 基于能量均衡、传输效率、节点存活率、总能量等4维指标
- **智能诊断**: 自动诊断网络问题（效率低、均衡差、节点死亡）并采取相应策略
- **平滑调整**: 带记忆的滑动窗口机制，避免参数震荡

### 2. 节点锁定机制

当传输时长 > 1分钟时，自动启用节点锁定机制：
- 参与传输路径的所有节点（donor、receiver、中继节点）会被锁定
- 锁定时间 = 当前时间 + 传输时长
- 在锁定期间，这些节点无法参与新的能量传输
- 避免节点过度使用，保护节点能量平衡

## 工作原理

### 综合得分计算

对于每个donor-receiver对和每个候选传输时长，计算综合得分：

```python
score = energy_benefit - energy_loss_penalty - aoi_penalty + info_bonus

其中：
- energy_benefit = delivered × Q_normalized
- energy_loss_penalty = V × loss
- aoi_penalty = w_aoi × duration × Q_normalized
- info_bonus = w_info × (duration × info_collection_rate) × info_factor
```

### 自适应策略

调度器根据反馈自动调整V参数：

| 问题类型 | 诊断条件 | 调整策略 | 效果 |
|---------|---------|---------|------|
| 效率低 | efficiency_score < -2.0 | 增大V | 更重视损耗，选择更近路径 |
| 均衡差 | balance_score < -2.0 | 减小V | 更重视均衡，增加传输量 |
| 节点死亡 | survival_score < -1.0 | 减小V（1.5倍速率）| 优先救活节点 |
| 趋势恶化 | recent_trend下降 | 轻微增大V | 预防性调整 |
| 表现优秀 | avg_feedback > threshold | 轻微减小V | 优化吞吐量 |
| 长期不佳 | 连续5次负反馈 | 重置到初始值 | 跳出局部最优 |

## 参数配置

### 初始化参数

```python
scheduler = AdaptiveDurationAwareLyapunovScheduler(
    node_info_manager=node_info_manager,
    
    # 基本参数
    V=0.5,                          # 初始V参数（会自动调整）
    K=3,                            # 每个receiver最多的donor数
    max_hops=3,                     # 最大跳数
    
    # 传输时长优化参数
    min_duration=1,                 # 最小传输时长（分钟）
    max_duration=5,                 # 最大传输时长（分钟）
    w_aoi=0.02,                    # AoI惩罚权重（较小）
    w_info=0.1,                    # 信息量奖励权重（较大）
    info_collection_rate=10000.0,  # 信息采集速率（bits/分钟）
    
    # 自适应参数调整参数
    window_size=10,                # 反馈窗口大小
    V_min=0.1,                     # V的最小值
    V_max=2.0,                     # V的最大值
    adjust_rate=0.1,               # 参数调整速率（10%增减）
    sensitivity=2.0                # 反馈敏感度（阈值）
)
```

### 参数说明

#### 传输时长优化参数

| 参数 | 默认值 | 范围 | 说明 |
|-----|--------|------|------|
| `min_duration` | 1 | ≥1 | 最小传输时长（分钟），建议≥1 |
| `max_duration` | 5 | 1-10 | 最大传输时长（分钟），影响节点锁定时长 |
| `w_aoi` | 0.02 | 0-1 | AoI惩罚权重，越小越鼓励长传输 |
| `w_info` | 0.1 | 0-1 | 信息量奖励权重，越大越鼓励信息搭便车 |
| `info_collection_rate` | 10000.0 | >0 | 信息采集速率（bits/分钟） |

#### 自适应参数

| 参数 | 默认值 | 范围 | 说明 |
|-----|--------|------|------|
| `V` | 0.5 | 0.1-2.0 | 初始V参数，会自动调整 |
| `V_min` | 0.1 | >0 | V的最小值，不建议<0.05 |
| `V_max` | 2.0 | >V_min | V的最大值，不建议>3.0 |
| `window_size` | 10 | 5-20 | 反馈窗口大小，记忆最近N次 |
| `adjust_rate` | 0.1 | 0.05-0.2 | 参数调整速率，越大响应越快 |
| `sensitivity` | 2.0 | 1.0-3.0 | 反馈敏感度，越小越敏感 |

## 使用方法

### 方法1：使用配置文件

```python
from config.simulation_config import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 设置调度器类型
config.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"

# 配置参数（可选，使用默认值即可）
config.scheduler_config.adaptive_lyapunov_v = 0.5
config.scheduler_config.adaptive_lyapunov_k = 3
config.scheduler_config.duration_min = 1
config.scheduler_config.duration_max = 5
config.scheduler_config.duration_w_aoi = 0.02
config.scheduler_config.duration_w_info = 0.1

# 保存配置
config.save_to_json("my_config.json")

# 运行仿真
from sim.refactored_main import run_simulation
run_simulation(config_file="my_config.json")
```

### 方法2：直接实例化

```python
from scheduling.schedulers import AdaptiveDurationAwareLyapunovScheduler
from info_collection.physical_center import NodeInfoManager

# 创建节点信息管理器
node_info_manager = NodeInfoManager()
node_info_manager.initialize_node_info(network.nodes, initial_time=0)

# 创建调度器
scheduler = AdaptiveDurationAwareLyapunovScheduler(
    node_info_manager=node_info_manager,
    V=0.5, K=3, max_hops=3,
    min_duration=1, max_duration=5,
    w_aoi=0.02, w_info=0.1,
    window_size=10, V_min=0.1, V_max=2.0
)

# 执行规划
plans, all_candidates = scheduler.plan(network, t=current_time)

# 提供反馈（在每次执行后）
feedback = {
    'total_score': score,
    'details': {
        'balance_score': ...,
        'efficiency_score': ...,
        'survival_score': ...,
        'efficiency': ...
    }
}
scheduler.post_step(network, t=current_time, feedback=feedback)
```

### 方法3：使用示例配置

```python
# 使用提供的示例配置
from config_examples.adaptive_duration_aware_lyapunov_config import create_adaptive_duration_aware_config

config = create_adaptive_duration_aware_config()
config.save_to_json("adaptive_duration_aware_config.json")

# 运行仿真
from sim.refactored_main import run_simulation
run_simulation(config_file="adaptive_duration_aware_config.json")
```

## 输出和监控

### 自适应调整日志

调度器在调整V参数时会输出日志：

```
[自适应@t=1440] V: 0.500 → 0.550 | 效率低(0.35) → 增大V(减少损耗)
           反馈: 总分=-6.20, 均衡=-2.50, 效率=-2.80(η=0.35), 存活=-0.90
```

### 获取统计信息

```python
# 获取自适应统计
stats = scheduler.get_adaptation_stats()
print(f"当前V: {stats['current_V']}")
print(f"调整次数: {stats['total_adjustments']}")
print(f"平均反馈: {stats['avg_feedback']}")

# 打印摘要
scheduler.print_adaptation_summary()
```

输出示例：

```
============================================================
自适应时长感知Lyapunov调度器 - 适应性总结
============================================================
初始V: 0.500
当前V: 0.650
V范围: [0.100, 2.000]
总调整次数: 12
平均反馈分数: -2.35
最佳反馈分数: 5.20
最差反馈分数: -8.50

最近5次调整:
  t=720: 0.500→0.550 | 效率低(0.35) → 增大V(减少损耗)
  t=1440: 0.550→0.495 | 均衡差(std变化8.50) → 减小V(增强均衡)
  t=2160: 0.495→0.545 | 趋势恶化+效率低 → 轻微增大V
  t=2880: 0.545→0.600 | 效率低(0.30) → 增大V(减少损耗)
  t=3600: 0.600→0.650 | 效率低(0.28) → 增大V(减少损耗)
============================================================
```

## 适用场景

### 推荐场景

✅ **长期运行且网络环境动态变化**: 自适应机制可以持续优化参数
✅ **需要平衡能量效率和传输效率**: 双重优化实现最佳权衡
✅ **有明确的信息传输任务**: 信息搭便车机制提高整体效率
✅ **需要保护节点能量平衡**: 节点锁定机制避免过度使用
✅ **对网络性能有较高要求**: 多维度优化确保综合性能

### 不推荐场景

❌ **短期仿真（<24小时）**: 自适应机制需要时间积累反馈
❌ **网络环境静态**: 固定参数的简单调度器可能更高效
❌ **无信息传输需求**: 信息搭便车优势无法发挥
❌ **计算资源受限**: 需要尝试多个传输时长，计算开销较大

## 性能特点

### 优势

1. **自动调优**: 无需人工调参，自动适应网络变化
2. **多目标优化**: 同时考虑能量、AoI、信息量等多个维度
3. **鲁棒性强**: 内置重置机制，避免陷入局部最优
4. **节点保护**: 锁定机制防止节点过度使用
5. **可解释性**: 清晰的调整日志，便于分析和调试

### 劣势

1. **计算开销**: 需要尝试多个传输时长，比简单调度器慢
2. **冷启动**: 需要积累反馈才能发挥最佳效果
3. **参数敏感**: w_aoi和w_info需要根据具体场景调整
4. **锁定限制**: 节点锁定可能降低短期灵活性

## 与其他调度器的比较

| 调度器 | 自适应 | 时长优化 | 计算复杂度 | 适用场景 |
|-------|-------|---------|-----------|---------|
| **LyapunovScheduler** | ❌ | ❌ | 低 | 基线对比 |
| **AdaptiveLyapunovScheduler** | ✅ | ❌ | 中 | 需要自适应 |
| **DurationAwareLyapunovScheduler** | ❌ | ✅ | 中高 | 需要时长优化 |
| **AdaptiveDurationAwareLyapunovScheduler** | ✅ | ✅ | 高 | 高级应用（推荐）|

## 调优建议

### 初次使用

建议使用默认参数：
```python
V=0.5, K=3, max_hops=3,
min_duration=1, max_duration=5,
w_aoi=0.02, w_info=0.1,
window_size=10, V_min=0.1, V_max=2.0
```

### 根据场景调整

| 场景 | 调整建议 |
|-----|---------|
| **能量充裕** | 增大 `max_duration` 到 7-10，增大 `w_info` 到 0.15 |
| **能量紧张** | 减小 `max_duration` 到 3，减小 `adjust_rate` 到 0.05 |
| **AoI敏感** | 增大 `w_aoi` 到 0.05-0.1 |
| **信息为主** | 增大 `w_info` 到 0.15-0.2 |
| **快速响应** | 增大 `adjust_rate` 到 0.15-0.2，减小 `window_size` 到 5-7 |
| **稳定优先** | 减小 `adjust_rate` 到 0.05，增大 `window_size` 到 15-20 |

## 常见问题

### Q1: 为什么自适应机制没有触发？

A: 自适应机制需要积累至少5次反馈才会开始调整。在初始阶段，调度器会先观察网络表现，然后才开始调整参数。

### Q2: 如何判断参数配置是否合理？

A: 观察自适应统计输出：
- 调整次数过多（>50%时间步）→ 减小 `adjust_rate` 或增大 `sensitivity`
- 调整次数过少（<5%时间步）→ 增大 `adjust_rate` 或减小 `sensitivity`
- V频繁到达边界 → 调整 `V_min` 或 `V_max`

### Q3: 节点锁定会影响网络性能吗？

A: 短期内可能降低灵活性，但长期有助于保护节点能量平衡。如果发现性能下降，可以：
- 减小 `max_duration` 减少锁定时间
- 增大 `K` 增加可用donor数量
- 增大 `max_hops` 允许更多路径选择

### Q4: 如何调整 w_aoi 和 w_info？

A: 这两个参数控制AoI和信息量的相对重要性：
- `w_aoi` 太大 → 倾向选择短传输，可能损失能量效率
- `w_aoi` 太小 → 可能选择过长传输，AoI增长过快
- `w_info` 太大 → 过度鼓励长传输，可能损失能量
- `w_info` 太小 → 无法充分利用信息搭便车机制

建议从默认值开始，根据实际效果逐步调整（每次调整±50%）。

## 实现细节

### 继承关系

```
BaseScheduler
    └── DurationAwareLyapunovScheduler
            └── AdaptiveDurationAwareLyapunovScheduler
```

### 核心方法

- `__init__()`: 初始化调度器，设置所有参数
- `plan(network, t)`: 执行规划，返回传输计划和候选信息
- `post_step(network, t, feedback)`: 接收反馈，自适应调整参数
- `get_adaptation_stats()`: 获取自适应统计信息
- `print_adaptation_summary()`: 打印自适应调整摘要

### 继承的方法

从 `DurationAwareLyapunovScheduler` 继承：
- `_path_eta(path)`: 计算路径总效率
- `_compute_duration_score()`: 计算特定传输时长的综合得分
- `_filter_regular_nodes()`: 过滤出普通节点
- `_filter_unlocked_nodes()`: 过滤出未锁定节点

## 参考资料

- [AdaptiveLyapunovScheduler文档](./AdaptiveLyapunovScheduler.md)
- [DurationAwareLyapunovScheduler文档](./传输时长优化功能说明.md)
- [配置示例](../config_examples/adaptive_duration_aware_lyapunov_config.py)
- [测试脚本](../test_adaptive_duration_aware.py)

## 版本历史

- **v1.0** (2025-11-10): 初始版本，结合自适应参数调整和传输时长优化

