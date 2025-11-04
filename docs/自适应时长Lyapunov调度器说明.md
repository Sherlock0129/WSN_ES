# 自适应传输时长Lyapunov调度器

## 功能概述

**`AdaptiveDurationLyapunovScheduler`** 是一个基于Lyapunov优化理论的能量传输调度器，能够**自适应选择传输时长**（1-5分钟），实现纯粹的能量优化。

### 核心特点

✅ **纯粹的Lyapunov能量优化** - 不考虑AoI和信息量，专注于能量传输效率  
✅ **自适应时长选择** - 根据路径效率和能量缺口智能选择1-5分钟的传输时长  
✅ **简单高效** - 得分函数清晰直观：`delivered × Q[r] - V × loss`  

## 设计原理

### Lyapunov优化框架

对于每个donor-receiver对，尝试不同的传输时长（1, 2, 3, 4, 5分钟），计算Lyapunov得分：

```
得分 = delivered × Q[r] - V × loss
```

其中：
- **`delivered`**: receiver实际接收的能量 = `duration × E_char × η`
- **`Q[r]`**: receiver的能量缺口 = `E_bar - E[r]`
- **`V`**: Lyapunov控制参数（默认0.5）
- **`loss`**: 传输损耗 = `duration × E_char - delivered`

### 时长选择逻辑

| 场景 | 传输时长选择 | 原因 |
|------|------------|------|
| **高效率路径** (η > 40%) | 选择**更长时长** (5分钟) | 损耗小，送达能量多，得分高 |
| **低效率路径** (η < 20%) | 选择**更短时长** (1分钟) | 损耗大，长时间传输得不偿失 |
| **高能量缺口** (Q[r]大) | 倾向**更长时长** | Q[r]大，delivered项权重高 |
| **低能量缺口** (Q[r]小) | 倾向**更短时长** | Q[r]小，更注重减少损耗 |

## 实现代码

**位置**: `src/scheduling/schedulers.py`

```python
class AdaptiveDurationLyapunovScheduler(BaseScheduler):
    """
    自适应传输时长的Lyapunov调度器
    
    得分函数：delivered × Q[r] - V × loss
    """
    def __init__(self, node_info_manager, V=0.5, K=2, max_hops=5,
                 min_duration=1, max_duration=5):
        # 初始化参数
        ...
    
    def _compute_duration_score(self, donor, receiver, path, eta, E_bar, Q_r, duration):
        """计算特定时长的Lyapunov得分"""
        E_char = getattr(donor, "E_char", 300.0)
        
        # 能量计算
        energy_sent_total = duration * E_char
        energy_delivered = energy_sent_total * eta
        energy_loss = energy_sent_total - energy_delivered
        
        # Lyapunov得分
        Q_normalized = Q_r / E_bar if E_bar > 0 else 0
        score = energy_delivered * Q_normalized - self.V * energy_loss
        
        return score, energy_delivered, energy_loss
    
    def plan(self, network, t):
        """规划能量传输，自适应选择传输时长"""
        for each donor-receiver pair:
            # 尝试不同时长
            for duration in [1, 2, 3, 4, 5]:
                score = self._compute_duration_score(...)
            
            # 选择得分最高的时长
            best_duration = argmax(score)
```

## 使用方法

### 1. 创建调度器

```python
from scheduling.schedulers import AdaptiveDurationLyapunovScheduler
from info_collection.physical_center import NodeInfoManager

# 创建节点信息管理器
nim = NodeInfoManager(initial_position=(5.0, 5.0))
nim.initialize_node_info(network.nodes, initial_time=0)

# 创建自适应时长Lyapunov调度器
scheduler = AdaptiveDurationLyapunovScheduler(
    node_info_manager=nim,
    V=0.5,  # Lyapunov控制参数
    K=2,  # 每个receiver最多接受的donor数量
    max_hops=5,  # 最大跳数
    min_duration=1,  # 最小传输时长（分钟）
    max_duration=5  # 最大传输时长（分钟）
)
```

### 2. 运行仿真

```python
from core.energy_simulation import EnergySimulation

simulation = EnergySimulation(
    network=network,
    time_steps=1000,
    scheduler=scheduler,
    enable_energy_sharing=True
)

simulation.simulate()
```

### 3. 在配置中使用

```python
from config.simulation_config import ConfigManager

config = ConfigManager()

# 调度器参数
config.scheduler_config.lyapunov_V = 0.5
config.scheduler_config.K = 2

# 传输时长参数
config.scheduler_config.duration_min = 1
config.scheduler_config.duration_max = 5
```

## 测试结果

运行 `test_adaptive_duration_lyapunov.py` 的测试结果：

### 自适应时长Lyapunov调度器
- 平均传输时长：**2.0分钟**
- 传输时长分布：**{1分钟: 6次, 5分钟: 2次}**
- 总传输能量：**8000J**
- 总送达能量：**2458J**
- **平均效率：30.72%**

### 标准Lyapunov调度器（对比）
- 平均传输时长：**1.0分钟（固定）**
- 总传输能量：**4000J**
- 总送达能量：**786J**
- **平均效率：19.64%**

### 关键发现

1. ✅ **效率提升 56%**：从19.64%提升到30.72%
2. ✅ **智能时长选择**：高效路径选5分钟，低效路径选1分钟
3. ✅ **送达能量增加 213%**：从786J增加到2458J
4. ✅ **传输能量翻倍**：从4000J增加到8000J

### 实际案例分析

| Donor→Receiver | 路径效率 | 选择时长 | Lyapunov得分 | 说明 |
|---------------|---------|---------|-------------|------|
| D14→R2 | **42.92%** | **5分钟** | **+181.91** | 高效率→长时长 |
| D11→R3 | 12.51% | 1分钟 | -66.06 | 低效率→短时长 |
| D12→R6 | 7.00% | 1分钟 | -103.04 | 极低效率→短时长 |

## 参数调优

### Lyapunov参数 V

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.3` | 更注重能量收益，倾向长时长 | 能量需求大 |
| `0.5` | **推荐值**，平衡收益和损耗 | 通用场景 |
| `0.8` | 更注重减少损耗，倾向短时长 | 追求效率 |

### 时长范围

| 范围 | 效果 | 适用场景 |
|------|------|---------|
| `1-3分钟` | 保守策略 | 网络变化快 |
| `1-5分钟` | **推荐值** | 通用场景 |
| `1-10分钟` | 激进策略 | 稳定网络 |

## 与标准Lyapunov的对比

| 特性 | 标准Lyapunov | 自适应时长Lyapunov |
|------|------------|------------------|
| 传输时长 | 固定1分钟 | 自适应1-5分钟 |
| 能量效率 | 19.64% | **30.72%** ↑56% |
| 送达能量 | 786J | **2458J** ↑213% |
| 计算复杂度 | O(N) | O(N×D) (D=时长选项) |
| 适用场景 | 快速响应 | 效率优先 |

## 优势与局限

### 优势

1. ✅ **效率大幅提升**：平均效率提升56%
2. ✅ **智能决策**：根据路径效率自适应选择时长
3. ✅ **简单直观**：纯粹的Lyapunov优化，无额外参数
4. ✅ **理论保证**：基于Lyapunov稳定性理论

### 局限

1. ⚠️ **计算开销增加**：需要遍历5种时长选项（5倍计算量）
2. ⚠️ **可能增加总能耗**：为了效率可能选择更长时长
3. ⚠️ **不考虑实时性**：专注能量优化，不关注AoI

## 与其他调度器的比较

| 调度器 | 时长选择 | 优化目标 | 复杂度 | 适用场景 |
|--------|---------|---------|--------|---------|
| **标准Lyapunov** | 固定1分钟 | 能量均衡 | 简单 | 通用 |
| **自适应时长Lyapunov** | 自适应1-5分钟 | 能量均衡+效率 | 中等 | **效率优先** |
| **DurationAwareLyapunov** | 自适应1-5分钟 | 能量+AoI+信息 | 复杂 | 多目标优化 |
| **PowerControl** | 固定1分钟 | 目标效率 | 简单 | 效率控制 |

## 相关文件

- **调度器实现**: `src/scheduling/schedulers.py` - `AdaptiveDurationLyapunovScheduler`
- **测试文件**: `test_adaptive_duration_lyapunov.py`
- **配置文件**: `src/config/simulation_config.py`

---

**创建日期**: 2024-11-03  
**功能状态**: ✅ 已实现并测试通过  
**推荐场景**: 追求能量传输效率的场景

