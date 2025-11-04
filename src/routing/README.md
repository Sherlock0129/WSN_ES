# Routing 模块

## 概述

`routing` 模块提供能量传输路径查找算法，包括EEOR、EETOR和机会路由等算法。

## 主要组件

### 1. EETOR (energy_transfer_routing.py)

Energy-Efficient Transfer Opportunistic Routing，专门为能量传输设计的路由算法。

**核心功能：**
- 基于能量传输效率的路由选择
- 能量状态感知路由
- 信息感知路由（可选）
- 自适应邻居构建
- 多跳路径查找

**关键特性：**
1. **能量传输效率模型**：
   - 使用能量传输效率（而非误码率）
   - 效率公式：`η(d) = η_0 / d^γ`（距离>1m）
   - 路径总效率：累积乘积 `η_path = ∏ η(hop)`

2. **能量状态感知**：
   - 低能量节点代价惩罚
   - 太阳能节点奖励
   - 综合考虑能量状态和传输效率

3. **信息感知路由**（可选）：
   - 优先选择信息量大的节点
   - 鼓励信息搭便车

4. **自适应邻居构建**：
   - 动态调整通信范围
   - 目标邻居数控制
   - 密集/稀疏网络自适应

**主要函数：**
- `eetor_find_path_adaptive()`: 自适应EETOR路径查找
- `calculate_energy_transfer_efficiency()`: 计算能量传输效率
- `calculate_path_efficiency()`: 计算路径总效率
- `build_neighbors_adaptive()`: 自适应邻居构建

**使用方法：**
```python
from routing.energy_transfer_routing import eetor_find_path_adaptive

# 查找路径
path = eetor_find_path_adaptive(
    source_node=donor,
    dest_node=receiver,
    nodes=all_nodes,
    max_hops=5,
    config=eetor_config
)
```

### 2. EEOR (EEOR.py)

Expected Energy-Efficient Opportunistic Routing，基于期望能量效率的机会路由算法。

**核心功能：**
- 基于期望代价的路由选择
- 多候选转发节点（Forwarding Set）
- 误码率模型
- 能量和距离权衡

**关键特性：**
- 使用期望代价函数选择最优路径
- 考虑链路误码率
- 支持多候选转发节点

**主要函数：**
- `_expected_cost_given_fwd()`: 计算给定转发集的期望代价
- `_build_neighbors()`: 构建邻居关系
- `_link_error_prob()`: 计算链路误码率

### 3. OpportunisticRouting (opportunistic_routing.py)

机会路由算法，基于能量和距离的启发式路由（OECR: Opportunistic Energy Cooperative Routing）。

**核心功能：**
- 基于能量和距离的路径选择
- 多跳路径查找
- 能量和距离权衡
- 考虑能量生成能力（太阳能）

**算法特点：**
- 综合评分：能量得分 + 距离得分 + 能量生成得分
- 贪心选择：每跳选择评分最高的下一跳节点
- 最大跳数限制：防止无限循环

**主要函数：**
- `opportunistic_routing(nodes, source_node, destination_node, max_hops, t, receive_WET)`: 
  - 机会路由主函数
  - `nodes`: 所有节点列表
  - `source_node`: 源节点
  - `destination_node`: 目标节点
  - `max_hops`: 最大跳数
  - `t`: 当前时间步（用于能量生成计算）
  - `receive_WET`: 接收到的WET能量（用于能量生成计算）

**评分公式：**
```python
score = 0.5 * energy_score + 0.3 * distance_score + 0.2 * energy_generation_score
```
- `energy_score`: 节点能量归一化（能量/容量）
- `distance_score`: 距离倒数（1/(距离+1)）
- `energy_generation_score`: 能量生成能力归一化（太阳能相关）

## 文件结构

```
routing/
├── __pycache__/
├── EEOR.py                      # EEOR路由算法
├── energy_transfer_routing.py  # EETOR路由算法
└── opportunistic_routing.py    # 机会路由算法
```

## 路由算法对比

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **EETOR** | 能量传输效率优先，能量状态感知，信息感知 | 能量传输优化，信息感知路由 |
| **EEOR** | 期望代价优化，多候选转发 | 可靠性要求高的场景 |
| **OpportunisticRouting** | 简单启发式，能量+距离 | 计算资源受限的场景 |

## EETOR配置参数

EETOR相关配置位于 `EETORConfig`：

```python
# 能量传输效率模型参数
eta_0: float = 0.6              # 1米处的参考效率
gamma: float = 2.0              # 距离衰减因子

# 邻居构建参数
max_range: float = 10.0         # 最大通信范围
min_efficiency: float = 0.01     # 最小传输效率阈值

# 能量状态感知参数
enable_energy_state_aware: bool = True
low_energy_threshold: float = 0.2
low_energy_penalty: float = 1.5
solar_bonus: float = 0.9

# 信息感知路由参数
enable_info_aware_routing: bool = True
info_reward_factor: float = 0.2
```

## 使用场景

### 场景1：基本路径查找

```python
from routing.energy_transfer_routing import eetor_find_path_adaptive

path = eetor_find_path_adaptive(
    source_node=donor,
    dest_node=receiver,
    nodes=network.nodes,
    max_hops=5
)
```

### 场景2：使用配置

```python
from config.simulation_config import get_config

config_manager = get_config()
eetor_config = config_manager.eetor_config

path = eetor_find_path_adaptive(
    source_node=donor,
    dest_node=receiver,
    nodes=network.nodes,
    max_hops=5,
    config=eetor_config
)
```

## 相关文档

- [EEOR路由算法详细逻辑说明](../../docs/EEOR路由算法详细逻辑说明.md)
- [EETOR路由算法逻辑说明](../../docs/EETOR路由算法逻辑说明.md)
- [信息感知路由验证报告](../../信息感知路由验证报告.md)

