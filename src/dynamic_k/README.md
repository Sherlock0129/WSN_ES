# Dynamic K 模块

## 概述

`dynamic_k` 模块负责K值的动态调整和自适应算法。K值控制每个接收端可同时匹配的捐能者数量上限，直接影响能量传输的并行度和效率。

## 主要组件

### 1. KAdaptationManager (k_adaptation.py)

K值自适应管理器，负责根据仿真统计信息动态调整K值。

**核心功能：**
- K值自适应调整
- 奖励函数计算（归一化）
- 前瞻模拟支持（可选）
- K值历史记录

**自适应算法：**
1. 计算归一化奖励值：
   - `reward = w_b × norm_improve + w_d × norm_delivered - w_l × norm_loss`
   - `norm_improve`: 能量标准差改进（归一化）
   - `norm_delivered`: 总送达能量（归一化）
   - `norm_loss`: 总损耗能量（归一化）

2. 根据奖励值调整K：
   - 如果奖励值增加：增加K（方向+1）
   - 如果奖励值减少：减少K（方向-1）
   - 使用滞回阈值避免频繁调整

3. K值限制：
   - 最小值：1
   - 最大值：`K_max`（默认24）

**关键参数：**
- `initial_K`: 初始K值
- `K_max`: K值上限
- `hysteresis`: 滞回阈值（防止频繁调整）
- `w_b`: 均衡改进权重
- `w_d`: 有效送达量权重
- `w_l`: 损耗惩罚权重
- `use_lookahead`: 是否使用前瞻模拟

**使用方法：**
```python
from dynamic_k.k_adaptation import KAdaptationManager

k_manager = KAdaptationManager(
    initial_K=1,
    K_max=24,
    hysteresis=0.2,
    w_b=0.8,
    w_d=0.8,
    w_l=1.5,
    use_lookahead=False
)

# 在仿真步骤后调用
stats = {
    "pre_std": 1000.0,
    "post_std": 800.0,
    "delivered_total": 5000.0,
    "total_loss": 1000.0
}
k_manager.adapt_K(stats, network=network, scheduler=scheduler)

# 获取当前K值
current_k = k_manager.K
```

### 2. Lookahead (lookahead.py)

前瞻模拟模块，用于短期评估不同K值的效果。

**核心功能：**
- 短期前瞻模拟（模拟未来几步）
- K值效果评估
- 辅助K值调整决策

**主要函数：**
- `pick_k_via_lookahead()`: 通过前瞻模拟选择最优K值
- `_eval_one_candidate()`: 评估单个K值候选
- `_compute_stats_for_network()`: 计算网络统计信息

**工作原理：**
1. **生成候选K值**：
   - 根据当前改进方向和滞回阈值生成候选列表
   - 候选包括：`[current_K, current_K ± d, current_K ± 2d, ...]`
   
2. **前瞻模拟评估**：
   - 对每个候选K值进行短期模拟（horizon_minutes分钟）
   - 先推进能量演化（不做传能）
   - 然后进行一次传能评估
   - 计算预期奖励值

3. **选择最优K值**：
   - 比较所有候选K值的预期奖励
   - 选择预期奖励最高的K值

**关键参数：**
- `horizon_minutes`: 前瞻窗口（分钟，默认60）
- `reward_fn`: 奖励函数（与KAdaptationManager的奖励函数一致）

**注意：** 前瞻模拟会增加计算开销（需要多次网络拷贝和模拟），通常只在计算资源充足时启用。对于大型网络，建议关闭前瞻模拟以提高性能。

## 文件结构

```
dynamic_k/
├── __pycache__/
├── k_adaptation.py    # K值自适应管理器
└── lookahead.py       # 前瞻模拟模块
```

## K值调整策略

### 奖励函数设计

奖励函数综合考虑三个因素：
1. **能量均衡改进** (`w_b × norm_improve`): 鼓励降低能量标准差
2. **有效送达量** (`w_d × norm_delivered`): 鼓励提高总送达能量
3. **损耗惩罚** (`-w_l × norm_loss`): 抑制传输损耗

### 调整逻辑

```
if reward > last_reward:
    direction = +1  # 增加K
else:
    direction = -1  # 减少K

if abs(reward - last_reward) > hysteresis:
    K = max(1, min(K_max, K + direction))
```

### 滞回机制

滞回阈值 `hysteresis` 用于防止K值频繁震荡：
- 只有当奖励值变化超过阈值时，才调整K值
- 避免因统计噪声导致的频繁调整

## 配置参数

K值自适应相关配置位于 `SimulationConfig`：

```python
enable_k_adaptation: bool = False     # 是否启用K值自适应
initial_K: int = 1                     # 初始K值
K_max: int = 24                        # K值上限
hysteresis: float = 0.2                # 滞回阈值
w_b: float = 0.8                       # 均衡改进权重
w_d: float = 0.8                       # 有效送达量权重
w_l: float = 1.5                       # 损耗惩罚权重
use_lookahead: bool = False            # 是否使用前瞻模拟
fixed_k: int = 1                       # 固定K值（当不使用自适应时）
```

## 使用场景

### 场景1：固定K值

```python
simulation = EnergySimulation(
    network=network,
    enable_k_adaptation=False,
    fixed_k=3
)
```

### 场景2：自适应K值

```python
simulation = EnergySimulation(
    network=network,
    enable_k_adaptation=True,
    initial_K=1,
    K_max=24,
    hysteresis=0.2,
    w_b=0.8,
    w_d=0.8,
    w_l=1.5,
    use_lookahead=False
)
```

## 性能考虑

- **计算开销**：K值自适应需要计算奖励函数，开销较小
- **前瞻模拟开销**：如果启用 `use_lookahead`，需要额外计算，开销较大
- **调整频率**：通过滞回阈值控制调整频率，避免过度计算

## 相关文档

- [关闭动态K配置说明](../../关闭动态K配置说明.md)
- [固定K快速参考](../../固定K快速参考.txt)

