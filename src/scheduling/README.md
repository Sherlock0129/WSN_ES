# Scheduling 模块

## 概述

`scheduling` 模块提供能量传输调度算法，包括多种调度策略和深度学习调度器。

## 主要组件

### 1. BaseScheduler (schedulers.py)

调度器基类，提供通用的调度器接口和基础功能。

**核心功能：**
- 统一的调度器接口
- 节点过滤（排除物理中心节点、锁定节点）
- 与路由算法集成

**关键方法：**
- `plan()`: 制定能量传输计划（抽象方法，必须实现）
- `post_step()`: 步骤后处理（可选，默认空实现）
- `get_name()`: 获取调度器名称（返回类名）
- `_filter_regular_nodes()`: 过滤普通节点（排除物理中心节点）
- `_filter_unlocked_nodes()`: 过滤未锁定节点（排除正在传输的节点）

**节点过滤机制：**
- **物理中心节点过滤**：物理中心节点（ID=0）不参与WET，包括：
  - 不能作为donor（捐能者）
  - 不能作为receiver（接收者）
  - 不能作为relay（中继节点）
  
- **节点锁定机制**：当传输时长>1分钟时，参与传输的所有节点会被锁定：
  - 锁定时间 = 当前时间 + 传输时长
  - 锁定期间，节点无法参与新的能量传输
  - 锁定到期后自动解锁

### 2. LyapunovScheduler (schedulers.py)

基于Lyapunov优化的调度器。

**核心功能：**
- 使用Lyapunov函数优化能量均衡
- 排队长度驱动的机会捐能匹配
- 能量差驱动的匹配策略

**关键参数：**
- `V`: Lyapunov控制强度（越大越保守/稳定）
- `K`: 最大捐能者数量
- `max_hops`: 最大跳数

### 3. DurationAwareLyapunovScheduler (schedulers.py)

传输时长优化的Lyapunov调度器。

**核心功能：**
- 将传输时长作为优化维度
- 综合考虑能量传输量、AoI变化、信息量累积
- 节点锁定机制（传输时长>1时启用）

**关键特性：**
- **传输时长优化**：为每个donor-receiver对选择最优传输时长（duration_min到duration_max）
- **节点锁定机制**：当传输时长>1分钟时，参与传输的所有节点会被锁定
  - 锁定时间 = 当前时间 + 传输时长
  - 锁定期间，节点无法参与新的能量传输
  - 锁定到期后自动解锁

**关键参数：**
- `min_duration`: 最小传输时长（分钟）
- `max_duration`: 最大传输时长（分钟）
- `w_aoi`: AoI惩罚权重
- `w_info`: 信息量奖励权重
- `info_collection_rate`: 信息采集速率（bits/分钟）

### 4. ClusterScheduler (schedulers.py)

基于聚类的调度器（类似LEACH）。

**核心功能：**
- 轮换簇首机制
- 簇内能量均衡
- 簇头选择概率控制

**关键参数：**
- `round_period`: 轮换簇首周期（分钟）
- `p_ch`: 成为簇头的期望概率

### 5. PredictionScheduler (schedulers.py)

基于预测的调度器。

**核心功能：**
- 基于能量趋势的预测
- 滑动平均估计
- 前瞻调度

**关键参数：**
- `alpha`: 指数平滑系数
- `horizon_min`: 预测窗口（分钟）

### 6. PowerControlScheduler (schedulers.py)

功率控制调度器。

**核心功能：**
- 以达成目标效率为导向的功率控制
- 根据路径效率反推送能
- 能量发送量自适应调整

**关键参数：**
- `target_eta`: 目标传输效率（0~1）

### 7. BaselineHeuristic (schedulers.py)

基线启发式调度器。

**核心功能：**
- 简单的能量均衡策略
- 低能量节点优先匹配
- 高能量节点优先捐能

### 8. DQNScheduler (dqn_scheduler.py)

DQN深度强化学习调度器（离散动作空间）。

**核心功能：**
- 将能量共享建模为MDP
- 离散动作空间：传输时长（1-10分钟）
- 经验回放和Q网络学习

**关键特性：**
- **状态空间**：节点能量、位置、距离等
- **动作空间**：10个离散动作（1-10分钟）
- **奖励函数**：能量均衡、网络存活、传输效率

**优势：**
- 离散动作更容易训练
- 计算效率更高
- 收敛更快更稳定

**使用方法：**
```python
from scheduling.dqn_scheduler import DQNScheduler

scheduler = DQNScheduler(
    node_info_manager=node_info_manager,
    K=3,
    max_hops=5,
    action_dim=10,
    training_mode=False  # 使用已训练模型
)
```

### 9. DDPGScheduler (ddpg_scheduler.py)

DDPG深度强化学习调度器（连续动作空间）。

**核心功能：**
- 将能量共享建模为MDP
- 连续动作空间：传输时长（1.0-5.0分钟）
- Actor-Critic架构

**关键特性：**
- **状态空间**：节点能量、位置、距离等
- **动作空间**：连续值（1.0-5.0分钟）
- **Actor网络**：策略网络，输出动作
- **Critic网络**：价值网络，评估Q值

**使用方法：**
```python
from scheduling.ddpg_scheduler import DDPGScheduler

scheduler = DDPGScheduler(
    node_info_manager=node_info_manager,
    K=3,
    max_hops=5,
    training_mode=False  # 使用已训练模型
)
```

### 10. InfoNode (info_node.py)

信息节点类，基于物理中心节点信息表的轻量级节点。

**核心功能：**
- 提供与SensorNode相同的接口
- 用于调度器和路由算法
- 所有计算公式与SensorNode完全一致

**关键特性：**
- 只包含调度和路由需要的属性和方法
- 所有计算公式与SensorNode完全一致（距离、效率、采集等）
- 数据来源于物理中心的节点信息表
- 支持物理中心节点标识

**使用方法：**
```python
from info_collection.info_node import InfoNode

# InfoNode由NodeInfoManager自动创建和管理
# 调度器从NodeInfoManager获取InfoNode列表
info_nodes = node_info_manager.get_info_nodes()
```

### 11. PassiveTransferManager (passive_transfer.py)

智能被动传能管理器。

**核心功能：**
- 综合决策是否触发能量传输
- 检查间隔控制
- 冷却期机制
- 低能量节点比例检测
- 能量分布方差检测
- 预测性分析

**关键特性：**
- **检查间隔**：只在指定间隔检查
- **冷却期**：避免过于频繁的传能
- **低能量比例**：检测低能量节点比例
- **能量方差**：检测能量分布方差
- **预测性触发**：基于能量趋势预测

**使用方法：**
```python
from scheduling.passive_transfer import PassiveTransferManager

passive_manager = PassiveTransferManager(
    passive_mode=True,
    check_interval=10,
    critical_ratio=0.2,
    energy_variance_threshold=0.3,
    cooldown_period=30,
    predictive_window=60
)

# 在仿真步骤中调用
should_trigger, reason = passive_manager.should_trigger_transfer(
    t=current_time,
    network=network
)
```

## 文件结构

```
scheduling/
├── __pycache__/
├── ddpg_scheduler.py    # DDPG深度强化学习调度器
├── dqn_scheduler.py      # DQN深度强化学习调度器
├── passive_transfer.py  # 智能被动传能管理器
└── schedulers.py        # 基础调度器和传统调度算法
```

## 调度器对比

| 调度器 | 特点 | 适用场景 |
|--------|------|----------|
| **LyapunovScheduler** | 能量均衡优化 | 通用场景 |
| **DurationAwareLyapunovScheduler** | 传输时长优化，节点锁定 | 需要优化传输时长的场景 |
| **ClusterScheduler** | 聚类轮换 | 大规模网络 |
| **PredictionScheduler** | 预测性调度 | 能量趋势可预测的场景 |
| **PowerControlScheduler** | 功率控制 | 需要精确控制传输效率的场景 |
| **DQNScheduler** | 深度学习，离散动作 | 需要自适应学习的场景 |
| **DDPGScheduler** | 深度学习，连续动作 | 需要精确控制传输时长的场景 |

## 调度器选择

### 场景1：基本能量均衡

```python
from scheduling.schedulers import LyapunovScheduler

scheduler = LyapunovScheduler(
    node_info_manager=node_info_manager,
    V=0.5,
    K=3,
    max_hops=5
)
```

### 场景2：传输时长优化

```python
from scheduling.schedulers import DurationAwareLyapunovScheduler

scheduler = DurationAwareLyapunovScheduler(
    node_info_manager=node_info_manager,
    V=0.5,
    K=3,
    max_hops=5,
    min_duration=1,
    max_duration=5,
    w_aoi=0.1,
    w_info=0.05
)
```

### 场景3：深度学习调度器

```python
from scheduling.dqn_scheduler import DQNScheduler

scheduler = DQNScheduler(
    node_info_manager=node_info_manager,
    K=3,
    max_hops=5,
    action_dim=10,
    training_mode=False  # 使用已训练模型
)
```

## 相关文档

- [DQN离散动作调度器说明](../../DQN离散动作调度器说明.md)
- [DDPG深度强化学习调度器说明](../../DDPG深度强化学习调度器说明.md)
- [自适应时长Lyapunov调度器说明](../../自适应时长Lyapunov调度器说明.md)
- [传输时长优化功能说明](../../传输时长优化功能说明.md)
- [智能被动传能系统说明](../../docs/智能被动传能系统说明.md)

