# DDPG深度强化学习调度器

## 功能概述

**`DDPGScheduler`** 将WSN节点间能量共享建模为**马尔可夫决策过程（MDP）**，并采用**深度确定性策略梯度（DDPG）**算法求解最优供能策略。

### 核心特点

✅ **端到端学习** - 直接从环境交互中学习最优策略  
✅ **连续动作空间** - 精确控制传输时长（1-5分钟，连续值）  
✅ **多目标优化** - 自动权衡能量均衡、传输效率、网络存活  
✅ **自适应策略** - 根据网络状态动态调整决策  

## 马尔可夫决策过程（MDP）建模

### 1. 状态空间 (State Space)

状态向量 `S_t` 包含：

| 维度 | 描述 | 取值范围 |
|------|------|---------|
| `E_1, ..., E_n` | 各节点归一化能量 | [0, 1] |
| `E_mean` | 平均能量 | [0, 1] |
| `E_std` | 能量标准差 | [0, 1] |
| `E_min` | 最小能量 | [0, 1] |
| `ratio_low` | 低能量节点比例 | [0, 1] |
| `t_norm` | 归一化时间步 | [0, 1] |

**状态维度**: `d_state = n + 5`（n为普通节点数）

### 2. 动作空间 (Action Space)

动作 `A_t` 表示**传输时长**：

```
A_t ∈ [1.0, 5.0] (连续值，单位：分钟)
```

- **连续动作**: 允许精确控制传输时长（如2.5分钟）
- **动作约束**: 通过Sigmoid函数限制在合理范围

### 3. 奖励函数 (Reward Function)

奖励 `R_t` 由多个组成部分构成：

```python
R_t = R_balance + R_efficiency + R_low_energy + R_death + R_survival
```

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| **能量均衡奖励** | `(σ_prev - σ_curr) × 10` | 10.0 | 方差减小越多，奖励越大 |
| **传输效率奖励** | `η × 5` | 5.0 | 效率越高，奖励越大 |
| **低能量惩罚** | `-N_low × 2` | -2.0 | 低能量节点越多，惩罚越大 |
| **死亡惩罚** | `-N_dead × 20` | -20.0 | 节点死亡严重惩罚 |
| **存活奖励** | `1.0 (if N_dead == 0)` | 1.0 | 无节点死亡时奖励 |

### 4. 状态转移 (State Transition)

```
S_{t+1} = f(S_t, A_t, environment)
```

- **确定性部分**: 能量传输的物理模型
- **随机性部分**: 太阳能采集、节点移动等

## DDPG算法架构

### 算法框图

```
┌─────────────────────────────────────────────────────────┐
│                      DDPG Agent                         │
│                                                          │
│  ┌──────────────┐                  ┌──────────────┐    │
│  │    Actor     │  μ(s|θ^μ)       │   Critic     │    │
│  │   Network    ├──────────────────►│   Network    │    │
│  │              │                  │   Q(s,a|θ^Q) │    │
│  └──────┬───────┘                  └───────▲──────┘    │
│         │                                  │            │
│         │ Soft Update (τ)                  │            │
│         ▼                                  │            │
│  ┌──────────────┐                  ┌──────┴──────┐    │
│  │ Target Actor │                  │Target Critic│    │
│  │              │                  │             │    │
│  └──────────────┘                  └─────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────┐        │
│  │         Experience Replay Buffer           │        │
│  │  (s_t, a_t, r_t, s_{t+1}, done)           │        │
│  └────────────────────────────────────────────┘        │
│                                                          │
│  ┌────────────────────────────────────────────┐        │
│  │        Ornstein-Uhlenbeck Noise            │        │
│  │        (for exploration)                    │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. Actor网络（策略网络）

**作用**: 学习策略 `μ(s|θ^μ)`，输出确定性动作

**架构**:
```
Input (state_dim) 
  → FC1 (256, ReLU) 
  → FC2 (256, ReLU) 
  → FC3 (action_dim, Sigmoid) 
  → Output (scaled to [1, 5])
```

**损失函数**:
```
L_actor = -E[Q(s, μ(s))]  # 最大化Q值
```

#### 2. Critic网络（价值网络）

**作用**: 评估状态-动作对的Q值 `Q(s, a|θ^Q)`

**架构**:
```
State Input (state_dim) → FC1 (256, ReLU)
                            ↓
Action Input (action_dim) → Concat → FC2 (256, ReLU) → FC3 (1) → Q-value
```

**损失函数**:
```
L_critic = E[(Q(s, a) - y)²]
where y = r + γ × Q'(s', μ'(s'))
```

#### 3. 经验回放缓冲区 (Replay Buffer)

**作用**: 打破样本相关性，提高训练稳定性

**容量**: 10,000个经验样本

**采样**: 随机批量采样（batch_size=64）

#### 4. OU噪声 (Ornstein-Uhlenbeck Noise)

**作用**: 生成时间相关的探索噪声

**参数**:
- `μ = 0.0` (均值)
- `θ = 0.15` (回归速度)
- `σ = 0.2` (波动强度)

**公式**:
```
dx = θ(μ - x) + σ × N(0,1)
x_{t+1} = x_t + dx
```

### 训练流程

```python
for episode in range(N_episodes):
    # 1. 重置环境
    s_t = env.reset()
    
    for t in range(T_max):
        # 2. 选择动作（添加探索噪声）
        a_t = μ(s_t) + OU_noise
        
        # 3. 执行动作，观察奖励和下一状态
        s_{t+1}, r_t, done = env.step(a_t)
        
        # 4. 存储经验
        replay_buffer.push(s_t, a_t, r_t, s_{t+1}, done)
        
        # 5. 采样批量经验，更新网络
        if len(replay_buffer) > batch_size:
            # 采样
            batch = replay_buffer.sample(batch_size)
            
            # 更新Critic
            y = r + γ × Q'(s', μ'(s'))
            L_critic = MSE(Q(s, a), y)
            
            # 更新Actor
            L_actor = -E[Q(s, μ(s))]
            
            # 软更新目标网络
            θ' ← τθ + (1-τ)θ'
```

## 实现代码

### 文件结构

```
src/scheduling/ddpg_scheduler.py
├── Actor              # Actor网络
├── Critic             # Critic网络
├── ReplayBuffer       # 经验回放缓冲区
├── OUNoise           # OU噪声生成器
├── DDPGAgent         # DDPG智能体
└── DDPGScheduler     # DDPG调度器（集成到仿真）
```

### 核心类：DDPGScheduler

```python
class DDPGScheduler(BaseScheduler):
    """基于DDPG深度强化学习的能量传输调度器"""
    
    def __init__(self, node_info_manager, K=2, max_hops=5,
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.001,
                 training_mode=True):
        # 初始化DDPG智能体
        self.agent = DDPGAgent(...)
    
    def plan(self, network, t):
        """使用DDPG策略生成传输计划"""
        # 1. 计算当前状态
        state = self._compute_state(network, t)
        
        # 2. 使用Actor选择动作（传输时长）
        action = self.agent.select_action(state)
        
        # 3. 生成传输计划
        plans = self._generate_plans(network, action)
        
        # 4. 计算奖励，更新网络（训练模式）
        if self.training_mode:
            reward = self._compute_reward(...)
            self.agent.replay_buffer.push(...)
            self.agent.update()
        
        return plans
```

## 使用方法

### 1. 训练DDPG调度器

```bash
# 训练50回合，每回合100步
python test_ddpg_scheduler.py --mode train --episodes 50 --steps 100
```

或在代码中：

```python
from scheduling.ddpg_scheduler import DDPGScheduler
from acdr.physical_center import NodeInfoManager

# 创建DDPG调度器（训练模式）
scheduler = DDPGScheduler(
    node_info_manager=nim,
    K=2,
    max_hops=3,
    actor_lr=1e-4,      # Actor学习率
    critic_lr=1e-3,     # Critic学习率
    gamma=0.99,         # 折扣因子
    tau=0.001,          # 软更新系数
    training_mode=True  # 训练模式
)

# 训练循环
for episode in range(N_episodes):
    network = create_network()
    simulation = EnergySimulation(network, scheduler=scheduler)
    simulation.simulate()

# 保存模型
scheduler.save_model("ddpg_model.pth")
```

### 2. 测试DDPG调度器

```bash
# 测试训练好的模型
python test_ddpg_scheduler.py --mode test
```

或在代码中：

```python
# 创建DDPG调度器（测试模式）
scheduler = DDPGScheduler(
    node_info_manager=nim,
    training_mode=False  # 测试模式（不添加噪声）
)

# 加载训练好的模型
scheduler.agent  # 先初始化agent
scheduler.load_model("ddpg_model.pth")

# 运行仿真
simulation = EnergySimulation(network, scheduler=scheduler)
simulation.simulate()
```

### 3. 训练+测试（完整流程）

```bash
python test_ddpg_scheduler.py --mode both --episodes 100 --steps 200
```

## 超参数调优

### Actor和Critic学习率

| Actor LR | Critic LR | 效果 | 适用场景 |
|----------|-----------|------|---------|
| `1e-4` | `1e-3` | **推荐值** | 通用 |
| `5e-5` | `5e-4` | 更保守，更稳定 | 训练不稳定时 |
| `1e-3` | `1e-2` | 快速收敛，可能不稳定 | 快速实验 |

### 折扣因子 γ

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.95` | 更注重短期奖励 | 快速响应 |
| `0.99` | **推荐值**，平衡短期和长期 | 通用 |
| `0.999` | 更注重长期奖励 | 长期优化 |

### 软更新系数 τ

| 值 | 效果 | 适用场景 |
|----|------|---------|
| `0.001` | **推荐值**，慢速更新 | 稳定训练 |
| `0.005` | 较快更新 | 快速适应 |
| `0.01` | 快速更新，可能不稳定 | 快速实验 |

## 训练技巧

### 1. 课程学习 (Curriculum Learning)

从简单任务逐步过渡到复杂任务：

```python
# 阶段1：小网络（10节点）
train_ddpg(num_nodes=10, episodes=50)

# 阶段2：中等网络（20节点）
train_ddpg(num_nodes=20, episodes=50)

# 阶段3：大网络（30节点）
train_ddpg(num_nodes=30, episodes=50)
```

### 2. 奖励塑形 (Reward Shaping)

调整奖励权重以改善学习：

```python
# 更注重能量均衡
balance_reward = (prev_std - current_std) * 20.0  # 增加权重

# 更注重传输效率
efficiency_reward = efficiency * 10.0  # 增加权重
```

### 3. 探索策略调整

调整OU噪声参数：

```python
# 增加探索（训练初期）
noise = OUNoise(action_dim, sigma=0.3)

# 减少探索（训练后期）
noise = OUNoise(action_dim, sigma=0.1)
```

## 预期效果

### 训练收敛

- **Actor Loss**: 逐渐下降并趋于稳定
- **Critic Loss**: 逐渐下降至较小值
- **缓冲区**: 逐渐填满至最大容量

### 性能提升

| 指标 | 标准Lyapunov | DDPG (预期) | 提升 |
|------|-------------|------------|------|
| 能量均衡性 (CV) | 0.25 | **0.20** | **20%** |
| 平均传输效率 | 25% | **35%** | **40%** |
| 网络存活时间 | 5000步 | **6000步** | **20%** |

## 优势与局限

### 优势

1. ✅ **端到端学习** - 无需人工设计复杂的启发式规则
2. ✅ **自适应能力强** - 能够适应不同的网络拓扑和能量分布
3. ✅ **多目标优化** - 自动权衡多个优化目标
4. ✅ **连续动作空间** - 精确控制传输时长

### 局限

1. ⚠️ **训练时间长** - 需要大量样本（数千到数万步）
2. ⚠️ **计算开销大** - 神经网络推理比启发式算法慢
3. ⚠️ **需要GPU加速** - 训练大网络时建议使用GPU
4. ⚠️ **超参数敏感** - 需要仔细调优学习率等参数
5. ⚠️ **可解释性差** - 难以理解神经网络的决策逻辑

## 与其他调度器的对比

| 调度器 | 决策方式 | 时长选择 | 优化目标 | 计算复杂度 | 适用场景 |
|--------|---------|---------|---------|-----------|---------|
| **标准Lyapunov** | 启发式 | 固定1分钟 | 能量均衡 | O(N²) | 快速部署 |
| **自适应时长Lyapunov** | 启发式 | 枚举1-5分钟 | 能量均衡+效率 | O(N²D) | 效率优先 |
| **DDPG** | 深度学习 | 连续1-5分钟 | 多目标（自学习） | O(N²+NN) | **最优性能** |

## 相关文件

- **DDPG实现**: `src/scheduling/ddpg_scheduler.py`
- **训练测试**: `test_ddpg_scheduler.py`
- **依赖库**: PyTorch (torch, torch.nn, torch.optim)

## 依赖安装

```bash
# 安装PyTorch
pip install torch torchvision

# 或使用conda
conda install pytorch torchvision -c pytorch
```

## 参考文献

1. **DDPG算法**: Lillicrap et al. "Continuous control with deep reinforcement learning." ICLR 2016.
2. **Actor-Critic**: Sutton & Barto. "Reinforcement Learning: An Introduction." 2018.
3. **WSN能量管理**: 相关WSN能量优化论文

---

**创建日期**: 2024-11-03  
**功能状态**: ✅ 已实现（需训练）  
**推荐场景**: 追求最优性能，愿意投入训练时间的场景

