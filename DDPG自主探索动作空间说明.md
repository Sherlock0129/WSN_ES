# DDPG自主探索动作空间说明

## 🎯 核心特性

**DDPG（Deep Deterministic Policy Gradient）** 支持**连续动作空间**，可以输出任意实数值，让深度学习网络**自己寻找最优的传输时长**，而不限于离散的几个选项。

## 🔄 DQN vs DDPG 对比

| 特性 | DQN | DDPG |
|------|-----|------|
| **动作类型** | 离散 | 连续 |
| **动作空间** | 10个固定选项（1-10分钟） | 任意实数（如2.3分钟、7.8分钟） |
| **探索方式** | ε-greedy（随机选择） | OU噪声（连续扰动） |
| **网络结构** | Q网络 | Actor-Critic双网络 |
| **优势** | 简单、易训练 | 精细控制、连续优化 |
| **劣势** | 动作粒度粗 | 训练难度大 |

## 🚀 已实现的改进

### 1. **可配置的动作范围**

**修改文件**: `src/config/simulation_config.py`

```python
# 新增参数
ddpg_action_min: float = 1.0   # 最小传输时长（分钟）
ddpg_action_max: float = 10.0  # 最大传输时长（分钟）
```

**效果**：
- DDPG可以输出 [1.0, 10.0] 范围内的**任意值**
- 例如：2.34分钟、5.67分钟、8.91分钟
- 完全由网络学习决定，不限于整数

### 2. **改进的网络初始化**

```python
# Actor网络
nn.init.kaiming_normal_(weight)  # He初始化（适合ReLU）

# Critic网络  
nn.init.kaiming_normal_(weight)  # He初始化
```

**优势**：
- 更稳定的训练
- 更快的收敛
- 避免梯度消失/爆炸

### 3. **归一化的奖励函数**

```python
# 奖励范围：-50 到 +50
total_reward = np.clip(total_reward, -50.0, 50.0)
```

**防止**：
- 奖励爆炸
- Q值发散
- 训练不稳定

### 4. **灵活的动作映射**

```python
# Actor输出通过tanh限制在[-1, 1]
x = torch.tanh(output)

# 映射到配置的范围
action = action_min + (x + 1.0) / 2.0 * (action_max - action_min)
```

**示例**：
- tanh输出 = -0.5 → action = 3.25分钟
- tanh输出 = 0.0 → action = 5.5分钟
- tanh输出 = 0.8 → action = 9.1分钟

## 📊 自主探索机制

### 1. **OU噪声探索**

DDPG使用Ornstein-Uhlenbeck过程生成探索噪声：

```python
# 训练模式
action = actor(state) + OU_noise
# 例如：5.0 + 0.3 = 5.3分钟

# 测试模式
action = actor(state)  # 无噪声，直接使用最优策略
```

**特点**：
- 时间相关的噪声（更平滑的探索）
- 自动衰减（随着训练进行）
- 适合连续控制

### 2. **学习过程**

```
初期：随机探索
├─ 回合1-20: action ∈ [1, 10]（均匀分布）
├─ 回合20-50: 逐渐收敛到高回报区域
└─ 回合50+: 稳定在最优值附近（如3-7分钟）

最终：自主发现最优传输时长
例如：网络学习到 duration ≈ 5.2分钟 最优
```

## ⚙️ 配置说明

### 基本配置（simulation_config.py）

```python
# 启用DDPG
enable_ddpg: bool = True
ddpg_training_mode: bool = True  # True=训练，False=使用已训练模型

# 训练参数
ddpg_training_episodes: int = 100  # 训练回合数
ddpg_save_interval: int = 10       # 每10回合保存一次模型

# 🎯 动作空间配置（让DDPG自己探索）
ddpg_action_min: float = 1.0       # 最小时长
ddpg_action_max: float = 10.0      # 最大时长

# 网络超参数
ddpg_actor_lr: float = 1e-4        # Actor学习率
ddpg_critic_lr: float = 1e-3       # Critic学习率
ddpg_gamma: float = 0.99           # 折扣因子
ddpg_tau: float = 0.001            # 软更新系数
```

### 高级配置

#### 场景1：保守探索（小范围）
```python
ddpg_action_min: float = 1.0
ddpg_action_max: float = 5.0  # 限制在5分钟内
```
- 适合：小规模网络
- 优势：训练更快、更稳定

#### 场景2：激进探索（大范围）
```python
ddpg_action_min: float = 0.5   # 允许极短传输
ddpg_action_max: float = 20.0  # 允许极长传输
```
- 适合：大规模网络、研究场景
- 优势：可能发现意外的最优策略

#### 场景3：非对称范围
```python
ddpg_action_min: float = 3.0   # 不允许太短的传输
ddpg_action_max: float = 15.0  # 鼓励长传输
```
- 适合：特定应用约束

## 🎓 工作原理

### 决策流程

```
时间步t → 观察状态s
         ↓
    Actor网络(s)
         ↓
    输出动作a ∈ [1, 10]分钟（连续值）
         ↓
    例如：a = 5.234分钟
         ↓
    执行传输（5.234 × 300J = 1570.2J）
         ↓
    获得奖励r
         ↓
    存储(s, a, r, s') → 经验回放
         ↓
    更新Actor和Critic网络
         ↓
    下次决策改进
```

### 与DurationAwareLyapunovScheduler的区别

| 特性 | DurationAware | DDPG |
|------|--------------|------|
| **决策方式** | 遍历1-5分钟，选最优 | 神经网络直接输出 |
| **计算量** | 高（每次尝试5个值） | 低（一次前向传播） |
| **灵活性** | 只能选整数 | 可以选任意实数 |
| **需要训练** | 否 | 是（100回合） |
| **适应性** | 固定策略 | 自适应学习 |

## 📈 使用流程

### 步骤1：训练DDPG模型

**配置**：
```python
enable_ddpg: bool = True
ddpg_training_mode: bool = True
ddpg_training_episodes: int = 100
```

**运行**：
```bash
python src/sim/refactored_main.py
```

**观察日志**：
```
============================================================
使用DDPG深度强化学习调度器（连续动作空间：自主探索）
  - 训练模式: True
  - 模型路径: ddpg_model.pth
  - 动作范围: [1.0, 10.0] 分钟
============================================================

[DDPG] 使用设备: cuda:0
[DDPG] GPU设备名称: NVIDIA GeForce RTX 3070 Ti Laptop GPU
[DDPG] 动作空间范围: [1.0, 10.0] 分钟

训练回合 1/100
...
训练统计:
  - 平均Actor损失: 12.3456
  - 平均Critic损失: 45.6789
  - 缓冲区: 1024
```

### 步骤2：使用训练好的模型

**配置**：
```python
ddpg_training_mode: bool = False  # 切换到测试模式
```

**运行**：
```bash
python src/sim/refactored_main.py
```

**效果**：
- 不再探索，直接使用学到的最优策略
- 决策速度快
- 性能稳定

### 步骤3：分析学到的策略

查看日志中的传输时长：
```
[DDPG] 选择传输时长: 5.23分钟
[DDPG] 选择传输时长: 3.67分钟
[DDPG] 选择传输时长: 7.89分钟
```

**分析**：
- 如果集中在某个范围（如4-6分钟），说明学到了最优区间
- 如果分布广泛，可能需要更多训练

## 🔬 实验建议

### 实验1：动作范围影响

```python
# 配置A：小范围
ddpg_action_max: float = 5.0

# 配置B：大范围  
ddpg_action_max: float = 20.0

# 比较：哪个收敛更快？最优值是多少？
```

### 实验2：探索策略

```python
# OU噪声参数（在ddpg_scheduler.py中）
sigma = 0.2  # 探索强度（更大=更多探索）
theta = 0.15 # 回归速度（更大=更快回归均值）
```

### 实验3：与固定策略对比

| 策略 | 平均能量方差 | 网络存活时间 | 传输效率 |
|------|-------------|-------------|---------|
| DurationAware(1-5分钟遍历) | ? | ? | ? |
| DDPG(1-10分钟学习) | ? | ? | ? |

## 📝 常见问题

### Q1: DDPG输出的小数时长如何执行？

**A**: 两种方式：

1. **四舍五入到整数**（当前实现）：
   ```python
   duration = int(np.round(5.234))  # → 5分钟
   ```

2. **保留小数**（更精细）：
   ```python
   duration = 5.234  # 直接使用
   energy = 5.234 × 300 = 1570.2J
   ```

### Q2: 如何知道DDPG学到了什么？

**A**: 三种方法：

1. **查看统计图**：`duration_statistics.html`
2. **检查模型权重**：可视化Actor网络
3. **测试模式运行**：观察输出的时长分布

### Q3: 动作范围应该设多大？

**A**: 根据场景：

- **小网络**（<30节点）：1-5分钟
- **中网络**（30-100节点）：1-10分钟
- **大网络**（>100节点）：1-20分钟

### Q4: DDPG vs DQN，选哪个？

| 场景 | 推荐 | 理由 |
|------|------|------|
| 快速原型 | DQN | 训练快、简单 |
| 精细控制 | DDPG | 连续值、更优 |
| 有限计算资源 | DQN | 计算量小 |
| 追求最优性能 | DDPG | 可能找到更好策略 |

## 🎨 可视化支持

DDPG生成的计划也支持duration可视化：

### 生成的图表

1. **传输时间线图**
   ```
   传输1: |████████████| (7.2分钟, 2160J) ← 非整数时长
   传输2: |██████| (3.8分钟, 1140J)
   ```

2. **传输路径图**
   - 标签：`"7.2min\n2160J"`
   - 线宽与时长成正比

3. **时长统计图**
   - X轴：连续的时长值
   - 可以看到时长的分布

## ⚙️ 运行示例

### 训练模式

```bash
# 1. 确认配置
python check_scheduler_config.py

# 输出应该包含：
# enable_ddpg: True
# ddpg_training_mode: True
# 动作范围: [1.0, 10.0] 分钟

# 2. 开始训练
python src/sim/refactored_main.py

# 预期日志：
# ============================================================
# 使用DDPG深度强化学习调度器（连续动作空间：自主探索）
#   - 训练模式: True
#   - 动作范围: [1.0, 10.0] 分钟
# ============================================================
# [DDPG] 使用设备: cuda:0
# [DDPG] 动作空间范围: [1.0, 10.0] 分钟
#
# 训练回合 1/100
# ...
```

### 测试模式

```python
# simulation_config.py
ddpg_training_mode: bool = False
```

```bash
python src/sim/refactored_main.py
```

**效果**：使用学到的最优策略，无探索噪声

## 🔍 调试技巧

### 打印动作分布

在 `ddpg_scheduler.py` 的 `plan()` 方法中添加：

```python
# 在生成计划后
durations = [p['duration'] for p in plans]
if durations:
    print(f"[DDPG] 本轮选择的时长: {durations}")
    print(f"[DDPG] 平均时长: {np.mean(durations):.2f}分钟")
    print(f"[DDPG] 时长范围: [{min(durations):.2f}, {max(durations):.2f}]分钟")
```

### 监控学习进度

```python
# 每回合结束后
stats = scheduler.get_training_stats()
print(f"缓冲区大小: {stats['buffer_size']}")
print(f"Actor损失: {stats['avg_actor_loss']:.4f}")
print(f"Critic损失: {stats['avg_critic_loss']:.4f}")
```

## 🎯 预期学习曲线

### 训练早期（回合1-20）

```
动作分布：几乎均匀分布在[1, 10]
损失：较高（Actor ~100, Critic ~500）
性能：较差，随机探索
```

### 训练中期（回合20-60）

```
动作分布：开始集中
损失：下降（Actor ~50, Critic ~200）
性能：改善，发现有效策略
```

### 训练后期（回合60-100）

```
动作分布：集中在最优区域（如4-7分钟）
损失：稳定（Actor ~10, Critic ~50）
性能：优秀，接近最优
```

## 💡 高级技巧

### 1. 课程学习（Curriculum Learning）

先训练小范围，再扩大：

```python
# 阶段1（前50回合）
ddpg_action_max: float = 5.0

# 阶段2（后50回合）
ddpg_action_max: float = 10.0
```

### 2. 动作约束学习

添加物理约束：
```python
# 在actor forward中
if energy_low:
    # 低能量时限制短传输
    action_max_adjusted = 3.0
else:
    action_max_adjusted = 10.0
```

### 3. 多目标优化

不同节点对使用不同动作：
```python
# 为每个(donor, receiver)对单独决策
for d, r in pairs:
    pair_state = compute_pair_state(d, r)
    duration = actor(pair_state)
```

## 📚 相关文档

- `DDPG深度强化学习调度器说明.md` - DDPG详细原理
- `DQN离散动作调度器说明.md` - DQN对比
- `传输时长可视化说明.md` - 可视化指南
- `深度学习调度器对比.md` - DQN vs DDPG

## ✅ 总结

| 改进 | 说明 |
|------|------|
| ✅ 连续动作空间 | 输出任意实数（如5.23分钟） |
| ✅ 可配置范围 | action_min到action_max自定义 |
| ✅ 自主探索 | 网络自己学习最优时长 |
| ✅ 归一化奖励 | 稳定训练 |
| ✅ GPU加速 | RTX 3070 Ti训练 |
| ✅ 完整可视化 | 支持所有时长感知图表 |

🚀 **DDPG现在可以在[1, 10]分钟范围内自由探索，找到最优的传输时长！**


