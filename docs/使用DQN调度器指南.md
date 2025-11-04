# 在系统中使用DQN调度器指南

## 概述

本指南介绍如何在现有WSN仿真系统中使用DQN深度强化学习调度器。系统已完全集成DQN，只需修改配置即可使用。

## 🚀 快速开始（3步）

### 第1步：训练DQN模型

首先需要训练一个DQN模型（如果还没有）：

```bash
# 使用独立训练脚本（推荐）
python run_dqn_simulation.py --train --episodes 50

# 或使用测试脚本
python test_dqn_scheduler.py --mode train --episodes 50
```

训练完成后会生成 `dqn_model.pth` 文件。

### 第2步：修改配置文件

**方式A：使用提供的示例配置**

```bash
# 直接使用示例配置运行
python src/sim/refactored_main.py --config config_dqn_example.yaml
```

**方式B：修改配置参数**

在 `src/config/simulation_config.py` 中修改：

```python
@dataclass
class SchedulerConfig:
    # 启用DQN调度器
    enable_dqn: bool = True  # 改为True
    dqn_model_path: str = "../tests/dqn_model.pth"
    dqn_training_mode: bool = False  # False=使用模型，True=训练

    # 其他参数保持默认即可
    dqn_action_dim: int = 10
    dqn_lr: float = 1e-3
    ...
```

### 第3步：运行仿真

```bash
# 使用系统总入口
python src/sim/refactored_main.py

# 或使用配置文件
python src/sim/refactored_main.py --config config_dqn_example.yaml
```

## 📋 详细使用方法

### 方法1：修改配置文件（推荐）⭐

#### 1.1 创建自定义配置文件

创建 `my_config.yaml`:

```yaml
# 调度器配置
scheduler:
  enable_dqn: true                    # 启用DQN
  dqn_model_path: "dqn_model.pth"     # 模型路径
  dqn_training_mode: false            # 测试模式
  
# 仿真配置
simulation:
  time_steps: 10080                   # 7天仿真
  enable_energy_sharing: true
  
# 网络配置
network:
  num_nodes: 25
```

#### 1.2 运行

```bash
python src/sim/refactored_main.py --config my_config.yaml
```

### 方法2：修改Python配置

在 `src/config/simulation_config.py` 中：

```python
@dataclass
class SchedulerConfig:
    # 直接修改默认值
    enable_dqn: bool = True  # ← 改这里
    dqn_model_path: str = "dqn_model.pth"
    dqn_training_mode: bool = False
```

然后直接运行：

```bash
python src/sim/refactored_main.py
```

### 方法3：编程方式

```python
from config.simulation_config import ConfigManager
from sim.refactored_main import run_simulation

# 创建配置
config = ConfigManager()

# 启用DQN
config.scheduler_config.enable_dqn = True
config.scheduler_config.dqn_model_path = "dqn_model.pth"
config.scheduler_config.dqn_training_mode = False

# 运行仿真
run_simulation()
```

## ⚙️ 配置参数说明

### 核心开关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_dqn` | bool | False | **启用DQN调度器**（优先级最高） |
| `enable_ddpg` | bool | False | 启用DDPG调度器 |

⚠️ **注意**: `enable_dqn`优先级高于`scheduler_type`，启用后会覆盖传统调度器。

### DQN参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dqn_model_path` | str | "dqn_model.pth" | 模型文件路径（相对项目根目录） |
| `dqn_training_mode` | bool | False | 训练模式开关 |
| `dqn_action_dim` | int | 10 | 动作空间维度（1-10分钟） |
| `dqn_lr` | float | 1e-3 | 学习率 |
| `dqn_gamma` | float | 0.99 | 折扣因子 |
| `dqn_tau` | float | 0.005 | 软更新系数 |
| `dqn_buffer_capacity` | int | 10000 | 经验回放容量 |
| `dqn_epsilon_start` | float | 1.0 | 初始探索率 |
| `dqn_epsilon_end` | float | 0.01 | 最终探索率 |
| `dqn_epsilon_decay` | float | 0.995 | 探索率衰减 |

## 📝 使用场景

### 场景1：使用已训练模型（最常用）⭐

```yaml
scheduler:
  enable_dqn: true
  dqn_model_path: "dqn_model.pth"
  dqn_training_mode: false    # 测试模式
```

**特点**:
- ✅ 直接使用训练好的模型
- ✅ 不进行探索（ε=0）
- ✅ 性能最优
- ✅ 运行速度快

### 场景2：在线训练（研究用）

```yaml
scheduler:
  enable_dqn: true
  dqn_training_mode: true     # 训练模式
  dqn_epsilon_start: 1.0      # 高探索率
```

**特点**:
- ✅ 边运行边学习
- ⚠️ 初期性能较差（高探索）
- ⚠️ 需要多个回合
- ⚠️ 计算时间长

### 场景3：对比测试

```yaml
# 配置1：DQN
scheduler:
  enable_dqn: true
  dqn_model_path: "dqn_model.pth"

# 配置2：传统Lyapunov（对比）
scheduler:
  enable_dqn: false
  scheduler_type: "LyapunovScheduler"
```

**使用方法**:
```bash
# 运行DQN
python src/sim/refactored_main.py --config config_dqn.yaml

# 运行Lyapunov
python src/sim/refactored_main.py --config config_lyapunov.yaml

# 对比结果
```

### 场景4：DDPG调度器

```yaml
scheduler:
  enable_ddpg: true            # 启用DDPG
  ddpg_model_path: "ddpg_model.pth"
  ddpg_training_mode: false
```

## 🔧 运行示例

### 示例1：标准运行

```bash
# 1. 修改配置启用DQN
# 在 simulation_config.py 中设置 enable_dqn = True

# 2. 运行仿真
python src/sim/refactored_main.py

# 3. 查看结果
# 结果保存在 data/[timestamp]/ 目录
```

### 示例2：使用配置文件

```bash
# 使用提供的示例配置
python src/sim/refactored_main.py --config config_dqn_example.yaml

# 或创建自定义配置
python src/sim/refactored_main.py --config my_dqn_config.yaml
```

### 示例3：长时间仿真

```yaml
# long_simulation.yaml
scheduler:
  enable_dqn: true
  dqn_model_path: "dqn_model.pth"
  
simulation:
  time_steps: 10080    # 7天
  
network:
  num_nodes: 30        # 大网络
```

```bash
python src/sim/refactored_main.py --config long_simulation.yaml
```

## 📊 运行时输出

### 正常启动输出

```
================================================================================
使用DQN深度强化学习调度器（离散动作空间：1-10分钟）
  - 训练模式: False
  - 模型路径: dqn_model.pth
  - 动作空间: 10个离散动作
================================================================================
[DQN] 模型已加载: dqn_model.pth
✓ DQN模型已加载: dqn_model.pth
```

### 如果模型文件不存在

```
⚠ DQN模型文件不存在: dqn_model.pth
  将使用随机初始化的网络（性能可能较差）
```

**解决方法**: 先训练模型
```bash
python run_dqn_simulation.py --train --episodes 50
```

## 🎯 最佳实践

### 1. 模型管理

```bash
# 训练不同版本
python run_dqn_simulation.py --train --episodes 50 --model dqn_v1.pth
python run_dqn_simulation.py --train --episodes 100 --model dqn_v2.pth

# 在配置中使用
scheduler:
  dqn_model_path: "dqn_v2.pth"  # 使用v2版本
```

### 2. 参数调优

```yaml
# 对于稳定网络
scheduler:
  dqn_gamma: 0.99      # 高折扣因子
  dqn_tau: 0.001       # 慢速更新

# 对于动态网络
scheduler:
  dqn_gamma: 0.95      # 低折扣因子
  dqn_tau: 0.01        # 快速更新
```

### 3. 性能监控

```python
# 在仿真后检查DQN统计
if hasattr(scheduler, 'get_training_stats'):
    stats = scheduler.get_training_stats()
    print(f"平均损失: {stats['avg_loss']}")
    print(f"缓冲区大小: {stats['buffer_size']}")
```

## 🔍 故障排除

### 问题1：PyTorch未安装

**错误**:
```
ImportError: DQN调度器已启用但PyTorch未安装
```

**解决**:
```bash
pip install torch torchvision
```

### 问题2：模型文件不存在

**错误**:
```
⚠ DQN模型文件不存在: dqn_model.pth
```

**解决**:
```bash
# 训练模型
python run_dqn_simulation.py --train --episodes 50

# 或下载预训练模型（如果有）
```

### 问题3：内存不足

**解决**: 减小缓冲区
```yaml
scheduler:
  dqn_buffer_capacity: 5000  # 降低容量
```

### 问题4：训练不稳定

**解决**: 调整学习参数
```yaml
scheduler:
  dqn_lr: 0.0005           # 降低学习率
  dqn_epsilon_decay: 0.998  # 慢速探索衰减
```

## 📚 相关文档

- `DQN快速入门指南.md` - DQN基础使用
- `DQN离散动作调度器说明.md` - DQN技术详解
- `深度学习调度器对比.md` - 调度器对比
- `config_dqn_example.yaml` - 配置示例

## 🎓 进阶使用

### 自定义奖励函数

如需修改奖励函数，编辑 `src/scheduling/dqn_scheduler.py`:

```python
def _compute_reward(self, prev_energies, current_energies, plans):
    # 自定义奖励计算
    balance_reward = (prev_std - current_std) * 10.0
    efficiency_reward = efficiency * 5.0
    # ... 添加自己的奖励项
    return total_reward
```

### 多调度器对比

```python
from config.simulation_config import ConfigManager

configs = []

# DQN配置
config_dqn = ConfigManager()
config_dqn.scheduler_config.enable_dqn = True
configs.append(("DQN", config_dqn))

# Lyapunov配置
config_lyap = ConfigManager()
config_lyap.scheduler_config.enable_dqn = False
configs.append(("Lyapunov", config_lyap))

# 运行对比
for name, config in configs:
    print(f"运行 {name}...")
    run_simulation()
```

## ✅ 检查清单

使用DQN前：
- [ ] 已安装PyTorch
- [ ] 已训练DQN模型（有dqn_model.pth文件）
- [ ] 配置文件中enable_dqn=true
- [ ] 模型路径正确

运行中：
- [ ] 看到"使用DQN深度强化学习调度器"提示
- [ ] 看到"模型已加载"确认
- [ ] 仿真正常运行

运行后：
- [ ] 查看传输时长分布
- [ ] 对比能量均衡性
- [ ] 分析动作选择

---

**现在可以开始使用DQN调度器了！** 🚀

```bash
# 快速开始
python src/sim/refactored_main.py --config config_dqn_example.yaml
```

