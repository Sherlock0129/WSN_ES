# Sim 模块

## 概述

`sim` 模块提供仿真主程序和并行执行功能，是系统运行的入口点。

## 主要组件

### 1. main.py

主仿真程序（旧版本）。

**核心功能：**
- 创建网络和仿真实例
- 运行仿真
- 保存结果

### 2. refactored_main.py

重构后的主仿真程序，使用新的配置管理和接口系统。

**核心功能：**
- 使用ConfigManager管理配置
- 支持多种调度器创建
- 支持并行仿真
- 支持深度学习调度器（DQN/DDPG）
- 支持ADCR和路径信息收集器

**主要函数：**
- `create_scheduler()`: 根据配置创建调度器
- `main()`: 主函数，运行仿真

**使用方法：**
```python
from sim.refactored_main import main
from config.simulation_config import load_config

# 加载配置
config_manager = load_config("config.json")

# 运行仿真
main(config_manager)
```

### 3. refactored_main_with_training.py

支持训练的仿真主程序。

**核心功能：**
- 支持DQN/DDPG模型训练
- 训练模式切换
- 模型保存和加载

### 4. parallel_main.py

并行仿真主程序。

**核心功能：**
- 多进程并行仿真
- 批量实验
- 结果汇总

### 5. parallel_executor.py

并行仿真执行器。

**核心功能：**
- 多进程任务分配
- 结果收集和汇总
- 权重扫描实验支持
- 种子管理（固定种子或不同种子）

**关键特性：**
- **多进程并行**：使用ProcessPoolExecutor实现并行执行
- **批量运行**：支持多次独立仿真运行
- **权重扫描**：支持对(w_b, w_d, w_l)进行线性扫描
- **种子管理**：
  - `use_same_seed=True`: 使用固定种子（对比实验，网络结构一致）
  - `use_same_seed=False`: 使用不同种子（独立实验，网络结构随机）
- **结果汇总**：自动生成汇总报告（均值、方差等）

**主要方法：**
- `run_parallel_simulations()`: 执行并行仿真
- `_run_single_simulation()`: 运行单次仿真（内部方法）
- `_calculate_weights()`: 计算当前运行的权重参数
- `_generate_summary_report()`: 生成汇总报告

**使用方法：**
```python
from sim.parallel_executor import ParallelSimulationExecutor
from config.simulation_config import load_config

# 加载配置
config_manager = load_config("config.json")

# 启用并行模式
config_manager.parallel_config.enabled = True
config_manager.parallel_config.num_runs = 10
config_manager.parallel_config.max_workers = 4
config_manager.parallel_config.use_same_seed = True  # 固定种子

# 可选：启用权重扫描
config_manager.parallel_config.enable_weight_scan = True
config_manager.parallel_config.w_b_start = 0.1
config_manager.parallel_config.w_b_step = 0.1

# 创建执行器
executor = ParallelSimulationExecutor(config_manager)

# 运行并行仿真
results = executor.run_parallel_simulations()
```

## 文件结构

```
sim/
├── __pycache__/
├── dqn_model.pth           # DQN模型文件（如果已训练）
├── main.py                 # 主仿真程序（旧版本）
├── parallel_executor.py    # 并行仿真执行器
├── parallel_main.py        # 并行仿真主程序
├── refactored_main.py      # 重构后的主仿真程序
└── refactored_main_with_training.py  # 支持训练的仿真主程序
```

## 运行方式

### 方式1：使用重构后的主程序

```bash
python -m sim.refactored_main
```

### 方式2：使用配置文件

```python
from sim.refactored_main import main
from config.simulation_config import load_config

config_manager = load_config("config.json")
main(config_manager)
```

### 方式3：并行仿真

```bash
python -m sim.parallel_main
```

### 方式4：训练模式

```python
from sim.refactored_main_with_training import main

# 设置训练模式
config_manager.scheduler_config.dqn_training_mode = True
config_manager.scheduler_config.dqn_training_episodes = 50

main(config_manager)
```

## 配置示例

### 基本配置

```json
{
  "simulation": {
    "time_steps": 10080,
    "enable_k_adaptation": false,
    "fixed_k": 3
  },
  "scheduler": {
    "scheduler_type": "DurationAwareLyapunovScheduler",
    "lyapunov_v": 0.5,
    "lyapunov_k": 3
  }
}
```

### DQN训练配置

```json
{
  "scheduler": {
    "enable_dqn": true,
    "dqn_training_mode": true,
    "dqn_training_episodes": 50,
    "dqn_action_dim": 10
  }
}
```

## 相关文档

- [快速启动指南](../../docs/快速启动指南.md)
- [DQN训练和使用完整指南](../../DQN训练和使用完整指南.md)
- [GPU加速使用指南](../../GPU加速使用指南.md)

