# DQN调度器系统集成说明

## 概述

DQN深度强化学习调度器已完全集成到WSN仿真系统中。现在可以通过配置文件轻松切换使用DQN、DDPG或传统调度器。

## 🎯 系统架构

```
WSN仿真系统
│
├── 配置层 (simulation_config.py)
│   ├── 调度器开关 (enable_dqn / enable_ddpg)
│   ├── DQN超参数配置
│   └── DDPG超参数配置
│
├── 主程序 (refactored_main.py)
│   ├── create_scheduler() ← 自动选择调度器
│   ├── 模型加载逻辑
│   └── 运行仿真
│
└── 调度器实现
    ├── DQNScheduler (dqn_scheduler.py)
    ├── DDPGScheduler (ddpg_scheduler.py)
    └── 传统调度器 (schedulers.py)
```

## 📝 已完成的集成工作

### 1. 配置文件修改 (`src/config/simulation_config.py`)

**添加的配置项**:

```python
@dataclass
class SchedulerConfig:
    # DQN配置（新增）
    enable_dqn: bool = False  # DQN开关
    dqn_model_path: str = "../tests/dqn_model.pth"
    dqn_training_mode: bool = False
    dqn_action_dim: int = 10
    dqn_lr: float = 1e-3
    dqn_gamma: float = 0.99
    dqn_tau: float = 0.005
    dqn_buffer_capacity: int = 10000
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: float = 0.995

    # DDPG配置（新增）
    enable_ddpg: bool = False  # DDPG开关
    ddpg_model_path: str = "ddpg_model.pth"
    ddpg_training_mode: bool = False
    ddpg_action_dim: int = 1
    ddpg_actor_lr: float = 1e-4
    ddpg_critic_lr: float = 1e-3
    ddpg_gamma: float = 0.99
    ddpg_tau: float = 0.001
    ddpg_buffer_capacity: int = 10000
```

### 2. 主程序修改 (`src/sim/refactored_main.py`)

**添加的功能**:

1. **导入深度学习调度器**
```python
try:
    from scheduling.dqn_scheduler import DQNScheduler
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

try:
    from scheduling.ddpg_scheduler import DDPGScheduler
    DDPG_AVAILABLE = True
except ImportError:
    DDPG_AVAILABLE = False
```

2. **自动选择调度器**
```python
def create_scheduler(config_manager, network):
    # 优先检查DQN/DDPG开关
    if config.enable_dqn:
        scheduler = DQNScheduler(...)
        # 自动加载模型
        if not training_mode and model_exists:
            scheduler.load_model(...)
    elif config.enable_ddpg:
        scheduler = DDPGScheduler(...)
    else:
        # 使用传统调度器
        scheduler = LyapunovScheduler(...)
```

### 3. 创建的配置文件

**`config_dqn_example.yaml`** - DQN使用示例配置

```yaml
scheduler:
  enable_dqn: true
  dqn_model_path: "dqn_model.pth"
  dqn_training_mode: false
  
simulation:
  time_steps: 1000
  
network:
  num_nodes: 20
```

### 4. 创建的文档

| 文档 | 说明 |
|------|------|
| `使用DQN调度器指南.md` | **系统使用指南**（推荐阅读） |
| `DQN系统集成说明.md` | 本文档（集成说明） |
| `DQN快速入门指南.md` | DQN基础教程 |
| `DQN离散动作调度器说明.md` | DQN技术详解 |
| `深度学习调度器对比.md` | 调度器对比 |

## 🚀 使用方式

### 方式1：配置文件（推荐）⭐

```bash
# 1. 使用示例配置
python src/sim/refactored_main.py --config config_dqn_example.yaml

# 2. 或创建自定义配置
python src/sim/refactored_main.py --config my_config.yaml
```

### 方式2：修改默认配置

在 `src/config/simulation_config.py` 中：

```python
@dataclass
class SchedulerConfig:
    enable_dqn: bool = True  # ← 改为True
    dqn_model_path: str = "dqn_model.pth"
```

然后直接运行：

```bash
python src/sim/refactored_main.py
```

### 方式3：编程方式

```python
from config.simulation_config import ConfigManager
from sim.refactored_main import run_simulation

config = ConfigManager()
config.scheduler_config.enable_dqn = True
run_simulation()
```

## ⚙️ 调度器切换

### 切换到DQN

```yaml
scheduler:
  enable_dqn: true      # 启用DQN
  enable_ddpg: false
```

### 切换到DDPG

```yaml
scheduler:
  enable_dqn: false
  enable_ddpg: true     # 启用DDPG
```

### 切换到传统调度器

```yaml
scheduler:
  enable_dqn: false
  enable_ddpg: false
  scheduler_type: "LyapunovScheduler"  # 使用传统调度器
```

## 📊 优先级规则

调度器选择的优先级：

```
1. enable_dqn = true      → 使用DQN（最高优先级）
2. enable_ddpg = true     → 使用DDPG
3. scheduler_type         → 使用传统调度器
```

⚠️ **注意**: `enable_dqn` 和 `enable_ddpg` 会覆盖 `scheduler_type`

## 🔧 完整使用流程

### 步骤1：训练模型（首次使用）

```bash
# 训练DQN模型
python run_dqn_simulation.py --train --episodes 50

# 或训练DDPG模型
python test_ddpg_scheduler.py --mode train --episodes 100
```

### 步骤2：配置系统

**选项A：使用YAML配置**

创建 `my_config.yaml`:
```yaml
scheduler:
  enable_dqn: true
  dqn_model_path: "dqn_model.pth"
  dqn_training_mode: false
```

**选项B：修改Python配置**

在 `src/config/simulation_config.py`:
```python
enable_dqn: bool = True
```

### 步骤3：运行仿真

```bash
# 使用系统总入口
python src/sim/refactored_main.py

# 或使用配置文件
python src/sim/refactored_main.py --config my_config.yaml
```

### 步骤4：查看结果

结果保存在 `data/[timestamp]/` 目录：
- 能量统计
- 传输计划
- 可视化图表

## 📈 运行示例

### 示例1：标准DQN运行

```bash
# 1. 确保有训练好的模型
ls dqn_model.pth

# 2. 运行（使用示例配置）
python src/sim/refactored_main.py --config config_dqn_example.yaml

# 3. 查看输出
# 会显示：
# ============================================================
# 使用DQN深度强化学习调度器（离散动作空间：1-10分钟）
#   - 训练模式: False
#   - 模型路径: dqn_model.pth
#   - 动作空间: 10个离散动作
# ============================================================
# ✓ DQN模型已加载: dqn_model.pth
```

### 示例2：对比测试

```bash
# 运行DQN
python src/sim/refactored_main.py --config config_dqn.yaml > dqn_result.txt

# 运行Lyapunov
python src/sim/refactored_main.py --config config_lyapunov.yaml > lyap_result.txt

# 对比结果
diff dqn_result.txt lyap_result.txt
```

### 示例3：长期仿真

```yaml
# config_long.yaml
scheduler:
  enable_dqn: true
  
simulation:
  time_steps: 10080  # 7天
  
network:
  num_nodes: 30      # 大网络
```

```bash
python src/sim/refactored_main.py --config config_long.yaml
```

## 🎯 关键特性

### 1. 自动模型加载

系统会自动检测并加载模型：

```
如果 training_mode = false 且模型文件存在:
    自动加载模型
否则:
    警告用户并使用随机初始化
```

### 2. 智能降级

如果PyTorch未安装：

```
尝试导入DQN调度器
  ↓
失败 → 显示警告，DQN不可用
  ↓
系统继续运行，使用传统调度器
```

### 3. 灵活配置

支持3种配置方式：
1. YAML文件配置
2. Python类配置
3. 代码动态配置

## 📚 文件清单

### 核心代码

| 文件 | 修改内容 |
|------|---------|
| `src/config/simulation_config.py` | ✅ 添加DQN/DDPG配置 |
| `src/sim/refactored_main.py` | ✅ 添加DQN/DDPG支持 |
| `src/scheduling/dqn_scheduler.py` | ✅ DQN实现 |
| `src/scheduling/ddpg_scheduler.py` | ✅ DDPG实现 |

### 配置文件

| 文件 | 说明 |
|------|------|
| `config_dqn_example.yaml` | ✅ DQN配置示例 |

### 文档

| 文件 | 说明 |
|------|------|
| `使用DQN调度器指南.md` | ✅ **系统使用指南** |
| `DQN系统集成说明.md` | ✅ 本文档 |
| `DQN快速入门指南.md` | ✅ DQN基础 |
| `DQN离散动作调度器说明.md` | ✅ DQN详解 |
| `深度学习调度器对比.md` | ✅ 调度器对比 |

### 辅助脚本

| 文件 | 说明 |
|------|------|
| `run_dqn_simulation.py` | ✅ 独立训练/测试脚本 |
| `test_dqn_scheduler.py` | ✅ 详细测试脚本 |

## ✅ 验证清单

系统集成完成度：

- [x] DQN调度器实现
- [x] DDPG调度器实现
- [x] 配置文件集成
- [x] 主程序集成
- [x] 模型自动加载
- [x] 错误处理
- [x] 文档完善
- [x] 配置示例
- [x] 使用指南

## 🔍 测试建议

### 测试1：基本功能

```bash
# 启用DQN
python src/sim/refactored_main.py --config config_dqn_example.yaml

# 检查：
# - 是否显示"使用DQN"提示
# - 是否成功加载模型
# - 是否正常运行
```

### 测试2：调度器切换

```bash
# 测试DQN
python src/sim/refactored_main.py --config config_dqn.yaml

# 测试Lyapunov（对比）
python src/sim/refactored_main.py --config config_lyapunov.yaml

# 对比结果
```

### 测试3：错误处理

```bash
# 测试没有模型文件的情况
rm dqn_model.pth
python src/sim/refactored_main.py --config config_dqn_example.yaml

# 应该看到警告但继续运行
```

## 💡 常见问题

### Q1: 如何确认DQN已启用？

**A**: 运行时会看到：

```
============================================================
使用DQN深度强化学习调度器（离散动作空间：1-10分钟）
  - 训练模式: False
  - 模型路径: dqn_model.pth
  - 动作空间: 10个离散动作
============================================================
```

### Q2: enable_dqn和scheduler_type哪个优先？

**A**: `enable_dqn`优先级更高，会覆盖`scheduler_type`

### Q3: 可以同时启用DQN和DDPG吗？

**A**: 不建议。如果同时启用，DQN优先级更高

### Q4: 如何切换回传统调度器？

**A**: 设置 `enable_dqn = false` 和 `enable_ddpg = false`

## 🎓 后续工作

### 短期

- [ ] 性能对比实验
- [ ] 参数调优
- [ ] 更多测试用例

### 长期

- [ ] 在线学习支持
- [ ] 多智能体DQN
- [ ] 分布式训练

## 📞 获取帮助

如有问题，请参考：

1. **`使用DQN调度器指南.md`** - 详细使用说明
2. **`DQN快速入门指南.md`** - 快速开始
3. **`DQN离散动作调度器说明.md`** - 技术详解

---

**集成完成！现在可以在系统中使用DQN调度器了！** 🎉

```bash
# 立即开始
python src/sim/refactored_main.py --config config_dqn_example.yaml
```

