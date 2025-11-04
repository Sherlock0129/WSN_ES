# Config 模块

## 概述

`config` 模块负责集中管理无线传感器网络仿真的所有配置参数。采用 dataclass 进行强类型分组，提供统一的配置管理接口。

## 主要功能

### 1. 配置类定义

模块定义了以下配置类，每个类对应一个功能领域：

- **NodeConfig**: 节点物理和能量模型参数
  - 电池容量、电压
  - 太阳能采集参数（效率、面积、辐照度等）
  - 无线能量传输参数（充电能量、电子能耗、功放损耗等）
  - 能量衰减率、传感能耗

- **NetworkConfig**: 网络拓扑与规模参数
  - 节点数量、区域尺寸
  - 分布模式（uniform/random）
  - 能量空洞模式配置
  - 物理中心节点配置
  - 能量分配模式

- **SimulationConfig**: 仿真运行控制参数
  - 时间步数、传能间隔
  - K值自适应参数
  - 智能被动传能参数
  - GPU加速开关
  - ADCR链路层开关

- **SchedulerConfig**: 调度器策略与超参数
  - 调度器类型选择
  - LyapunovScheduler 参数
  - ClusterScheduler 参数
  - PredictionScheduler 参数
  - PowerControlScheduler 参数
  - DurationAwareLyapunovScheduler 参数（传输时长优化）
  - DQN/DDPG 深度强化学习调度器参数

- **ADCRConfig**: ADCR链路层算法参数
  - 聚类轮换周期
  - 邻居半径、簇头选择参数
  - 路径规划与能耗结算参数
  - 可视化参数

- **EETORConfig**: EETOR路由算法参数
  - 能量传输效率模型参数
  - 邻居构建参数
  - 能量状态感知参数
  - 信息感知路由参数

- **PathCollectorConfig**: 路径信息收集器配置
  - 能量消耗模式（free/full）
  - 信息量累积模式
  - 机会主义信息传递参数
  - 延迟上报参数

- **PathCollectorConfig**: 基于路径的信息收集器配置
  - 能量消耗模式（free/full）
  - 信息量累积模式
  - 机会主义信息传递参数
  - 延迟上报参数
  - 估算参数（衰减率、太阳能模型）

- **ParallelConfig**: 并行仿真配置
  - 批量运行参数（运行次数、进程数）
  - 种子管理（固定种子或不同种子）
  - 权重扫描实验参数（w_b, w_d, w_l）
  - 输出管理参数（结果保存、汇总报告）

### 2. ConfigManager 配置管理器

`ConfigManager` 类提供统一的配置管理接口：

- **默认值管理**: 使用 dataclass 默认值作为单一真实来源（Single Source of Truth）
- **配置文件加载**: 支持从 JSON 文件加载配置并覆盖默认值
- **配置保存**: 支持将当前配置保存到 JSON 文件
- **配置导出**: 支持导出配置到会话目录
- **工厂方法**: 提供 `create_network()`, `create_sensor_node()`, `create_energy_simulation()` 等工厂方法，自动将配置注入到对象构造函数

## 使用方法

### 基本使用

```python
from config.simulation_config import ConfigManager, get_config

# 获取全局配置管理器（使用默认值）
config_manager = get_config()

# 从文件加载配置
config_manager = ConfigManager(config_file="config.json")

# 创建网络对象（自动注入配置）
network = config_manager.create_network()

# 创建传感器节点
node = config_manager.create_sensor_node(
    node_id=1,
    position=[0.0, 0.0],
    has_solar=True
)

# 创建能量仿真实例
simulation = config_manager.create_energy_simulation(
    network=network,
    scheduler=scheduler
)
```

### 配置文件格式

配置文件为 JSON 格式，支持按模块分组：

```json
{
  "node": {
    "initial_energy": 40000.0,
    "low_threshold": 0.1,
    "high_threshold": 0.9,
    ...
  },
  "network": {
    "num_nodes": 30,
    "max_hops": 3,
    ...
  },
  "simulation": {
    "time_steps": 10080,
    "enable_k_adaptation": true,
    ...
  },
  "scheduler": {
    "scheduler_type": "DurationAwareLyapunovScheduler",
    ...
  }
}
```

## 配置约定

### 单位约定

- **能量**: 焦耳 (J)
- **距离**: 米 (m)
- **时间**: 分钟 (min)
- **电池容量**: mAh（参考值，用于能量计算）

### 重要参数说明

- **K值**: 控制每个接收端可匹配的捐能者数量上限
  - `fixed_k`: 固定K值（当不使用自适应时）
  - `initial_K`: 初始K值（自适应起点）
  - `K_max`: K的上限（避免过度并行）

- **能量阈值**:
  - `low_threshold`: 低能量阈值（0~1，相对容量），用于判定需要能量的节点
  - `high_threshold`: 高能量阈值（0~1，相对容量），用于判定富余能量的节点

- **传输时长**（DurationAwareLyapunovScheduler）:
  - `duration_min`: 最小传输时长（分钟）
  - `duration_max`: 最大传输时长（分钟）
  - 当传输时长 > 1 时，节点锁定机制自动启用

## 文件结构

```
config/
├── __pycache__/
└── simulation_config.py    # 配置类定义和ConfigManager
```

## 相关文档

- [快速启动指南](../../docs/快速启动指南.md)
- [配置参数说明](../../docs/能耗计算验证报告.md)

