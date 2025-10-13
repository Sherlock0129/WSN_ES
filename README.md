# 无线传感器网络能量传输仿真系统

## 项目概述

本项目是一个无线传感器网络（WSN）能量传输仿真系统，用于研究和模拟传感器节点之间的无线能量传输（WET）过程。系统支持多种能量调度算法、路由协议和可视化功能。

## 项目架构

### 📁 目录结构

```
WSN_ES/
├── src/                          # 源代码目录
│   ├── config/                   # 配置管理
│   │   ├── simulation_config.py  # 仿真配置类
│   │   └── default_config.json   # 默认配置文件
│   ├── core/                     # 核心模块
│   │   ├── SensorNode.py         # 传感器节点模型
│   │   ├── Network.py            # 网络拓扑管理
│   │   ├── energy_simulation.py  # 能量仿真实例
│   │   └── energy_management.py  # 能量管理
│   ├── scheduling/               # 调度层
│   │   ├── schedulers.py         # 能量调度算法
│   │   └── lookahead.py          # 前瞻调度
│   ├── routing/                  # 路由层
│   │   ├── opportunistic_routing.py # 机会路由
│   │   ├── EEOR.py               # 能量效率优化路由
│   │   └── adcr_link_layer.py    # ADCR链路层
│   ├── viz/                      # 可视化层
│   │   └── plotter.py            # 数据可视化
│   ├── utils/                    # 工具层
│   │   ├── parameters.py         # 参数管理
│   │   ├── utils.py              # 通用工具
│   │   └── error_handling.py     # 错误处理
│   ├── sim/                      # 仿真层
│   │   ├── main.py               # 主仿真程序
│   │   └── parallel_main.py      # 并行仿真
│   └── interfaces.py             # 接口定义
├── tests/                        # 测试目录
│   └── test_framework.py         # 测试框架
├── data/                         # 数据目录
├── logs/                         # 日志目录
└── README.md                     # 项目说明
```

### 🏗️ 架构层次

1. **配置层 (Config Layer)**
   - 统一管理所有仿真参数
   - 支持JSON配置文件
   - 提供参数验证功能

2. **接口层 (Interface Layer)**
   - 定义清晰的模块接口
   - 提高代码可维护性
   - 支持依赖注入

3. **核心层 (Core Layer)**
   - 传感器节点模型
   - 网络拓扑管理
   - 能量仿真引擎

4. **算法层 (Algorithm Layer)**
   - 能量调度算法
   - 路由协议
   - 优化策略

5. **工具层 (Utility Layer)**
   - 错误处理
   - 日志记录
   - 性能监控

6. **可视化层 (Visualization Layer)**
   - 数据图表
   - 网络拓扑图
   - 性能分析图

## 核心功能

### 🔋 能量管理

- **太阳能采集**: 基于时间的光照模型
- **能量衰减**: 电池自然损耗
- **无线能量传输**: 支持多跳传输
- **能量均衡**: 自动平衡节点能量

### 📡 网络拓扑

- **节点分布**: 支持均匀分布和随机分布
- **移动节点**: 支持圆形、直线、振荡运动模式
- **太阳能节点**: 可配置太阳能采集能力
- **邻居发现**: 基于距离的邻居关系建立

### 🚀 调度算法

1. **LyapunovScheduler**: 基于Lyapunov优化的调度
2. **ClusterScheduler**: 基于簇的调度
3. **PredictionScheduler**: 基于预测的调度
4. **PowerControlScheduler**: 功率控制调度
5. **BaselineHeuristic**: 基线启发式调度

### 🛣️ 路由协议

1. **Opportunistic Routing**: 机会路由
2. **EEOR**: 能量效率优化路由
3. **ADCR**: 自适应动态协作路由

### 📊 可视化功能

- 节点分布图
- 能量传输路径图
- 能量随时间变化图
- K值动态调整图
- 性能统计图表

## 快速开始

### 环境要求

- Python 3.7+
- NumPy
- Matplotlib
- Plotly
- Pandas

### 安装依赖

```bash
pip install numpy matplotlib plotly pandas
```

### 基本使用

```python
from src.config.simulation_config import ConfigManager
from src.core.Network import Network
from src.core.energy_simulation import EnergySimulation
from src.scheduling.schedulers import LyapunovScheduler

# 1. 加载配置
config = ConfigManager("src/config/default_config.json")

# 2. 创建网络
network_config = config.get_network_config_dict()
network = Network(network_config["num_nodes"], network_config)

# 3. 创建调度器
scheduler_params = config.get_scheduler_params()
scheduler = LyapunovScheduler(**scheduler_params)

# 4. 运行仿真
simulation = EnergySimulation(network, config.simulation_config.time_steps, scheduler)
simulation.simulate()

# 5. 查看结果
simulation.print_statistics()
```

### 配置自定义仿真

```python
# 修改配置文件
config.node_config.initial_energy = 50000.0
config.network_config.num_nodes = 50
config.scheduler_config.scheduler_type = "ClusterScheduler"

# 保存配置
config.save_to_file("custom_config.json")
```

## API 文档

### ConfigManager

配置管理器，统一管理所有仿真参数。

#### 主要方法

- `load_from_file(config_file: str)`: 从JSON文件加载配置
- `save_to_file(config_file: str)`: 保存配置到JSON文件
- `get_network_config_dict() -> Dict`: 获取网络配置字典
- `get_scheduler_params() -> Dict`: 获取调度器参数

#### 配置类

- `NodeConfig`: 节点配置参数
- `NetworkConfig`: 网络配置参数
- `SimulationConfig`: 仿真配置参数
- `SchedulerConfig`: 调度器配置参数

### SensorNode

传感器节点模型，实现INode接口。

#### 主要方法

- `get_id() -> int`: 获取节点ID
- `get_position() -> Tuple[float, float]`: 获取节点位置
- `get_current_energy() -> float`: 获取当前能量
- `distance_to(other_node: INode) -> float`: 计算到其他节点的距离
- `energy_transfer_efficiency(target_node: INode) -> float`: 计算能量传输效率
- `energy_harvest(time_step: int) -> float`: 计算能量采集
- `update_energy(time_step: int) -> Tuple[float, float]`: 更新能量状态

### Network

网络拓扑管理器，实现INetwork接口。

#### 主要方法

- `get_nodes() -> List[INode]`: 获取所有节点
- `get_node_by_id(node_id: int) -> Optional[INode]`: 根据ID获取节点
- `update_network_energy(time_step: int)`: 更新网络能量状态
- `execute_energy_transfer(plans: List[EnergyTransferPlan])`: 执行能量传输

### IScheduler

调度器接口，定义能量传输计划制定方法。

#### 主要方法

- `plan(network: INetwork, time_step: int) -> Tuple[List[EnergyTransferPlan], List[Dict]]`: 制定能量传输计划
- `post_step(network: INetwork, time_step: int, stats: SimulationStats)`: 步骤后处理

### EnergySimulation

能量仿真实例，管理整个仿真过程。

#### 主要方法

- `simulate()`: 运行仿真
- `print_statistics()`: 打印统计信息
- `save_results(filename: str)`: 保存结果到文件

## 扩展开发

### 添加新的调度算法

1. 继承`IScheduler`接口
2. 实现`plan`和`post_step`方法
3. 在`SchedulerFactory`中注册

```python
class MyScheduler(IScheduler):
    def plan(self, network: INetwork, time_step: int):
        # 实现调度逻辑
        pass
    
    def post_step(self, network: INetwork, time_step: int, stats: SimulationStats):
        # 实现后处理逻辑
        pass
```

### 添加新的路由协议

1. 继承`IRouter`接口
2. 实现`find_path`方法
3. 在`RouterFactory`中注册

### 添加新的可视化功能

1. 继承`IPlotter`接口
2. 实现绘图方法
3. 在可视化模块中集成

## 测试

### 运行测试

```bash
cd tests
python test_framework.py
```

### 测试覆盖

- 单元测试：测试各个模块的功能
- 集成测试：测试模块间的协作
- 性能测试：测试系统性能
- 错误处理测试：测试异常情况

## 日志和调试

### 日志级别

- `DEBUG`: 详细调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

### 日志文件

- `logs/simulation_YYYYMMDD.log`: 完整日志
- `logs/errors_YYYYMMDD.log`: 错误日志

### 性能监控

系统提供性能监控功能，可以监控：
- 函数执行时间
- 内存使用情况
- 网络性能指标

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 参与讨论

---

**注意**: 这是一个重构中的项目，部分功能可能仍在开发中。请参考最新的代码和文档获取最新信息。
