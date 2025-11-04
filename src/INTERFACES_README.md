# Interfaces 模块

## 概述

`interfaces.py` 定义了无线传感器网络仿真的核心接口，采用抽象基类（ABC）设计，提高代码可维护性和可扩展性。

## 设计原则

- **接口隔离**：每个接口只定义必要的抽象方法
- **依赖倒置**：高层模块依赖接口而非具体实现
- **开闭原则**：对扩展开放，对修改关闭

## 主要接口

### 1. 数据类（Dataclass）

#### EnergyTransferPlan

能量传输计划数据类。

**字段：**
- `donor_id`: 捐能者ID
- `receiver_id`: 接收者ID
- `path`: 节点ID路径（列表）
- `distance`: 路径距离
- `energy_sent`: 发送能量
- `energy_delivered`: 送达能量
- `energy_loss`: 损耗能量
- `efficiency`: 传输效率

#### SimulationStats

仿真统计信息数据类。

**字段：**
- `time_step`: 时间步
- `pre_std`: 传输前能量标准差
- `post_std`: 传输后能量标准差
- `delivered_total`: 总送达能量
- `total_loss`: 总损耗能量
- `k_value`: K值
- `active_plans`: 活跃计划数

### 2. 节点接口

#### INode

传感器节点接口。

**主要方法：**
- `get_id()`: 获取节点ID
- `get_position()`: 获取节点位置
- `get_current_energy()`: 获取当前能量
- `get_energy_capacity()`: 获取能量容量
- `has_solar()`: 是否有太阳能
- `is_mobile()`: 是否可移动
- `distance_to()`: 计算到其他节点的距离
- `energy_transfer_efficiency()`: 计算能量传输效率
- `energy_consumption()`: 计算能量消耗
- `energy_harvest()`: 计算能量采集
- `update_energy()`: 更新能量状态
- `update_position()`: 更新位置

### 3. 网络接口

#### INetwork

网络接口。

**主要方法：**
- `get_nodes()`: 获取所有节点
- `get_node_by_id()`: 根据ID获取节点
- `get_num_nodes()`: 获取节点数量
- `update_network_energy()`: 更新网络能量状态
- `execute_energy_transfer()`: 执行能量传输

### 4. 调度器接口

#### IScheduler

调度器接口。

**主要方法：**
- `plan()`: 制定能量传输计划
- `post_step()`: 步骤后处理
- `get_name()`: 获取调度器名称

### 5. 路由器接口

#### IRouter

路由器接口。

**主要方法：**
- `find_path()`: 查找路径
- `get_name()`: 获取路由器名称

### 6. 仿真器接口

#### ISimulator

仿真器接口。

**主要方法：**
- `run_simulation()`: 运行仿真
- `get_stats()`: 获取统计信息
- `save_results()`: 保存结果

### 7. 日志接口

#### ILogger

日志接口。

**主要方法：**
- `info()`: 信息日志
- `warning()`: 警告日志
- `error()`: 错误日志
- `debug()`: 调试日志

### 8. 配置接口

#### IConfigProvider

配置提供者接口。

**主要方法：**
- `get_node_config()`: 获取节点配置
- `get_network_config()`: 获取网络配置
- `get_simulation_config()`: 获取仿真配置
- `get_scheduler_config()`: 获取调度器配置

### 9. 数据收集接口

#### IDataCollector

数据收集器接口。

**主要方法：**
- `collect_node_data()`: 收集节点数据
- `collect_network_data()`: 收集网络数据
- `collect_simulation_data()`: 收集仿真数据
- `save_data()`: 保存数据

### 10. 绘图接口

#### IPlotter

绘图器接口。

**主要方法：**
- `plot_node_distribution()`: 绘制节点分布
- `plot_energy_over_time()`: 绘制能量随时间变化
- `plot_energy_paths()`: 绘制能量传输路径
- `plot_k_values()`: 绘制K值变化

### 11. 工厂接口

#### ISchedulerFactory

调度器工厂接口。

**主要方法：**
- `create_scheduler()`: 创建调度器
- `get_available_schedulers()`: 获取可用的调度器类型

#### IRouterFactory

路由器工厂接口。

**主要方法：**
- `create_router()`: 创建路由器
- `get_available_routers()`: 获取可用的路由器类型

### 12. 事件系统接口

#### IEventBus

事件总线接口。

**主要方法：**
- `subscribe()`: 订阅事件
- `unsubscribe()`: 取消订阅
- `publish()`: 发布事件

#### IEventHandler

事件处理器接口。

**主要方法：**
- `handle()`: 处理事件

## 使用示例

### 示例1：实现节点接口

```python
from interfaces import INode

class MyNode(INode):
    def __init__(self, node_id, position, energy):
        self.node_id = node_id
        self.position = position
        self.current_energy = energy
    
    def get_id(self):
        return self.node_id
    
    def get_position(self):
        return self.position
    
    # ... 实现其他接口方法
```

### 示例2：实现调度器接口

```python
from interfaces import IScheduler, EnergyTransferPlan

class MyScheduler(IScheduler):
    def plan(self, network, time_step):
        # 制定传输计划
        plans = []
        # ... 实现计划制定逻辑
        return plans, []
    
    def post_step(self, network, time_step, stats):
        # 步骤后处理
        pass
    
    def get_name(self):
        return "MyScheduler"
```

## 接口继承关系

```
INode (节点接口)
  └── SensorNode (实现)
  └── InfoNode (实现)

INetwork (网络接口)
  └── Network (实现)

IScheduler (调度器接口)
  └── BaseScheduler (实现)
      ├── LyapunovScheduler
      ├── ClusterScheduler
      └── ...

IRouter (路由器接口)
  └── EETOR (实现)
  └── EEOR (实现)
```

## 设计优势

1. **可扩展性**：易于添加新的调度器、路由器等实现
2. **可测试性**：可以使用Mock对象进行单元测试
3. **可维护性**：接口定义清晰，便于理解和维护
4. **解耦合**：高层模块依赖接口而非具体实现

## 相关文档

- [快速启动指南](../docs/快速启动指南.md)

