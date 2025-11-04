# Core 模块

## 概述

`core` 模块是无线传感器网络仿真的核心模块，提供了网络、节点、能量仿真等基础功能。

## 主要组件

### 1. Network (network.py)

网络管理器，负责管理整个传感器网络。

**主要功能：**
- **网络拓扑生成和管理**：
  - 支持多种分布模式（uniform/random/energy_hole）
  - 支持能量空洞模式（非太阳能节点聚集分布）
  - 支持物理中心节点（ID=0，高能量，不参与WET）
  
- **节点创建和初始化**：
  - 自动创建传感器节点
  - 支持太阳能节点比例配置
  - 支持移动节点（可选）
  - 支持能量分配模式（uniform/center_decreasing）
  
- **距离矩阵计算**（支持GPU加速）：
  - 自动计算节点间距离矩阵
  - 支持GPU加速（CuPy）或CPU计算
  - 距离矩阵缓存和复用
  
- **能量传输执行**：
  - 执行单跳和多跳能量传输
  - 支持传输时长（duration）配置
  - 自动计算传输效率、损耗和能耗
  - 更新节点能量状态和历史记录
  
- **物理中心节点管理**：
  - 物理中心节点（ID=0）不参与WET
  - 作为信息上报目标
  - 支持高初始能量配置
  
- **能量空洞模式支持**：
  - 非太阳能节点聚集在某个中心附近
  - 形成能量空洞区域
  - 用于测试能量均衡算法

**关键特性：**
- 支持多种节点分布模式（uniform/random/energy_hole）
- 支持物理中心节点（ID=0，高能量，不参与WET）
- 支持能量分配模式（uniform/center_decreasing）
- 支持GPU加速的距离计算
- 支持移动节点和位置历史记录

**使用方法：**
```python
from core.network import Network

network = Network(
    num_nodes=30,
    low_threshold=0.1,
    high_threshold=0.9,
    node_initial_energy=40000.0,
    max_hops=3,
    distribution_mode="random",
    network_area_width=5.0,
    network_area_height=5.0,
    enable_physical_center=True
)

# 获取节点
nodes = network.get_nodes()
physical_center = network.get_physical_center()

# 执行能量传输
plans = [...]  # 能量传输计划列表
network.execute_energy_transfer(plans)
```

### 2. SensorNode (SensorNode.py)

传感器节点类，代表网络中的单个节点。

**主要功能：**
- 能量状态管理（当前能量、能量历史）
- 能量采集计算（太阳能）
- 能量消耗计算（通信、传感、维护）
- 能量传输效率计算
- 位置管理（支持移动节点）
- 能量传输记录（接收/发送历史）

**关键特性：**
- 支持太阳能能量采集
- 支持移动节点（位置历史记录）
- 支持能量传输效率计算（基于距离）
- 支持能量阈值判断（低能量/高能量）
- 支持物理中心节点标识

**能量模型：**
- **能量采集**：基于太阳能辐照度模型
  - 采集能量 = `solar_efficiency × solar_area × irradiance × env_correction_factor`
  - 辐照度随时间变化（白天/夜晚周期）
  
- **能量消耗**：
  - 通信能耗：
    - 发送：`E_tx = E_elec × B + ε_amp × B × d^τ`
    - 接收：`E_rx = E_elec × B`
    - 总通信能耗：`(E_tx + E_rx) / 2 + E_sen`（平均化ACK往返）
  - 传感能耗：固定值 `E_sen`（每步）
  - 维护能耗：`E_decay`（每步，待机损耗）
  - WET传输能耗：`E_char × duration`（传输时长×单次传输能量）
  
- **能量传输效率**：
  - 距离≤1m：`η(d) = η_0 + (1 - η_0) × (1 - d)`（线性插值）
  - 距离>1m：`η(d) = η_0 / d^γ`（逆幂律衰减）
  - 路径总效率：`η_path = ∏ η(hop_i)`（各跳效率乘积）

### 3. EnergySimulation (energy_simulation.py)

能量仿真实例，负责运行整个仿真过程。

**主要功能：**
- 仿真主循环
- 能量传输触发判断（智能被动传能）
- K值自适应管理
- 统计信息收集
- 结果保存

**关键特性：**
- **智能被动传能模式**：
  - 基于能量方差阈值触发
  - 基于低能量节点比例触发
  - 基于预测性分析触发
  - 支持冷却期机制（避免频繁触发）
  
- **K值自适应**：
  - 动态调整并行捐能者数量
  - 基于奖励函数（能量均衡、送达量、损耗）
  - 支持滞回阈值（防止频繁调整）
  
- **固定K值模式**：
  - 不使用自适应时，使用固定K值
  - 提高仿真速度和稳定性
  
- **前瞻模拟**（可选）：
  - 短期模拟评估不同K值效果
  - 辅助K值调整决策
  - 增加计算开销，可选启用
  
- **GPU加速**：
  - 支持GPU加速统计计算
  - 自动检测CuPy可用性
  - 无缝切换CPU/GPU计算

**仿真流程：**
1. 初始化网络和调度器
2. 对每个时间步：
   - 更新网络能量状态（采集、消耗、衰减）
   - 判断是否触发能量传输（智能被动传能或定时触发）
   - 如果触发：制定传输计划 → 执行传输 → 收集信息 → 更新统计
   - 如果启用K值自适应：根据统计信息调整K值
3. 生成最终统计和结果

**使用方法：**
```python
from core.energy_simulation import EnergySimulation

simulation = EnergySimulation(
    network=network,
    time_steps=10080,  # 7天（分钟）
    scheduler=scheduler,
    enable_k_adaptation=True,
    initial_K=1,
    K_max=24,
    passive_mode=True
)

# 运行仿真
stats = simulation.simulate()

# 获取统计信息
final_stats = simulation.stats.get_final_stats()
```

### 4. EnergyManagement (energy_management.py)

能量管理工具函数。

**主要功能：**
- `get_neighbors_for_node()`: 获取节点的邻居列表（基于距离阈值）
- `balance_energy()`: 能量平衡函数（当前已禁用）
- `handle_low_energy()`: 处理低能量节点
- `handle_high_energy()`: 处理高能量节点

**注意：** `balance_energy()` 函数当前已临时禁用，避免非太阳能节点意外获取能量。该函数原本用于自动平衡网络能量，但可能导致能量守恒问题。

### 5. SimulationStats (simulation_stats.py)

仿真统计和可视化管理器。

**主要功能：**
- 单步统计计算（能量标准差、传输量、损耗等）
- 统计信息累积和记录
- 最终统计汇总
- 支持GPU加速统计计算

**统计指标：**
- `pre_std`: 传输前的能量标准差
- `post_std`: 传输后的能量标准差
- `delivered_total`: 总送达能量
- `total_loss`: 总损耗能量
- `k_value`: 当前K值
- `active_plans`: 活跃传输计划数

### 6. ResultManager (result_manager.py)

结果记录和保存管理器。

**主要功能：**
- 节点能量状态记录
- 仿真结果保存到CSV
- 结果文件管理

## 文件结构

```
core/
├── __pycache__/
├── energy_management.py    # 能量管理工具函数
├── energy_simulation.py    # 能量仿真实例
├── network.py              # 网络管理器
├── result_manager.py       # 结果管理器
├── SensorNode.py           # 传感器节点类
└── simulation_stats.py      # 仿真统计管理器
```

## 数据流

1. **网络初始化**：
   - `Network` 创建节点列表
   - 初始化节点位置和能量
   - 设置物理中心节点

2. **仿真循环**：
   - `EnergySimulation` 检查是否触发能量传输
   - `Scheduler` 制定传输计划
   - `Network` 执行能量传输
   - `SimulationStats` 记录统计信息
   - `KAdaptationManager` 调整K值（如果启用）

3. **结果保存**：
   - `SimulationStats` 生成最终统计
   - `ResultManager` 保存结果到CSV

## 关键接口

### Network接口
- `get_nodes()`: 获取所有节点
- `get_node_by_id(node_id)`: 根据ID获取节点
- `get_physical_center()`: 获取物理中心节点
- `execute_energy_transfer(plans)`: 执行能量传输
- `update_network_energy(time_step)`: 更新网络能量状态

### SensorNode接口
- `get_current_energy()`: 获取当前能量
- `energy_harvest(time_step)`: 计算能量采集
- `energy_consumption(target_node, transfer_wet)`: 计算能量消耗
- `energy_transfer_efficiency(target_node)`: 计算传输效率
- `distance_to(other_node)`: 计算到其他节点的距离

### EnergySimulation接口
- `simulate()`: 运行仿真
- `K`: 当前K值（属性）
- `stats`: 统计管理器（属性）

## 相关文档

- [能量管理逻辑说明](../../docs/能量管理逻辑说明.md)
- [物理中心节点改造总结](../../docs/物理中心节点改造总结.md)
- [能耗计算验证报告](../../docs/能耗计算验证报告.md)

