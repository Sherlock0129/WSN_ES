# Info Collection 模块

## 概述

`info_collection` 模块负责节点信息的收集、管理和更新。提供基于路径的信息收集机制和ADCR链路层功能。

## 主要组件

### 1. PathBasedInfoCollector (path_based_collector.py)

基于路径的机会主义信息收集器。

**核心功能：**
- 利用能量传输路径收集节点信息（piggyback，搭载）
- 路径内节点实时采集最新信息
- Receiver（路径最后一个节点）作为信息汇聚点上报到物理中心节点
- 支持机会主义信息传递机制
- 支持延迟上报和批量上报

**关键特性：**
- **零通信开销**：信息搭载在传能路径上
- **高更新频率**：每次传能都更新
- **信息新鲜度高**：路径节点为实时采集
- **能量消耗可配置**：free模式（零能耗）或full模式（完全真实）

**工作流程：**
1. 路径节点（a、b、c）各自收集信息，沿路径聚合到终点c
2. 终点c将路径节点的聚合信息上报到物理中心节点（ID=0）
3. 虚拟中心（节点信息表管理器）只更新路径节点（a、b、c）的信息
4. 非路径节点不做任何处理（不收集、不估算、不更新）

**能量模型：**
- **free模式**：零能耗（信息完全搭载）
- **full模式**：路径逐跳 + 上报跳（→物理中心节点）都消耗能量

**使用方法：**
```python
from info_collection.path_based_collector import PathBasedInfoCollector
from info_collection.physical_center import VirtualCenter

# 创建虚拟中心
virtual_center = VirtualCenter(...)

# 创建路径信息收集器
collector = PathBasedInfoCollector(
    virtual_center=virtual_center,
    physical_center=physical_center,
    energy_mode="free",  # 或 "full"
    enable_opportunistic_info_forwarding=True,
    enable_delayed_reporting=True
)

# 在能量传输后收集信息
collector.collect_path_info(plans, time_step)
```

### 2. NodeInfoManager (physical_center.py)

节点信息管理器，维护全网节点信息表。

**核心功能：**
- 三级缓存架构（L1:字典 L2:deque L3:CSV）
- 节点信息查询和统计
- 支持可视化数据导出
- 管理物理中心的位置信息（固定不变）

**信息表结构：**
```python
{
    node_id: {
        'energy': float,           # 节点能量（可能是估算值）
        'record_time': int,        # 信息记录时间（采集时刻）
        'arrival_time': int,       # 到达物理中心的时间
        'position': (x, y),        # 节点位置
        'is_solar': bool,          # 是否有太阳能
        'cluster_id': int,         # 所属簇ID
        'data_size': int,          # 数据包大小
        'aoi': int,               # Age of Information（信息年龄）
        'is_estimated': bool,      # 是否为估算值
        't': int,                  # 全局时间戳
        'info_volume': int,        # 累积的信息量（bits）
        'info_waiting_since': int, # 开始等待的时间戳
        'info_is_reported': bool,  # 是否已上报
    }
}
```

**关键特性：**
- **高性能**：三级缓存架构（L1:字典 L2:deque L3:CSV）
- **轻量级**：与物理中心节点实体（SensorNode）解耦
- **单一职责**：只负责信息表管理，不持有节点实体

### 3. InfoNode (info_node.py)

信息节点类，基于物理中心节点信息表的轻量级节点。

**核心功能：**
- 提供与SensorNode相同的接口
- 用于调度器和路由算法
- 所有计算公式与SensorNode完全一致

**关键特性：**
- 只包含调度和路由需要的属性和方法
- 所有计算公式与SensorNode完全一致
- 数据来源于物理中心的节点信息表

### 4. ADCRLinkLayer (adcr_link_layer.py)

ADCR链路层算法实现（ADCRLinkLayerVirtual类）。

**核心功能：**
- **聚类算法**：类似LEACH的分布式聚类
  - 估计最优簇数 K*（基于近邻统计启发式）
  - 能量感知 + 空间抑制的簇头选择
  - 成簇与一次性细化
  
- **路径规划**：为簇头到虚拟中心规划真实节点路径
  - 使用opportunistic_routing或eetor_find_path_adaptive
  - 支持直接传输优化（当直接传输能耗不超过阈值时）
  
- **能耗结算**：对通信过程执行能耗结算
  - 路径逐跳通信能耗
  - 信息聚合能耗
  - 支持动态数据量（基于簇大小）
  
- **可视化**：提供Plotly可视化功能
  - 节点分布图
  - 簇结构图
  - 路径图
  - 虚拟中心标记

**关键特性：**
- **轮换机制**：支持定期重聚类（round_period）
- **路径规划**：可配置是否规划路径（plan_paths）
- **能耗结算**：可配置是否消耗能量（consume_energy）
- **直接传输优化**：当直接传输能耗不超过锚点传输的110%时，使用直接传输
- **自动画图**：可配置每次重聚类后自动画图（auto_plot）

**关键参数：**
- `round_period`: 重聚类周期（分钟，默认1440=1天）
- `r_neighbor`: 邻居检测半径（默认1.732）
- `r_min_ch`: 簇头间最小距离（默认1.0）
- `c_k`: K值估计系数（默认1.2）
- `max_hops`: 最大多跳数（默认5）
- `distance_weight`: 距离权重（默认1.0）
- `energy_weight`: 能量权重（默认0.2）
- `enable_direct_transmission_optimization`: 是否启用直接传输优化（默认True）
- `direct_transmission_threshold`: 直接传输阈值（默认0.1，即110%）

**使用方法：**
```python
from info_collection.adcr_link_layer import ADCRLinkLayerVirtual

adcr = ADCRLinkLayerVirtual(
    network=network,
    round_period=1440,
    r_neighbor=1.732,
    plan_paths=True,
    consume_energy=True,
    enable_direct_transmission_optimization=True
)

# 在仿真步骤中调用
adcr.update(t=current_time)
```

## 文件结构

```
info_collection/
├── __pycache__/
├── __init__.py
├── adcr_link_layer.py      # ADCR链路层算法
├── info_node.py            # 信息节点类
├── path_based_collector.py # 基于路径的信息收集器
└── physical_center.py       # 节点信息管理器（物理中心）
```

## 数据流

1. **信息收集**：
   - 能量传输路径上的节点收集信息
   - 信息沿路径聚合到终点（Receiver）
   - Receiver上报到物理中心节点

2. **信息管理**：
   - NodeInfoManager更新节点信息表
   - 三级缓存架构保证高性能
   - 支持信息查询和统计

3. **信息使用**：
   - 调度器从NodeInfoManager获取InfoNode
   - 路由算法使用InfoNode进行路径规划
   - 所有计算与真实节点保持一致

## 相关文档

- [路径信息收集器使用说明](../../docs/路径信息收集器使用说明.md)
- [路径信息收集器实现完成](../../docs/路径信息收集器实现完成.md)
- [机会主义信息传递机制设计](../../docs/机会主义信息传递机制设计.md)
- [虚拟中心节点信息表使用说明](../../docs/虚拟中心节点信息表使用说明.md)

